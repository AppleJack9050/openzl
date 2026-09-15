# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Read the compression traces written by `zli compress --trace`.

zli records the same stream graph in two forms:

* the CBOR file written to the ``--trace`` path, whose schema is described in
  ``tools/visualization_app/src/interfaces``;
* the DOT text printed to stdout while tracing
  (``ChunkTrace::printCodecMetadata``).

Both are normalized into a :class:`Trace`: a list of :class:`Chunk` objects,
each holding the streams, codecs and graphs of one graph run. A segmented
compression has the top-level run (start -> segmenter) as chunk 0 and one
chunk per segment after it; an unsegmented one has a single chunk.
"""

from __future__ import annotations

import dataclasses
import re
import struct
import zlib
from typing import Dict, List, Optional, Tuple

STREAM_TYPES = {1: "serial", 2: "struct", 4: "numeric", 8: "string"}
DOT_STREAM_TYPES = {
    "Serialized": "serial",
    "Fixed_Width": "struct",
    "Numeric": "numeric",
    "Variable_Size": "string",
}
GRAPH_TYPES = [
    "Standard",
    "Static",
    "Selector",
    "Function",
    "Multiple_Input",
    "Parameterized",
    "Segmenter",
]


MAX_TRACE_BYTES = 1 << 30
"""The largest trace accepted, measured after gzip decompression."""


class TraceFormatError(ValueError):
    """The input is not a trace this tool can read."""


@dataclasses.dataclass
class Params:
    ints: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    # (paramId, size in bytes); the bytes themselves are not kept
    copies: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    refs: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Stream:
    id: int
    type: str
    output_index: int
    width: int
    elements: int
    csize: int
    """Compressed cost of the stream and everything downstream of it."""
    content_size: Optional[int] = None
    """Raw size; only the CBOR trace records it."""

    @property
    def raw_size(self) -> Optional[int]:
        if self.content_size is not None:
            return self.content_size
        if self.type == "string" or not self.width:
            return None
        return self.width * self.elements


@dataclasses.dataclass
class Codec:
    id: int
    name: str
    header_size: int = 0
    standard: bool = True
    params: Params = dataclasses.field(default_factory=Params)
    inputs: List[int] = dataclasses.field(default_factory=list)
    outputs: List[int] = dataclasses.field(default_factory=list)
    failure: Optional[str] = None
    graph: Optional[int] = None


@dataclasses.dataclass
class Graph:
    id: int
    name: str
    type: str
    params: Params = dataclasses.field(default_factory=Params)
    codecs: List[int] = dataclasses.field(default_factory=list)
    failure: Optional[str] = None


@dataclasses.dataclass
class Chunk:
    index: int
    streams: List[Stream]
    codecs: List[Codec]
    graphs: List[Graph]


@dataclasses.dataclass
class Trace:
    format: str
    """"cbor" or "dot"."""
    chunks: List[Chunk]
    trace_version: Optional[int] = None


# ---------------------------------------------------------------------------
# CBOR
# ---------------------------------------------------------------------------

_BREAK = object()


class _CborReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _take(self, n: int) -> int:
        start = self.pos
        if start + n > len(self.data):
            raise TraceFormatError("The CBOR data ends early.")
        self.pos += n
        return start

    def _argument(self, info: int) -> Optional[int]:
        if info < 24:
            return info
        if info == 24:
            return self.data[self._take(1)]
        if info == 25:
            return struct.unpack_from(">H", self.data, self._take(2))[0]
        if info == 26:
            return struct.unpack_from(">I", self.data, self._take(4))[0]
        if info == 27:
            return struct.unpack_from(">Q", self.data, self._take(8))[0]
        if info == 31:
            return None  # indefinite length
        raise TraceFormatError("The CBOR data is malformed.")

    def _items_until_break(self) -> List[object]:
        items = []
        while True:
            item = self.read()
            if item is _BREAK:
                return items
            items.append(item)

    def read(self) -> object:
        initial = self.data[self._take(1)]
        major, info = initial >> 5, initial & 31
        if major == 0:
            return self._argument(info)
        if major == 1:
            return -1 - self._argument(info)
        if major in (2, 3):
            length = self._argument(info)
            if length is None:
                parts = self._items_until_break()
                return b"".join(parts) if major == 2 else "".join(parts)
            start = self._take(length)
            raw = self.data[start : start + length]
            return bytes(raw) if major == 2 else raw.decode("utf-8", "replace")
        if major == 4:
            length = self._argument(info)
            if length is None:
                return self._items_until_break()
            return [self.read() for _ in range(length)]
        if major == 5:
            length = self._argument(info)
            result: Dict[object, object] = {}
            if length is None:
                while True:
                    key = self.read()
                    if key is _BREAK:
                        return result
                    result[_hashable(key)] = self.read()
            for _ in range(length):
                key = self.read()
                result[_hashable(key)] = self.read()
            return result
        if major == 6:
            self._argument(info)  # tags carry no meaning in traces
            return self.read()
        if info == 20:
            return False
        if info == 21:
            return True
        if info in (22, 23):
            return None
        if info == 24:
            return self.data[self._take(1)]
        if info == 25:
            return struct.unpack_from(">e", self.data, self._take(2))[0]
        if info == 26:
            return struct.unpack_from(">f", self.data, self._take(4))[0]
        if info == 27:
            return struct.unpack_from(">d", self.data, self._take(8))[0]
        if info == 31:
            return _BREAK
        return info


def _hashable(key: object) -> object:
    try:
        hash(key)
        return key
    except TypeError:
        return repr(key)


def decode_cbor(data: bytes) -> object:
    """Decode one CBOR item (RFC 8949) from ``data``."""
    reader = _CborReader(data)
    item = reader.read()
    if item is _BREAK:
        raise TraceFormatError("The CBOR data is malformed.")
    return item


def _params_from_cbor(obj: object) -> Params:
    if not isinstance(obj, dict):
        return Params()
    params = Params()
    for p in obj.get("intParams") or []:
        params.ints.append((int(p.get("paramId", 0)), int(p.get("paramValue", 0))))
    for p in obj.get("copyParams") or []:
        size = p.get("paramSize")
        if size is None:
            size = len(p.get("paramData") or b"")
        params.copies.append((int(p.get("paramId", 0)), int(size)))
    for p in obj.get("refParams") or []:
        params.refs.append(int(p.get("paramId", 0)))
    return params


MAX_MISSING_STREAMS = 1 << 16


def _placeholder_stream(sid: int) -> Stream:
    return Stream(id=sid, type="unknown", output_index=0, width=0, elements=0, csize=0)


def _check_chunk(chunk: Chunk) -> Chunk:
    """Validate references, fill in unrecorded streams, attach graphs to codecs."""
    referenced = [sid for c in chunk.codecs for sid in c.inputs + c.outputs]
    if any(sid < 0 for sid in referenced):
        raise TraceFormatError(f"Chunk {chunk.index} refers to a negative stream id.")
    highest = max(referenced, default=-1)
    if highest >= len(chunk.streams):
        if highest - len(chunk.streams) > MAX_MISSING_STREAMS:
            raise TraceFormatError(
                f"Chunk {chunk.index} refers to stream {highest}, but the trace has "
                f"only {len(chunk.streams)} streams."
            )
        # The tracer records input streams when the first codec starts, so a
        # compression that failed before any codec ran (e.g. a graph failing in
        # strict mode) references streams it never recorded.
        chunk.streams += [
            _placeholder_stream(i) for i in range(len(chunk.streams), highest + 1)
        ]
    produced = {sid for c in chunk.codecs for sid in c.outputs}
    start = next(
        (c for c in chunk.codecs if not c.inputs and c.name.endswith("#start")), None
    )
    if start is not None:
        consumed = {sid for c in chunk.codecs for sid in c.inputs}
        start.outputs += [
            sid
            for sid in range(len(chunk.streams))
            if sid in consumed and sid not in produced
        ]
    for graph in chunk.graphs:
        for cid in graph.codecs:
            if not 0 <= cid < len(chunk.codecs):
                raise TraceFormatError(
                    f"Chunk {chunk.index}: graph {graph.id} ({graph.name}) refers "
                    f"to codec {cid}, which is not in the trace."
                )
            chunk.codecs[cid].graph = graph.id
    return chunk


def trace_from_cbor(obj: object) -> Trace:
    """Build a Trace from the decoded CBOR object of a zli trace file."""
    if not isinstance(obj, dict):
        raise TraceFormatError("This CBOR file is not a zli trace.")
    raw_chunks = obj.get("chunks")
    version = obj.get("traceVersion")
    if not isinstance(raw_chunks, list):
        if not (
            isinstance(obj.get("streams"), list) and isinstance(obj.get("codecs"), list)
        ):
            raise TraceFormatError(
                "This CBOR file has no chunks, streams or codecs, so it is not a zli trace."
            )
        # Trace format version 0: one chunk and no start node in front of stream 0.
        start = {"name": "zl.#start", "inputStreams": [], "outputStreams": [0]}
        graphs = [
            {**g, "codecIDs": [i + 1 for i in g.get("codecIDs") or []]}
            for g in obj.get("graphs") or []
        ]
        raw_chunks = [
            {
                "streams": obj["streams"],
                "codecs": [start] + obj["codecs"],
                "graphs": graphs,
            }
        ]
        version = 0
    chunks = []
    for index, ch in enumerate(raw_chunks):
        if not isinstance(ch, dict):
            raise TraceFormatError(f"Chunk {index} of the trace is not a map.")
        streams = [
            Stream(
                id=i,
                type=STREAM_TYPES.get(s.get("type"), "unknown"),
                output_index=int(s.get("outputIdx") or 0),
                width=int(s.get("eltWidth") or 0),
                elements=int(s.get("numElts") or 0),
                csize=int(s.get("cSize") or 0),
                content_size=(
                    int(s["contentSize"]) if s.get("contentSize") is not None else None
                ),
            )
            for i, s in enumerate(ch.get("streams") or [])
        ]
        codecs = [
            Codec(
                id=i,
                name=str(c.get("name") or ""),
                header_size=int(c.get("cHeaderSize") or 0),
                standard=bool(c.get("cType", True)),
                params=_params_from_cbor(c.get("cLocalParams")),
                inputs=[int(x) for x in c.get("inputStreams") or []],
                outputs=[int(x) for x in c.get("outputStreams") or []],
                failure=c.get("cFailureString") or None,
            )
            for i, c in enumerate(ch.get("codecs") or [])
        ]
        graphs = []
        for i, g in enumerate(ch.get("graphs") or []):
            gtype = g.get("gType")
            graphs.append(
                Graph(
                    id=i,
                    name=str(g.get("gName") or ""),
                    type=(
                        GRAPH_TYPES[gtype]
                        if isinstance(gtype, int) and 0 <= gtype < len(GRAPH_TYPES)
                        else str(gtype)
                    ),
                    params=_params_from_cbor(g.get("gLocalParams")),
                    codecs=[int(x) for x in g.get("codecIDs") or []],
                    failure=g.get("gFailureString") or None,
                )
            )
        chunks.append(_check_chunk(Chunk(index, streams, codecs, graphs)))
    if not chunks:
        raise TraceFormatError("The trace has no chunks.")
    return Trace("cbor", chunks, version if isinstance(version, int) else None)


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------

_STREAM_RE = re.compile(
    r'^S(\d+) \[shape=record, label="Stream:? ?\d*\\nType: #?(\w+)\\nOutputIdx: (\d+)'
    r'\\nEltWidth: (\d+)\\n#Elts: (\d+)\\nCSize: (\d+)\\nShare: [^"]*"\];'
)
_CODEC_RE = re.compile(
    r'^T(\d+) \[shape=Mrecord, label="(.*)\(ID: (-?\d+)\)\\n ?(\w+) transform \d+'
    r'\\n ?Header size: (\d+)(.*)"\];\s*$'
)
_EDGE_RE = re.compile(r'^([ST])(\d+) -> ([ST])(\d+)\s*\[label="#(\d+)"\];')
_LABEL_RE = re.compile(r'^label="(.*)";\s*$')
_PAIR_RE = re.compile(r"\((-?\d+), (-?\d+)\)")
_SINGLE_RE = re.compile(r"\((-?\d+)\)")


def _params_from_dot(parts: List[str]) -> Tuple[Params, Optional[str]]:
    params = Params()
    failure = None
    for part in parts:
        text = part.strip()
        if text.startswith("Failure:"):
            failure = text[len("Failure:") :].strip() or "failed"
        elif text.startswith("IntParams"):
            params.ints += [(int(a), int(b)) for a, b in _PAIR_RE.findall(text)]
        elif text.startswith("CopyParams"):
            params.copies += [(int(a), int(b)) for a, b in _PAIR_RE.findall(text)]
        elif text.startswith("RefParams"):
            params.refs += [int(a) for a in _SINGLE_RE.findall(text)]
    return params, failure


def _chunk_from_dot(lines: List[str]) -> Chunk:
    streams: Dict[int, Stream] = {}
    codecs: Dict[int, Codec] = {}
    graphs: List[Graph] = []
    edges: List[Tuple[str, int, str, int]] = []
    open_graph: Optional[Graph] = None
    expect_label = False
    for line in lines:
        if line.startswith("subgraph cluster_"):
            expect_label, open_graph = True, None
            continue
        if expect_label:
            m = _LABEL_RE.match(line)
            if m:
                parts = m.group(1).split("\\n")
                params, failure = _params_from_dot(parts[2:])
                gtype = parts[1][len("type=") :] if len(parts) > 1 else ""
                open_graph = Graph(len(graphs), parts[0], gtype, params, [], failure)
                graphs.append(open_graph)
                expect_label = False
                continue
        if line.strip() == "}":
            open_graph, expect_label = None, False
            continue
        m = _STREAM_RE.match(line)
        if m:
            sid = int(m.group(1))
            streams[sid] = Stream(
                id=sid,
                type=DOT_STREAM_TYPES.get(m.group(2), "unknown"),
                output_index=int(m.group(3)),
                width=int(m.group(4)),
                elements=int(m.group(5)),
                csize=int(m.group(6)),
            )
            continue
        m = _CODEC_RE.match(line)
        if m:
            cid = int(m.group(1))
            params, failure = _params_from_dot(m.group(6).split("\\n"))
            codecs[cid] = Codec(
                id=cid,
                name=m.group(2),
                header_size=int(m.group(5)),
                standard=m.group(4) == "Standard",
                params=params,
                failure=failure,
            )
            if open_graph is not None:
                open_graph.codecs.append(cid)
            continue
        m = _EDGE_RE.match(line)
        if m:
            edges.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4))))
    for src_kind, src, dst_kind, dst in edges:
        if src_kind == "T" and dst_kind == "S" and src in codecs:
            codecs[src].outputs.append(dst)
        elif src_kind == "S" and dst_kind == "T" and dst in codecs:
            codecs[dst].inputs.append(src)
    missing = sorted(set(range(len(codecs))) - set(codecs))
    if missing:
        raise TraceFormatError(f"The DOT text has no codec {missing[0]}.")
    chunk = Chunk(
        0,
        [
            streams.get(i) or _placeholder_stream(i)
            for i in range(max(streams, default=-1) + 1)
        ],
        [codecs[i] for i in range(len(codecs))],
        graphs,
    )
    return _check_chunk(chunk)


def trace_from_dot(text: str) -> Trace:
    """Build a Trace from the DOT text zli prints to stdout with --trace."""
    blocks: List[List[str]] = []
    for line in text.splitlines():
        if re.match(r"^digraph\b", line):
            blocks.append([])
        elif blocks:
            blocks[-1].append(line)
    if not blocks:
        raise TraceFormatError(
            "It is neither a CBOR trace nor DOT text (no `digraph` found). Use the "
            ".cbor file zli writes with --trace, or everything zli printed to stdout."
        )
    chunks = [c for c in (_chunk_from_dot(b) for b in blocks) if c.codecs]
    if not chunks:
        raise TraceFormatError("The DOT text has no codecs in it.")
    if len(chunks) > 1:
        # zli prints each chunk's graph when it finishes and the top-level graph
        # last; the CBOR file puts the top level first. Use the CBOR order.
        top = next(
            (i for i, c in enumerate(chunks) if _has_segmenter(c)), len(chunks) - 1
        )
        chunks.insert(0, chunks.pop(top))
    for index, chunk in enumerate(chunks):
        chunk.index = index
    return Trace("dot", chunks)


def _has_segmenter(chunk: Chunk) -> bool:
    return any(not c.outputs and "segmenter" in c.name.lower() for c in chunk.codecs)


def _too_large(limit: int) -> TraceFormatError:
    return TraceFormatError(f"The trace is larger than {limit:,} bytes.")


def _gunzip(data: bytes, limit: int) -> bytes:
    """Decompress every gzip member, stopping as soon as the output passes limit."""
    parts: List[bytes] = []
    size = 0
    while data[:2] == b"\x1f\x8b":
        member = zlib.decompressobj(wbits=31)
        pending = data
        while pending and not member.eof:
            piece = member.decompress(pending, limit + 1 - size)
            size += len(piece)
            if size > limit:
                raise _too_large(limit)
            parts.append(piece)
            pending = member.unconsumed_tail
        if not member.eof:
            raise TraceFormatError("The gzip data ends too early.")
        data = member.unused_data.lstrip(b"\x00")
    return b"".join(parts)


def load_trace_bytes(data: bytes, limit: int = MAX_TRACE_BYTES) -> Trace:
    """Read a trace from bytes: CBOR or DOT text, optionally gzip-compressed."""
    try:
        if data[:2] == b"\x1f\x8b":
            data = _gunzip(data, limit)
        elif len(data) > limit:
            raise _too_large(limit)
        start = re.match(rb"[ \t\r\n]*", data).end()
        if start == len(data):
            raise TraceFormatError("The file is empty.")
        # A CBOR trace is a single map (major type 5); anything else is DOT text.
        if 0xA0 <= data[start] <= 0xBF:
            return trace_from_cbor(decode_cbor(data[start:] if start else data))
        return trace_from_dot(data[start:].decode("utf-8", "replace"))
    except TraceFormatError:
        raise
    except zlib.error as e:
        raise TraceFormatError(f"The gzip data is damaged: {e}") from None
    except (
        AttributeError,
        TypeError,
        KeyError,
        IndexError,
        ValueError,
        struct.error,
    ) as e:
        # Well-formed CBOR or DOT whose values are not shaped like a zli trace.
        raise TraceFormatError(
            f"The data is not shaped like a zli trace ({type(e).__name__}: {e})."
        ) from None


def load_trace(path: str) -> Trace:
    with open(path, "rb") as f:
        return load_trace_bytes(f.read())
