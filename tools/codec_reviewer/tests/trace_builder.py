# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Build zli-shaped compression traces in memory for tests.

The builder mirrors what ``cpp/src/openzl/cpp/experimental/trace`` records:
a ``zl.#start`` codec producing the input streams, codecs with input and output
stream lists, ``zl.store`` for every stream that reaches the frame, and each
stream's compressed size computed like ``ChunkTrace::fillCSize``. It can emit
the CBOR file and the DOT text zli would print.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence, Tuple

SERIAL, STRUCT, NUMERIC, STRING = 1, 2, 4, 8
_DOT_TYPES = {1: "Serialized", 2: "Fixed_Width", 4: "Numeric", 8: "Variable_Size"}
_GRAPH_TYPES = [
    "Standard",
    "Static",
    "Selector",
    "Function",
    "Multiple_Input",
    "Parameterized",
    "Segmenter",
]


def encode_cbor(value: object) -> bytes:
    """Minimal CBOR encoder for the value types a trace uses."""

    def head(major: int, n: int) -> bytes:
        if n < 24:
            return bytes([major << 5 | n])
        if n < 1 << 8:
            return bytes([major << 5 | 24, n])
        if n < 1 << 16:
            return bytes([major << 5 | 25]) + struct.pack(">H", n)
        if n < 1 << 32:
            return bytes([major << 5 | 26]) + struct.pack(">I", n)
        return bytes([major << 5 | 27]) + struct.pack(">Q", n)

    if value is None:
        return b"\xf6"
    if value is True:
        return b"\xf5"
    if value is False:
        return b"\xf4"
    if isinstance(value, int):
        return head(0, value) if value >= 0 else head(1, -1 - value)
    if isinstance(value, float):
        return b"\xfb" + struct.pack(">d", value)
    if isinstance(value, bytes):
        return head(2, len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return head(3, len(raw)) + raw
    if isinstance(value, (list, tuple)):
        return head(4, len(value)) + b"".join(encode_cbor(v) for v in value)
    if isinstance(value, dict):
        return head(5, len(value)) + b"".join(
            encode_cbor(k) + encode_cbor(v) for k, v in value.items()
        )
    raise TypeError(f"cannot encode {type(value)}")


class ChunkBuilder:
    def __init__(self) -> None:
        # stream: [type, outputIdx, width, elements, content]
        self.streams: List[List[int]] = []
        # codec: dict(name, header, ints, copies, inputs, outputs, failure, standard)
        self.codecs: List[dict] = []
        # graph: dict(name, type, ints, codecs, failure)
        self.graphs: List[dict] = []
        self._open_graph: Optional[dict] = None
        self._finalized_csize: Optional[List[int]] = None

    # -- building ---------------------------------------------------------
    def _new_stream(
        self, index: int, stype: int, width: int, elements: int, content: Optional[int]
    ) -> int:
        if content is None:
            content = width * elements
        self.streams.append([stype, index, width, elements, content])
        return len(self.streams) - 1

    def start(self, content: int, stype: int = SERIAL, width: int = 1) -> int:
        assert not self.codecs, "start() must come first"
        self.codecs.append(
            dict(
                name="zl.#start",
                header=0,
                ints=[],
                copies=[],
                inputs=[],
                outputs=[],
                failure=None,
                standard=True,
            )
        )
        sid = self._new_stream(0, stype, width, content // width, content)
        self.codecs[0]["outputs"].append(sid)
        return sid

    def codec(
        self,
        name: str,
        inputs: Sequence[int],
        outputs: Sequence[Tuple[int, int, int]] = (),
        header: int = 0,
        ints: Sequence[Tuple[int, int]] = (),
        copies: Sequence[Tuple[int, int]] = (),
        failure: Optional[str] = None,
        standard: bool = True,
        output_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Add a codec; ``outputs`` are (type, width, content bytes). Returns stream ids.

        ``output_indices`` gives each created stream's output index when it
        differs from creation order.
        """
        codec = dict(
            name=name,
            header=header,
            ints=list(ints),
            copies=list(copies),
            inputs=list(inputs),
            outputs=[],
            failure=failure,
            standard=standard,
        )
        self.codecs.append(codec)
        if self._open_graph is not None:
            self._open_graph["codecs"].append(len(self.codecs) - 1)
        for n, (stype, width, content) in enumerate(outputs):
            index = output_indices[n] if output_indices is not None else n
            elements = content // width if width else content
            codec["outputs"].append(
                self._new_stream(index, stype, width, elements, content)
            )
        return codec["outputs"]

    def one(
        self,
        name: str,
        source: int,
        content: int,
        stype: int = SERIAL,
        width: int = 1,
        **kw,
    ) -> int:
        """Single-input, single-output codec."""
        return self.codec(name, [source], [(stype, width, content)], **kw)[0]

    def begin_graph(
        self,
        name: str,
        gtype: int = 0,
        ints: Sequence[Tuple[int, int]] = (),
        failure: Optional[str] = None,
    ) -> None:
        self._open_graph = dict(
            name=name, type=gtype, ints=list(ints), codecs=[], failure=failure
        )
        self.graphs.append(self._open_graph)

    def end_graph(self) -> None:
        self._open_graph = None

    def store_rest(self) -> None:
        """Send every stream without a consumer to zl.store, like finalizeTrace."""
        consumed = {s for c in self.codecs for s in c["inputs"]}
        for sid in range(len(self.streams)):
            if sid not in consumed:
                self.codecs.append(
                    dict(
                        name="zl.store",
                        header=0,
                        ints=[],
                        copies=[],
                        inputs=[sid],
                        outputs=[],
                        failure=None,
                        standard=True,
                    )
                )

    # -- sizes ---------------------------------------------------------------
    def csizes(self) -> List[int]:
        consumer: Dict[int, int] = {}
        for cid, codec in enumerate(self.codecs):
            for sid in codec["inputs"]:
                consumer[sid] = cid  # the last consumer wins, as in the tracer
        memo: Dict[int, int] = {}

        def fill(sid: int) -> int:
            if sid in memo:
                return memo[sid]
            cid = consumer.get(sid)
            if cid is None or not self.codecs[cid]["outputs"]:
                memo[sid] = self.streams[sid][4]
                return memo[sid]
            codec = self.codecs[cid]
            total = codec["header"] + sum(fill(o) for o in codec["outputs"])
            memo[sid] = total // len(codec["inputs"])
            return memo[sid]

        # Streams are created before the codecs that consume them, so filling from
        # the last one keeps the recursion shallow.
        for sid in reversed(range(len(self.streams))):
            fill(sid)
        return [memo[s] for s in range(len(self.streams))]

    # -- output --------------------------------------------------------------
    def to_cbor_object(self, index: int) -> dict:
        csize = self.csizes()
        total = csize[0] if csize else 0

        def params(ints, copies):
            return {
                "intParams": [{"paramId": a, "paramValue": b} for a, b in ints],
                "copyParams": [
                    {"paramId": a, "paramSize": b, "paramData": bytes(b)}
                    for a, b in copies
                ],
                "refParams": [],
            }

        return {
            "chunkId": index,
            "streams": [
                {
                    "chunkId": index,
                    "type": s[0],
                    "outputIdx": s[1],
                    "eltWidth": s[2],
                    "numElts": s[3],
                    "cSize": csize[i],
                    "share": (csize[i] / total * 100.0) if total else 0.0,
                    "contentSize": s[4],
                }
                for i, s in enumerate(self.streams)
            ],
            "codecs": [
                dict(
                    {
                        "chunkId": index,
                        "name": c["name"],
                        "cType": c["standard"],
                        "cID": 0,
                        "cHeaderSize": c["header"],
                    },
                    **({"cFailureString": c["failure"]} if c["failure"] else {}),
                    cLocalParams=params(c["ints"], c["copies"]),
                    inputStreams=c["inputs"],
                    outputStreams=c["outputs"],
                )
                for c in self.codecs
            ],
            "graphs": [
                dict(
                    {"chunkId": index, "gType": g["type"], "gName": g["name"]},
                    **({"gFailureString": g["failure"]} if g["failure"] else {}),
                    gLocalParams=params(g["ints"], []),
                    codecIDs=g["codecs"],
                )
                for g in self.graphs
            ],
        }

    def to_dot(self) -> str:
        """The digraph ChunkTrace::printStreamMetadata/printCodecMetadata print."""
        csize = self.csizes()
        total = csize[0] if csize else 0
        lines = ["digraph stream_topo {"]
        for sid, s in enumerate(self.streams):
            share = f"{csize[sid] / total * 100:.2f}" if total else "inf"
            lines.append(
                f'S{sid} [shape=record, label="Stream: {sid}\\nType: {_DOT_TYPES[s[0]]}'
                f"\\nOutputIdx: {s[1]}\\nEltWidth: {s[2]}\\n#Elts: {s[3]}"
                f'\\nCSize: {csize[sid]}\\nShare: {share}%"];'
            )
        lines.append("")
        graph_at = {
            g["codecs"][0]: (i, g) for i, g in enumerate(self.graphs) if g["codecs"]
        }
        closes = {g["codecs"][-1] for g in self.graphs if g["codecs"]}
        for cid, c in enumerate(self.codecs):
            if cid in graph_at:
                gi, g = graph_at[cid]
                label = f"{g['name']}\\ntype={_GRAPH_TYPES[g['type']]}"
                if g["failure"]:
                    label += "\\nFailure: [PLACEHOLDER]"
                if g["ints"]:
                    label += "\\nIntParams (paramId, paramValue): " + ", ".join(
                        f"({a}, {b})" for a, b in g["ints"]
                    )
                lines += [
                    f"subgraph cluster_{gi}{{",
                    f'label="{label}";',
                    "color=maroon",
                ]
            label = f"{c['name']}(ID: 0)\\n {'Standard' if c['standard'] else 'Custom'} transform {cid}\\n Header size: {c['header']}"
            if c["failure"]:
                label += "\\n Failure: [PLACEHOLDER]"
            if c["ints"]:
                label += "\\nIntParams (paramId, paramValue): " + ", ".join(
                    f"({a}, {b})" for a, b in c["ints"]
                )
            if c["copies"]:
                label += "\\nCopyParams (paramId, paramSize): " + ", ".join(
                    f"({a}, {b})" for a, b in c["copies"]
                )
            lines.append(f'T{cid} [shape=Mrecord, label="{label}"];')
            outs = sorted(c["outputs"])
            for n, sid in enumerate(outs):
                # zli labels output edges in reverse sid order
                lines.append(f'T{cid} -> S{sid}[label="#{len(outs) - 1 - n}"];')
            for n, sid in enumerate(sorted(c["inputs"])):
                lines.append(f'S{sid} -> T{cid}[label="#{n}"];')
            if cid in closes:
                lines.append("}")
        lines.append("}")
        return "\n".join(lines) + "\n"


class TraceBuilder:
    def __init__(self) -> None:
        self.chunks: List[ChunkBuilder] = []

    def chunk(self) -> ChunkBuilder:
        self.chunks.append(ChunkBuilder())
        return self.chunks[-1]

    def to_cbor(self) -> bytes:
        return encode_cbor(
            {
                "libraryVersion": 1,
                "frameVersion": 23,
                "traceVersion": 1,
                "chunks": [c.to_cbor_object(i) for i, c in enumerate(self.chunks)],
            }
        )

    def to_dot(self) -> str:
        # Chunks print as they finish; the top-level graph (chunk 0) prints last.
        order = self.chunks[1:] + self.chunks[:1]
        return "\n".join(c.to_dot() for c in order)


def serial_trace(content: int = 1000, compressed: int = 400) -> TraceBuilder:
    tb = TraceBuilder()
    ch = tb.chunk()
    s0 = ch.start(content)
    ch.begin_graph("zl.ace#0", gtype=5)
    ch.one("zl.private.zstd", s0, compressed, ints=[(100, 7)])
    ch.end_graph()
    ch.store_rest()
    return tb
