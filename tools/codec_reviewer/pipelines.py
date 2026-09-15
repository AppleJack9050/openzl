# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Collapse a trace's codec graph into the pipelines a reviewer reads.

A compression graph repeats itself: the same codec pipeline usually runs for
many Parquet columns, CSV fields or chunks. This module finds those
repetitions and describes each distinct pipeline once.

* The graph is cut at *merges* (codecs with more than one input, such as the
  ``concat_num`` that joins a cluster of columns) and at *splits* (dispatch
  codecs that hand each tag to its own graph, or any codec with very many
  outputs). Each cut piece is a pipeline root.
* Every root gets a structural signature: codec names, graph names and types,
  int parameters and copy/ref parameter ids, output order, and where each output
  stream ends (another codec, the frame, a merge, or a split).
* Roots with the same signature form one :class:`Pipeline`; sizes are summed
  over its runs.

Every codec node and every stored byte of the trace lands in exactly one
pipeline, and :class:`Coverage` checks that.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import re
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

from trace_format import Chunk, Codec, Params, Trace

# A dispatch codec with at least this many continuing outputs is a split point;
# any other codec needs this many. ``transpose*`` codecs, which make one stream
# per byte of a record, are never split points.
DISPATCH_SPLIT_MIN = 3
GENERIC_SPLIT_MIN = 24

# Trees are walked recursively; a chain of codecs thousands deep is legal.
RECURSION_LIMIT = 20000

ZSTD_PARAMS = {
    100: "level",
    101: "windowLog",
    102: "hashLog",
    103: "chainLog",
    104: "searchLog",
    105: "minMatch",
    106: "targetLength",
    107: "strategy",
    160: "long matching",
    161: "ldmHashLog",
    162: "ldmMinMatch",
    163: "ldmBucketSizeLog",
    164: "ldmHashRateLog",
}
PARAM_NAMES = [
    (re.compile(r"^(zstd|delta_zstd|range_pack_zstd)"), ZSTD_PARAMS),
    (re.compile(r"^field_lz"), {181: "level"}),
    (re.compile(r"^tokenize"), {0: "sorted alphabet"}),
    (re.compile(r"^divide_by"), {112: "divisor"}),
    (re.compile(r"^convert_serial_to_struct"), {1: "struct size"}),
    (re.compile(r"^splitN"), {324: "segments"}),
    # Parquet segmenter: 2; CSV segmenter: 100 and 225-227.
    (
        re.compile(r"segmenter"),
        {
            2: "chunk size",
            100: "chunk size",
            225: "has header",
            226: "separator",
            227: "null aware",
        },
    ),
    (re.compile(r"^cluster"), {316: "config size"}),
]


def allow_deep_trees(fn):
    """Run ``fn`` with enough recursion depth for long codec chains."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, RECURSION_LIMIT))
        try:
            return fn(*args, **kwargs)
        finally:
            sys.setrecursionlimit(limit)

    return wrapper


def short_name(name: str) -> str:
    """``zl.private.field_lz#3`` -> ``field_lz#3``."""
    name = name[1:] if name.startswith("!") else name
    for prefix in ("zl.private.", "zl."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def base_name(name: str) -> str:
    """``zl.private.field_lz#3`` -> ``field_lz``."""
    return re.sub(r"#\d+$", "", short_name(name))


def is_conversion(name: str) -> bool:
    return base_name(name).startswith("convert_")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SplitOutputs:
    """The outputs of a split codec, each listed as its own pipeline."""

    outputs: int
    """Outputs over all runs."""
    raw: int
    csize: int
    refs: List[Tuple[int, int]]  # (chunk index, stream id)
    runs: int = 1
    pipelines: List[str] = dataclasses.field(default_factory=list)

    @property
    def per_run(self) -> int:
        return self.outputs // max(self.runs, 1)


@dataclasses.dataclass
class NodeRuns:
    """One codec position in a pipeline, aggregated over the pipeline's runs."""

    codec: str
    graph: str
    graph_type: str
    graph_params: Optional[Params]
    codec_params: Params
    header: int
    inputs: int
    failures: List[str]
    graph_failures: List[str]
    copy_sizes: Dict[int, int]
    children: List["StreamRuns"] = dataclasses.field(default_factory=list)
    split: Optional[SplitOutputs] = None

    @property
    def failed(self) -> bool:
        return bool(self.failures or self.graph_failures)


@dataclasses.dataclass
class StreamRuns:
    """One stream position in a pipeline, aggregated over the pipeline's runs."""

    index: int
    type: str
    width: int
    elements: int
    raw: Optional[int]
    csize: int
    kind: str
    """"node", "store" (written to the frame), "junction" (feeds a merge),
    "progress" (compression stopped) or "end" (no consumer)."""
    node: Optional[NodeRuns] = None
    junction_codec: Optional[str] = None
    junction_refs: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    merges: List[str] = dataclasses.field(default_factory=list)
    progress_messages: List[str] = dataclasses.field(default_factory=list)
    progress_owner: bool = True
    """A placeholder with several inputs is counted, with its messages, on one."""
    attempts: List[NodeRuns] = dataclasses.field(default_factory=list)
    """Consumers that took the stream first and were abandoned (a failed codec,
    a failed graph's placeholder, or work that permissive mode rolled back)."""
    kept: bool = True
    """False inside abandoned work, whose stores never reached the frame."""
    unfinished: bool = False
    """Kept work that no codec finished: compression stopped here."""


Tree = Union[NodeRuns, StreamRuns]


@dataclasses.dataclass
class Pipeline:
    id: str
    kind: str
    """"entry" (starts at the input), "merge" or "split" (one split output)."""
    count: int
    chunks: List[int]
    tree: Tree
    raw: Optional[int]
    csize: Optional[int]
    nodes: int
    """Codec nodes per run."""
    stored: int
    """Bytes written to the frame by all runs (abandoned work excluded)."""
    codecs: Set[str]
    failed: bool
    splitter: Optional[str] = None
    outputs: List[int] = dataclasses.field(default_factory=list)
    feeder: bool = False
    """A split output that only passes through to a merge."""
    fanin: Optional[int] = None
    origins: List[Tuple[str, int]] = dataclasses.field(default_factory=list)
    other_inputs: int = 0
    """Merge inputs per run that did not come straight from a split output."""
    top_level: bool = False
    chunk_count: Optional[int] = None

    @property
    def ratio(self) -> Optional[float]:
        if self.raw and self.csize:
            return self.raw / self.csize
        return None


@dataclasses.dataclass
class Coverage:
    nodes: int
    nodes_total: int
    stored: int
    stored_total: int
    split_outputs: int
    split_outputs_total: int
    merges: int
    merges_total: int

    @property
    def ok(self) -> bool:
        return (
            self.nodes == self.nodes_total
            and self.stored == self.stored_total
            and self.split_outputs == self.split_outputs_total
            and self.merges == self.merges_total
        )


@dataclasses.dataclass
class CensusRow:
    codec: str
    runs: int
    header: int
    stored: int


@dataclasses.dataclass
class ChunkSummary:
    index: int
    top_level: bool
    input: int
    csize: Optional[int]
    codecs: int


@dataclasses.dataclass
class Analysis:
    selection: Union[str, int]
    pipelines: List[Pipeline]
    coverage: Coverage
    chunk_count: int
    input: Optional[int]
    csize: Optional[int]
    """Bytes in streams; when compression failed, of the chunks that finished."""
    codecs_run: int
    failures: int
    """Codecs and graphs that recorded an error."""
    unfinished: int
    """Streams of kept work that no codec finished: their last consumer failed or
    is an #in_progress placeholder."""
    compression_failed: bool
    census: List[CensusRow]

    def get(self, pipeline_id: str) -> Optional[Pipeline]:
        wanted = pipeline_id.upper()
        return next((p for p in self.pipelines if p.id == wanted), None)

    @property
    def ratio(self) -> Optional[float]:
        if self.input and self.csize:
            return self.input / self.csize
        return None


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


class _ChunkGraph:
    def __init__(self, chunk: Chunk) -> None:
        self.chunk = chunk
        streams = chunk.streams
        self.consumers: Dict[int, List[int]] = collections.defaultdict(list)
        self.producer: Dict[int, int] = {}
        self.outputs: Dict[int, List[int]] = {}
        for codec in chunk.codecs:
            # Order children by the stream's output index at its producer.
            self.outputs[codec.id] = sorted(
                codec.outputs, key=lambda s: (streams[s].output_index, s)
            )
            for sid in codec.inputs:
                self.consumers[sid].append(codec.id)
            for sid in codec.outputs:
                self.producer[sid] = codec.id
        self._split: Dict[int, Optional[List[int]]] = {}
        self.kept: Set[int] = set()
        self.unfinished: Set[int] = set()
        self._follow_kept_work()

    def codec(self, cid: int) -> Codec:
        return self.chunk.codecs[cid]

    def _follow_kept_work(self) -> None:
        """Walk from the input along each stream's final consumer. Streams reached
        this way are kept; the rest belong to abandoned work. A kept stream whose
        final consumer failed or is a placeholder was never compressed."""
        stack = [s for c in self.chunk.codecs if not c.inputs for s in c.outputs]
        visited: Set[int] = set()
        while stack:
            sid = stack.pop()
            if sid in self.kept:
                continue
            self.kept.add(sid)
            cid = self.primary(sid)
            if cid is None:
                continue
            codec = self.codec(cid)
            if codec.failure or self.is_progress(codec):
                self.unfinished.add(sid)
            elif cid not in visited:
                visited.add(cid)
                stack.extend(codec.outputs)

    def primary(self, sid: int) -> Optional[int]:
        """The consumer a stream finally went to: the last one that didn't fail."""
        consumers = self.consumers.get(sid, [])
        if not consumers:
            return None
        ok = [c for c in consumers if not self.codec(c).failure]
        return ok[-1] if ok else consumers[-1]

    def attempts(self, sid: int) -> List[int]:
        """Earlier consumers that were abandoned. Stores are left out, and so is a
        merge that is the final consumer of another input: it is its own
        pipeline. A codec with several inputs is listed under its anchor input."""
        primary = self.primary(sid)
        return [
            c
            for c in self.consumers.get(sid, [])
            if c != primary
            and not self.is_store(self.codec(c))
            and (len(self.codec(c).inputs) <= 1 or self.anchor(c) == sid)
        ]

    def anchor(self, cid: int) -> int:
        """The input a codec is drawn under: the first one it finally took, or its
        first input when it was abandoned by all of them."""
        inputs = self.codec(cid).inputs
        return next((s for s in inputs if self.primary(s) == cid), inputs[0])

    def is_merge(self, codec: Codec) -> bool:
        """A codec joining several streams that at least one of them went to."""
        return (
            len(codec.inputs) > 1
            and not self.is_progress(codec)
            and any(self.primary(s) == codec.id for s in codec.inputs)
        )

    def is_store(self, codec: Codec) -> bool:
        return not codec.outputs and base_name(codec.name) == "store"

    def is_progress(self, codec: Codec) -> bool:
        return not codec.outputs and base_name(codec.name) == "#in_progress"

    def split_outputs(self, codec: Codec) -> Optional[List[int]]:
        if codec.id not in self._split:
            continuing = [
                sid
                for sid in self.outputs[codec.id]
                if self.primary(sid) is None
                or not self.is_store(self.codec(self.primary(sid)))
            ]
            name = base_name(codec.name)
            if name.startswith("dispatch"):
                minimum = DISPATCH_SPLIT_MIN
            elif name.startswith("transpose"):
                minimum = len(continuing) + 1
            else:
                minimum = GENERIC_SPLIT_MIN
            self._split[codec.id] = continuing if len(continuing) >= minimum else None
        return self._split[codec.id]

    def children(self, codec: Codec) -> List[int]:
        """Output streams that stay in the codec's own pipeline."""
        split = self.split_outputs(codec)
        outs = self.outputs[codec.id]
        return [s for s in outs if s not in split] if split else outs

    def tail(self, sid: int) -> Tuple[str, Optional[int]]:
        cid = self.primary(sid)
        if cid is None:
            return "end", None
        codec = self.codec(cid)
        if self.is_store(codec):
            return "store", cid
        if self.is_progress(codec):
            return "progress", cid
        if len(codec.inputs) > 1:
            return "junction", cid
        return "node", cid

    def graph_failure(self, codec: Codec) -> Optional[str]:
        """A graph's error, reported once: on the first codec of the graph."""
        if codec.graph is None:
            return None
        graph = self.chunk.graphs[codec.graph]
        if graph.failure and graph.codecs and graph.codecs[0] == codec.id:
            return graph.failure
        return None


def is_top_level(trace: Trace, chunk: Chunk) -> bool:
    """The top-level run of a segmented compression: start -> segmenter."""
    return (
        len(trace.chunks) > 1
        and chunk.index == 0
        and any(not c.outputs and "segmenter" in c.name.lower() for c in chunk.codecs)
    )


def chunk_summaries(trace: Trace) -> List[ChunkSummary]:
    result = []
    for chunk in trace.chunks:
        top = is_top_level(trace, chunk)
        raw = csize = 0
        for codec in chunk.codecs:
            if not codec.inputs:
                for sid in codec.outputs:
                    raw += chunk.streams[sid].raw_size or 0
                    csize += chunk.streams[sid].csize
        result.append(
            ChunkSummary(
                chunk.index, top, raw, None if top else csize, len(chunk.codecs)
            )
        )
    return result


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _param_key(params: Params) -> tuple:
    # Copy parameter sizes can vary run to run; ids identify the setting.
    return (
        tuple(params.ints),
        tuple(pid for pid, _ in params.copies),
        tuple(params.refs),
    )


class _Signer:
    """Interns structural signatures so equal subtrees get equal small ints."""

    def __init__(self) -> None:
        self._table: Dict[tuple, int] = {}
        self._streams: Dict[Tuple[int, int], int] = {}
        self._nodes: Dict[Tuple[int, int], int] = {}

    def _intern(self, key: tuple) -> int:
        return self._table.setdefault(key, len(self._table))

    def node_key(self, g: _ChunkGraph, codec: Codec) -> tuple:
        graph = g.chunk.graphs[codec.graph] if codec.graph is not None else None
        graph_key = None
        if graph is not None:
            # "#N" only numbers graph instances (e.g. each trained ACE graph).
            graph_key = (
                re.sub(r"#\d+$", "", graph.name),
                graph.type,
                _param_key(graph.params),
                bool(graph.failure),
            )
        fanin = len(codec.inputs) if len(codec.inputs) > 1 else 0
        return (
            re.sub(r"#\d+$", "", codec.name),
            graph_key,
            _param_key(codec.params),
            bool(codec.failure),
            fanin,
        )

    def stream(self, g: _ChunkGraph, sid: int) -> int:
        memo_key = (id(g), sid)
        if memo_key in self._streams:
            return self._streams[memo_key]
        s = g.chunk.streams[sid]
        kind, cid = g.tail(sid)
        if kind == "node":
            tail: tuple = ("node", self.node(g, cid))
        elif kind == "junction":
            tail = ("junction", g.codec(cid).name)
        elif kind == "progress":
            tail = ("progress", g.anchor(cid) == sid)
        else:
            tail = (kind,)
        attempts = tuple(self.node(g, c) for c in g.attempts(sid))
        value = self._intern(("stream", s.type, s.width, tail, attempts))
        self._streams[memo_key] = value
        return value

    def node(self, g: _ChunkGraph, cid: int) -> int:
        memo_key = (id(g), cid)
        if memo_key in self._nodes:
            return self._nodes[memo_key]
        codec = g.codec(cid)
        split = g.split_outputs(codec)
        kids = tuple(self.stream(g, s) for s in g.children(codec))
        value = self._intern(
            ("node", self.node_key(g, codec), len(split) if split else 0, kids)
        )
        self._nodes[memo_key] = value
        return value


# ---------------------------------------------------------------------------
# Aggregation over identical runs
# ---------------------------------------------------------------------------


def _agg_stream(
    graphs: Dict[int, _ChunkGraph], runs: List[Tuple[int, int]]
) -> StreamRuns:
    first_chunk, first_sid = runs[0]
    g0 = graphs[first_chunk]
    s0 = g0.chunk.streams[first_sid]
    kind, cid0 = g0.tail(first_sid)
    raw: Optional[int] = 0
    elements = csize = 0
    for ci, sid in runs:
        s = graphs[ci].chunk.streams[sid]
        elements += s.elements
        csize += s.csize
        r = s.raw_size
        raw = None if raw is None or r is None else raw + r
    result = StreamRuns(s0.output_index, s0.type, s0.width, elements, raw, csize, kind)
    result.kept = any(sid in graphs[ci].kept for ci, sid in runs)
    result.unfinished = any(sid in graphs[ci].unfinished for ci, sid in runs)
    attempts = [graphs[ci].attempts(sid) for ci, sid in runs]
    for k in range(len(attempts[0])):
        result.attempts.append(
            _agg_node(graphs, [(ci, attempts[r][k]) for r, (ci, _) in enumerate(runs)])
        )
    if kind == "progress":
        result.progress_owner = g0.anchor(cid0) == first_sid
    if kind == "progress" and result.progress_owner:
        # A placeholder the tracer adds for a failed graph carries that graph's error.
        for ci, sid in runs:
            g = graphs[ci]
            message = g.codec(g.primary(sid)).failure or g.graph_failure(
                g.codec(g.primary(sid))
            )
            if message:
                result.progress_messages.append(message)
    if kind == "node":
        result.node = _agg_node(
            graphs, [(ci, graphs[ci].primary(sid)) for ci, sid in runs]
        )
    elif kind == "junction":
        result.junction_codec = g0.codec(cid0).name
        result.junction_refs = [(ci, graphs[ci].primary(sid)) for ci, sid in runs]
    return result


def _agg_node(graphs: Dict[int, _ChunkGraph], runs: List[Tuple[int, int]]) -> NodeRuns:
    first_chunk, first_cid = runs[0]
    g0 = graphs[first_chunk]
    c0 = g0.codec(first_cid)
    graph0 = g0.chunk.graphs[c0.graph] if c0.graph is not None else None
    node = NodeRuns(
        codec=c0.name,
        graph=graph0.name if graph0 else "",
        graph_type=graph0.type if graph0 else "",
        graph_params=graph0.params if graph0 else None,
        codec_params=c0.params,
        header=0,
        inputs=len(c0.inputs),
        failures=[],
        graph_failures=[],
        copy_sizes={},
    )
    for ci, cid in runs:
        g = graphs[ci]
        codec = g.codec(cid)
        node.header += codec.header_size
        if codec.failure:
            node.failures.append(codec.failure)
        if g.graph_failure(codec):
            node.graph_failures.append(g.graph_failure(codec))
        for pid, size in codec.params.copies:
            node.copy_sizes[pid] = node.copy_sizes.get(pid, 0) + size
    kids = [graphs[ci].children(graphs[ci].codec(cid)) for ci, cid in runs]
    for k in range(len(kids[0])):
        node.children.append(
            _agg_stream(graphs, [(ci, kids[r][k]) for r, (ci, _) in enumerate(runs)])
        )
    if g0.split_outputs(c0):
        split = SplitOutputs(0, 0, 0, [], runs=len(runs))
        for ci, cid in runs:
            g = graphs[ci]
            for sid in g.split_outputs(g.codec(cid)):
                s = g.chunk.streams[sid]
                split.outputs += 1
                split.raw += s.raw_size or 0
                split.csize += s.csize
                split.refs.append((ci, sid))
        node.split = split
    return node


def walk_nodes(tree: Optional[Tree]):
    """Every codec position under ``tree``, abandoned attempts included."""
    stack = [tree]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, StreamRuns):
            stack.append(item.node)
            stack.extend(reversed(item.attempts))
            continue
        yield item
        stack.extend(reversed(item.children))


def _streams(tree: Tree):
    """Every stream position under ``tree``, the root stream included."""
    if isinstance(tree, StreamRuns):
        yield tree
    for node in walk_nodes(tree):
        yield from node.children


def _count_nodes(tree: Tree) -> int:
    # #in_progress placeholders are codec nodes too.
    return sum(1 for _ in walk_nodes(tree)) + sum(
        1 for s in _streams(tree) if s.kind == "progress" and s.progress_owner
    )


def _stored_bytes(tree: Tree) -> int:
    return sum(s.csize for s in _streams(tree) if s.kind == "store" and s.kept)


def _has_failure(tree: Tree) -> bool:
    return any(n.failed for n in walk_nodes(tree)) or any(
        s.attempts or s.kind == "progress" for s in _streams(tree)
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@allow_deep_trees
def analyze(trace: Trace, selection: Union[str, int] = "all") -> Analysis:
    """Group the codec graph of ``trace`` into pipelines.

    ``selection`` is ``"all"`` or the index of one chunk of the trace.
    """
    if selection == "all":
        chunks = trace.chunks
    else:
        chunks = [c for c in trace.chunks if c.index == selection]
        if not chunks:
            raise ValueError(f"The trace has no chunk {selection}.")
    return _analyze(trace, selection, chunks)


def _analyze(trace: Trace, selection: Union[str, int], chunks: List[Chunk]) -> Analysis:
    graphs = {c.index: _ChunkGraph(c) for c in chunks}
    signer = _Signer()
    groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    split_info: Dict[Tuple[int, int], Tuple[str, int]] = {}
    totals = collections.Counter()

    for chunk in chunks:
        g = graphs[chunk.index]
        for codec in chunk.codecs:
            totals["codecs"] += 1
            if g.is_store(codec):
                totals["stores"] += 1
                totals["stored"] += sum(
                    chunk.streams[s].csize for s in codec.inputs if s in g.kept
                )
                continue
            if codec.failure:
                totals["failures"] += 1
            if g.is_progress(codec):
                totals["progress"] += 1
            ref = (chunk.index, codec.id)
            if not codec.inputs:
                totals["entries"] += 1
                groups.setdefault(("entry", signer.node(g, codec.id)), []).append(ref)
            if g.is_merge(codec):
                totals["merges"] += 1
                groups.setdefault(("merge", signer.node(g, codec.id)), []).append(ref)
            for sid in g.split_outputs(codec) or []:
                totals["split_outputs"] += 1
                split_info[(chunk.index, sid)] = (
                    base_name(codec.name),
                    chunk.streams[sid].output_index,
                )
                key = ("split", base_name(codec.name), signer.stream(g, sid))
                groups.setdefault(key, []).append((chunk.index, sid))
        totals["failures"] += sum(1 for gr in chunk.graphs if gr.failure)
        totals["unfinished"] += len(g.unfinished)

    segmented = len(trace.chunks) > 1 and is_top_level(trace, trace.chunks[0])
    all_chunk_bytes = sum(c.csize or 0 for c in chunk_summaries(trace))

    pipelines: List[Pipeline] = []
    members: Dict[int, List[Tuple[int, int]]] = {}
    for key, runs in groups.items():
        kind = key[0]
        chunk_list = sorted({ci for ci, _ in runs})
        if kind == "split":
            tree = _agg_stream(graphs, runs)
            infos = [split_info[r] for r in runs]
            end = tree
            while (
                end.node is not None
                and len(end.node.children) == 1
                and not end.node.split
            ):
                end = end.node.children[0]
            stored = _stored_bytes(tree)
            p = Pipeline(
                id="",
                kind=kind,
                count=len(runs),
                chunks=chunk_list,
                tree=tree,
                raw=tree.raw,
                csize=tree.csize,
                nodes=_count_nodes(tree),
                stored=stored,
                codecs={base_name(n.codec) for n in walk_nodes(tree)},
                failed=_has_failure(tree),
                splitter=infos[0][0],
                outputs=sorted({idx for _, idx in infos}),
                feeder=end.kind == "junction" and stored == 0,
            )
        else:
            tree = _agg_node(graphs, runs)
            outs = sum(c.csize for c in tree.children)
            p = Pipeline(
                id="",
                kind=kind,
                count=len(runs),
                chunks=chunk_list,
                tree=tree,
                raw=None,
                csize=tree.header + outs + (tree.split.csize if tree.split else 0),
                nodes=_count_nodes(tree),
                stored=_stored_bytes(tree),
                codecs={base_name(n.codec) for n in walk_nodes(tree)},
                failed=_has_failure(tree),
            )
            raw: Optional[int] = 0
            if kind == "merge":
                p.fanin = tree.inputs
                origins = set()
                for n, (ci, cid) in enumerate(runs):
                    g = graphs[ci]
                    for sid in g.codec(cid).inputs:
                        r = g.chunk.streams[sid].raw_size
                        raw = None if raw is None or r is None else raw + r
                        origin = _split_origin(g, sid, split_info)
                        if origin:
                            origins.add(origin)
                        elif n == 0:
                            p.other_inputs += 1
                p.origins = sorted(origins)
            else:
                p.top_level = all(
                    is_top_level(trace, graphs[ci].chunk) for ci, _ in runs
                )
                for ci, cid in runs:
                    g = graphs[ci]
                    for sid in g.codec(cid).outputs:
                        r = g.chunk.streams[sid].raw_size
                        raw = None if raw is None or r is None else raw + r
                if p.top_level and segmented:
                    # The top level ends at the segmenter, so its stream sizes are
                    # raw sizes; the compressed bytes are in the chunk traces.
                    p.csize = all_chunk_bytes
                    p.chunk_count = len(trace.chunks) - 1
            p.raw = raw
        members[id(p)] = runs
        pipelines.append(p)

    _assign_ids(pipelines)
    _resolve_links(pipelines, members)

    body = [c for c in chunks if not is_top_level(trace, c)]
    top = next((c for c in chunks if is_top_level(trace, c)), None)
    input_raw: Optional[int] = 0
    for chunk in [top] if top else body:
        for codec in chunk.codecs:
            if not codec.inputs:
                for sid in codec.outputs:
                    r = chunk.streams[sid].raw_size
                    input_raw = (
                        None if input_raw is None or r is None else input_raw + r
                    )
    # A failed segmenter leaves its input unfinished too.
    compression_failed = bool(totals["unfinished"])
    if compression_failed:
        # Only work that finished has compressed sizes; a ratio would mislead.
        stopped = {ci for ci, g in graphs.items() if g.unfinished}
        for p in pipelines:
            runs = members[id(p)]
            if (
                p.top_level
                or (p.kind == "entry" and any(ci in stopped for ci, _ in runs))
                or any(s.unfinished for s in _streams(p.tree))
                or (
                    p.kind == "merge"
                    and any(
                        s in graphs[ci].unfinished
                        for ci, cid in runs
                        for s in graphs[ci].codec(cid).inputs
                    )
                )
            ):
                p.csize = None
    finished = [
        p.csize
        for p in pipelines
        if p.kind == "entry" and not p.top_level and p.csize is not None
    ]
    # Without a finished chunk (or with only the top level selected) there is no size.
    csize = sum(finished) if finished else None
    coverage = Coverage(
        nodes=sum(p.nodes * p.count for p in pipelines) + totals["stores"],
        nodes_total=totals["codecs"],
        stored=sum(p.stored for p in pipelines),
        stored_total=totals["stored"],
        split_outputs=sum(p.count for p in pipelines if p.kind == "split"),
        split_outputs_total=totals["split_outputs"],
        merges=sum(p.count for p in pipelines if p.kind == "merge"),
        merges_total=totals["merges"],
    )

    census: Dict[str, CensusRow] = {}
    for chunk in chunks:
        g = graphs[chunk.index]
        for codec in chunk.codecs:
            row = census.setdefault(
                base_name(codec.name), CensusRow(base_name(codec.name), 0, 0, 0)
            )
            row.runs += 1
            row.header += codec.header_size
            if g.is_store(codec):
                row.stored += sum(
                    chunk.streams[s].csize for s in codec.inputs if s in g.kept
                )

    return Analysis(
        selection=selection,
        pipelines=pipelines,
        coverage=coverage,
        chunk_count=len(body),
        input=input_raw,
        csize=csize,
        codecs_run=totals["codecs"]
        - totals["stores"]
        - totals["progress"]
        - totals["entries"],
        failures=totals["failures"],
        unfinished=totals["unfinished"],
        compression_failed=compression_failed,
        census=sorted(census.values(), key=lambda r: (-r.runs, r.codec)),
    )


def _split_origin(
    g: _ChunkGraph, sid: int, split_info: Dict[Tuple[int, int], Tuple[str, int]]
) -> Optional[Tuple[str, int]]:
    """Walk up single-input codecs to the split output a merge input came from."""
    seen = set()
    while sid not in seen:
        seen.add(sid)
        info = split_info.get((g.chunk.index, sid))
        if info:
            return info
        producer = g.producer.get(sid)
        if producer is None or len(g.codec(producer).inputs) != 1:
            return None
        sid = g.codec(producer).inputs[0]
    return None


def _assign_ids(pipelines: List[Pipeline]) -> None:
    """E1.., M1.., S1..: numbered by compressed bytes within each kind."""
    prefix = {"entry": "E", "merge": "M", "split": "S"}
    order = {"entry": 0, "merge": 1, "split": 2}
    pipelines.sort(
        key=lambda p: (order[p.kind], not p.top_level, p.feeder, -(p.csize or 0))
    )
    counters = collections.Counter()
    for p in pipelines:
        counters[p.kind] += 1
        p.id = f"{prefix[p.kind]}{counters[p.kind]}"


def _resolve_links(
    pipelines: List[Pipeline], members: Dict[int, List[Tuple[int, int]]]
) -> None:
    merge_of: Dict[Tuple[int, int], str] = {}
    split_of: Dict[Tuple[int, int], str] = {}
    for p in pipelines:
        for ref in members[id(p)]:
            if p.kind == "merge":
                merge_of[ref] = p.id
            elif p.kind == "split":
                split_of[ref] = p.id

    def ordered(ids):
        return sorted(set(ids), key=lambda i: (i[0], int(i[1:])))

    def resolve_stream(s: StreamRuns) -> None:
        if s.kind == "junction":
            s.merges = ordered(merge_of[r] for r in s.junction_refs)
        resolve_node(s.node)

    def resolve_node(n: Optional[NodeRuns]) -> None:
        if n is None:
            return
        if n.split:
            n.split.pipelines = ordered(split_of[r] for r in n.split.refs)
        for child in n.children:
            resolve_stream(child)

    for p in pipelines:
        if isinstance(p.tree, StreamRuns):
            resolve_stream(p.tree)
        else:
            resolve_node(p.tree)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _param_table(name: str) -> Dict[int, str]:
    b = base_name(name)
    return next((table for pattern, table in PARAM_NAMES if pattern.search(b)), {})


def _int_labels(params: Params, names: Dict[int, str]) -> List[str]:
    labels = []
    for pid, value in params.ints:
        name = names.get(pid)
        if name == "long matching":
            labels.append(f"long matching {({1: 'on', 2: 'off'}).get(value, value)}")
        elif name == "sorted alphabet":
            labels.append(f"sorted alphabet {'yes' if value else 'no'}")
        elif name == "chunk size":
            size = (
                f"{value // 1_000_000} MB"
                if value and value % 1_000_000 == 0
                else f"{value:,} B"
            )
            labels.append(f"chunk size {size}")
        elif name:
            labels.append(
                f"{name} {value:,}" if abs(value) >= 10000 else f"{name} {value}"
            )
        else:
            labels.append(
                f"param {pid} = {value:,}"
                if abs(value) >= 10000
                else f"param {pid} = {value}"
            )
    return labels


def settings(node: NodeRuns) -> List[str]:
    """Readable int parameters: the graph's first, then the codec's own."""
    labels: List[str] = []
    if node.graph_params is not None:
        names = _param_table(node.graph) or _param_table(node.codec)
        labels += _int_labels(node.graph_params, names)
    labels += _int_labels(node.codec_params, _param_table(node.codec))
    return list(dict.fromkeys(labels))


def blobs(node: NodeRuns) -> List[str]:
    labels = []
    if node.graph_params is not None:
        labels += [
            f"graph blob {pid} ({size:,} B)" for pid, size in node.graph_params.copies
        ]
        labels += [f"graph ref {pid}" for pid in node.graph_params.refs]
    labels += [
        f"blob {pid} ({node.copy_sizes.get(pid, 0):,} B)"
        for pid, _ in node.codec_params.copies
    ]
    labels += [f"ref {pid}" for pid in node.codec_params.refs]
    return labels


def type_label(stream_type: str, width: int) -> str:
    if stream_type == "serial":
        return "bytes"
    if stream_type == "string":
        return "strings"
    if stream_type == "numeric":
        return f"{width * 8}-bit numbers" if width else "numbers"
    if stream_type == "struct":
        return f"{width}-byte records" if width else "records"
    return "stream"


def number_list(values: List[int], limit: int = 12) -> str:
    shown = ", ".join(f"#{v}" for v in values[:limit])
    return f"{shown} and {len(values) - limit} more" if len(values) > limit else shown


def main_chain(pipeline: Pipeline, limit: int = 5) -> str:
    """The codecs along the most expensive path, conversions left out."""
    tree = pipeline.tree
    names: List[str] = []
    if pipeline.top_level:
        return ""  # the title already says "segmenter -> N chunks"
    if isinstance(tree, StreamRuns) and tree.kind == "junction":
        return f"{base_name(tree.junction_codec or '')} ({', '.join(tree.merges)})"
    node = tree.node if isinstance(tree, StreamRuns) else tree
    path: List[
        Tuple[str, str]
    ] = []  # (label, "codec" | "conversion" | "merge" | "split")
    while node is not None:
        name = base_name(node.codec)
        if name != "#start":
            path.append((name, "conversion" if is_conversion(node.codec) else "codec"))
        if node.split:
            path.append((f"{node.split.per_run} outputs", "split"))
            break
        nexts = [c for c in node.children if c.kind != "store"]
        if not nexts:
            break
        best = max(nexts, key=lambda c: c.csize)
        if best.kind == "junction":
            merges = ", ".join(best.merges)
            path.append((f"{base_name(best.junction_codec or '')} ({merges})", "merge"))
            break
        node = best.node
    # Leave conversions out, unless converting is all the pipeline does.
    if any(kind == "codec" for _, kind in path):
        path = [(label, kind) for label, kind in path if kind != "conversion"]
    names = [label for label, _ in path]
    if len(names) > limit:
        names = names[:limit] + ["…"]
    return " → ".join(names)


def title(pipeline: Pipeline, analysis: Analysis) -> Tuple[str, str]:
    """(name, description) for listing a pipeline."""
    p = pipeline
    if p.kind == "entry":
        if p.top_level:
            n = p.chunk_count or 0
            return "Top level", f"segmenter → {n} chunk{'s' if n != 1 else ''}"
        if analysis.chunk_count <= 1:
            return "Input", ""
        if p.count == analysis.chunk_count:
            return "Every chunk", ""
        if p.count == 1:
            return f"Chunk {p.chunks[0]}", ""
        return f"{p.count} chunks", ""
    if p.kind == "merge":
        name = f"{base_name(p.tree.codec)} of {p.fanin} streams"
        if p.count > 1:
            name = f"{p.count}× {name}"
        if p.origins:
            by_splitter: Dict[str, List[int]] = collections.defaultdict(list)
            for splitter, index in p.origins:
                by_splitter[splitter].append(index)
            desc = "; ".join(
                f"from outputs {number_list(sorted(v))} of {k}"
                for k, v in by_splitter.items()
            )
            if p.other_inputs:
                desc += f", and {p.other_inputs} other input{'s' if p.other_inputs != 1 else ''}"
        else:
            desc = f"{p.fanin} inputs"
        return name, desc
    name = (
        f"Output #{p.outputs[0]}"
        if len(p.outputs) == 1
        else f"{len(p.outputs)} outputs"
    )
    desc = type_label(p.tree.type, p.tree.width)
    if p.tree.kind == "store":
        desc += ", stored as is"
    if analysis.chunk_count > 1 and len(p.chunks) > 1:
        desc += f", in {len(p.chunks)} chunks"
    return name, desc
