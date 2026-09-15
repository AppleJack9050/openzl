# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Plain-text rendering of an analysis: the review summary and pipeline trees."""

from __future__ import annotations

import dataclasses
import textwrap
from typing import List, Optional, Tuple

from pipelines import (
    allow_deep_trees,
    Analysis,
    NodeRuns,
    Pipeline,
    StreamRuns,
    base_name,
    is_conversion,
    is_top_level,
    main_chain,
    number_list,
    settings,
    short_name,
    title,
    walk_nodes,
)
from trace_format import Trace


@dataclasses.dataclass(frozen=True)
class Glyphs:
    tee: str = "├── "
    last: str = "└── "
    pipe: str = "│   "
    blank: str = "    "
    arrow: str = "→"
    times: str = "×"
    warn: str = "⚠"
    dots: str = "…"


ASCII = Glyphs("|-- ", "`-- ", "|   ", "    ", "->", "x", "!", "...")
UNICODE = Glyphs()


def _b(n: Optional[int]) -> str:
    return "—" if n is None else f"{n:,}"


def _ratio(raw: Optional[int], csize: Optional[int], g: Glyphs) -> str:
    return f"{raw / csize:.2f}{g.times}" if raw and csize else ""


def _clip(text: str, n: int, g: Glyphs) -> str:
    return text if len(text) <= n else text[: n - len(g.dots)] + g.dots


_TO_ASCII = str.maketrans(
    {
        "→": "->",
        "×": "x",
        "…": "...",
        "·": "|",
        "—": "-",
        "⚠": "!",
        "├": "|",
        "└": "`",
        "│": "|",
        "─": "-",
    }
)


def _plain(text: str, g: Glyphs) -> str:
    if g is UNICODE:
        return text
    # Trace strings can hold any character; keep ASCII output ASCII.
    return text.translate(_TO_ASCII).encode("ascii", "replace").decode("ascii")


def _selection_label(trace: Trace, analysis: Analysis) -> str:
    if analysis.selection == "all":
        if analysis.chunk_count > 1:
            return f"all {analysis.chunk_count} chunks"
        return "whole trace"
    chunk = trace.chunks[analysis.selection]
    return (
        "top level only" if is_top_level(trace, chunk) else f"chunk {chunk.index} only"
    )


@allow_deep_trees
def render_summary(
    name: str,
    trace: Trace,
    analysis: Analysis,
    limit: int = 15,
    glyphs: Glyphs = UNICODE,
) -> str:
    g = glyphs
    a = analysis
    out: List[str] = []
    fmt = "CBOR" if trace.format == "cbor" else "DOT text"
    if trace.trace_version is not None:
        fmt += f" (trace version {trace.trace_version})"
    out.append(f"Codec review: {name}")
    out.append(f"{fmt} · {_selection_label(trace, a)}")
    facts = []
    if a.compression_failed and a.csize is None:
        size = f"{_b(a.input)} B in, " if a.input is not None else ""
        facts.append(f"{size}nothing written: compression failed")
    elif a.compression_failed:
        facts.append(
            f"{_b(a.input)} B in; compression failed after {_b(a.csize)} B in streams"
        )
    elif a.input is not None and a.csize is not None:
        ratio = _ratio(a.input, a.csize, g)
        facts.append(
            f"{_b(a.input)} B in {g.arrow} {_b(a.csize)} B in streams"
            + (f" ({ratio})" if ratio else "")
        )
    elif a.input is not None:
        facts.append(f"{_b(a.input)} B in")
    facts.append(f"{a.codecs_run:,} codec{'s' if a.codecs_run != 1 else ''} ran")
    distinct = sum(1 for p in a.pipelines if not p.feeder)
    facts.append(f"{distinct} distinct pipeline{'s' if distinct != 1 else ''}")
    out.append(" · ".join(facts))
    c = a.coverage
    status = "Coverage OK" if c.ok else f"{g.warn} Coverage INCOMPLETE"
    out.append(
        f"{status}: {c.nodes:,} of {c.nodes_total:,} codec nodes, "
        f"{c.stored:,} of {c.stored_total:,} stored bytes, "
        f"{c.split_outputs:,} of {c.split_outputs_total:,} split outputs, "
        f"{c.merges:,} of {c.merges_total:,} merges"
    )
    if a.failures or a.unfinished:
        parts = []
        if a.failures:
            parts.append(
                f"{a.failures} failure{'s' if a.failures != 1 else ''} recorded"
            )
        if a.unfinished:
            parts.append(
                f"{a.unfinished} stream{'s' if a.unfinished != 1 else ''} never compressed"
            )
        out.append(f"{g.warn} {'; '.join(parts)}; see FAILURES below")

    def section(heading: str, pipelines: List[Pipeline], note: str = "") -> None:
        if not pipelines:
            return
        out.append("")
        out.append(f"{heading} ({len(pipelines)})")
        if note:
            out.append(f"  {note}")
        out.append(
            f"  {'ID':<5} {'Bytes':>11} {'Ratio':>8} {'Codecs':>6} {'Runs':>5}  Pipeline"
        )
        shown = pipelines if limit <= 0 else pipelines[:limit]
        for p in shown:
            name_, desc = title(p, a)
            label = name_ + (f": {desc}" if desc else "")
            if p.failed:
                label += f"  {g.warn} failed"
            out.append(
                f"  {p.id:<5} {_b(p.csize):>11} {_ratio(p.raw, p.csize, g):>8} "
                f"{p.nodes:>6} {p.count:>5}  {_plain(label, g)}"
            )
            chain = main_chain(p)
            if chain:
                out.append(
                    f"  {'':<5} {'':>11} {'':>8} {'':>6} {'':>5}    {_plain(chain, g)}"
                )
        if len(pipelines) > len(shown):
            out.append(f"  {g.dots} {len(pipelines) - len(shown)} more (use --limit 0)")

    entries = [p for p in a.pipelines if p.kind == "entry"]
    merges = [p for p in a.pipelines if p.kind == "merge"]
    splits = [p for p in a.pipelines if p.kind == "split" and not p.feeder]
    feeders = [p for p in a.pipelines if p.kind == "split" and p.feeder]
    section("START", entries)
    section("MERGES", merges)
    for splitter in dict.fromkeys(p.splitter for p in splits):
        section(
            f"OUTPUTS OF {splitter}",
            [p for p in splits if p.splitter == splitter],
        )
    if feeders:
        runs = sum(p.count for p in feeders)
        section(
            "OUTPUTS THAT ONLY LEAD INTO A MERGE",
            feeders,
            f"{runs:,} outputs; their bytes are counted in the merges",
        )

    failures = _failures(a)
    if failures:
        out.append("")
        out.append(f"FAILURES ({len(failures)})")
        for pid, where, message in failures if limit <= 0 else failures[:limit]:
            out.append(
                f"  {pid:<5} {where}: {_clip(' '.join(message.split()), 160, g)}"
            )

    rows = a.census if limit <= 0 else a.census[: max(limit, 12)]
    out.append("")
    out.append(f"CODECS ({len(a.census)})")
    out.append(f"  {'Codec':<32} {'Runs':>7} {'Header B':>10} {'Stored B':>12}")
    for row in rows:
        out.append(
            f"  {_clip(row.codec, 32, g):<32} {row.runs:>7,} "
            f"{(f'{row.header:,}' if row.header else ''):>10} "
            f"{(f'{row.stored:,}' if row.stored else ''):>12}"
        )
    if len(a.census) > len(rows):
        out.append(f"  {g.dots} {len(a.census) - len(rows)} more (use --limit 0)")
    out.append("")
    example = a.pipelines[0].id if a.pipelines else "E1"
    out.append(
        f"Show a pipeline as a tree with --show ID (for example --show {example})."
    )
    return _plain("\n".join(out) + "\n", g)


def _failures(analysis: Analysis) -> List[Tuple[str, str, str]]:
    found = []
    for p in analysis.pipelines:
        for node in walk_nodes(p.tree):
            for message in node.failures:
                found.append((p.id, base_name(node.codec), message))
            for message in node.graph_failures:
                found.append((p.id, f"graph {short_name(node.graph)}", message))
            for child in node.children:
                if child.kind == "progress":
                    found += _progress_failures(p.id, child)
        if isinstance(p.tree, StreamRuns) and p.tree.kind == "progress":
            found += _progress_failures(p.id, p.tree)
    return found


def _progress_failures(pid: str, s: StreamRuns) -> List[Tuple[str, str, str]]:
    # A placeholder with several inputs is reported once, on its anchor input.
    if not s.progress_owner:
        return []
    where = f"stream #{s.index}"
    if s.progress_messages:
        return [(pid, where, m) for m in s.progress_messages]
    return [(pid, where, "never compressed: the trace ended before a codec took it")]


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Line:
    left: str
    right: str = ""


@allow_deep_trees
def render_pipeline(
    pipeline: Pipeline,
    analysis: Analysis,
    show_conversions: bool = False,
    glyphs: Glyphs = UNICODE,
) -> str:
    g = glyphs
    p = pipeline
    name, desc = title(p, analysis)
    lines = [f"{p.id}  {_plain(name, g)}" + (f" — {_plain(desc, g)}" if desc else "")]
    facts = []
    if p.raw is not None:
        facts.append(f"{_b(p.raw)} raw B {g.arrow} {_b(p.csize)} B")
    elif p.csize is not None:
        facts.append(f"{_b(p.csize)} B")
    ratio = _ratio(p.raw, p.csize, g)
    if ratio:
        facts.append(ratio)
    facts.append(f"{p.nodes} codec{'s' if p.nodes != 1 else ''} per run")
    facts.append(f"ran {p.count}{g.times}, sizes summed" if p.count > 1 else "ran once")
    if analysis.chunk_count > 1 and p.kind != "entry":
        label = "chunks" if len(p.chunks) > 1 else "chunk"
        facts.append(f"{label} {', '.join(map(str, p.chunks))}")
    lines.append("    " + " · ".join(facts))
    if p.kind == "split":
        lines.append(
            f"    {'Outputs' if len(p.outputs) > 1 else 'Output'} {number_list(p.outputs)} of {p.splitter}"
        )
    lines.append("")

    body: List[_Line] = []
    if isinstance(p.tree, StreamRuns):
        label = (
            f"Output #{p.outputs[0]}"
            if len(p.outputs) == 1
            else f"{len(p.outputs)} outputs"
        )
        source = f"{label} in" + (f" ({_b(p.raw)} B)" if p.raw is not None else "")
        if p.tree.kind == "store":
            source += " · stored as is"
        body.append(_Line(source, f"{_b(p.tree.csize)} B"))
        if p.tree.kind != "store" or p.tree.attempts:
            _stream_lines(p.tree, "", True, False, body, show_conversions, g)
    elif p.kind == "merge":
        body.append(
            _Line(
                f"{p.fanin} streams in"
                + (f" ({_b(p.raw)} B)" if p.raw is not None else ""),
                f"{_b(p.csize)} B",
            )
        )
        _node_lines(p.tree, "", True, None, [], body, show_conversions, g, p.csize)
    else:
        start = p.tree
        label = "Input file" if p.top_level else "Input"
        body.append(
            _Line(
                label + (f" ({_b(p.raw)} B)" if p.raw is not None else ""),
                f"{_b(p.csize)} B",
            )
        )
        stored = [c for c in start.children if c.kind == "store"]
        if stored:
            body[-1].left += f" · stored as is: {sum(c.csize for c in stored):,} B"
        _children_lines(start, "", body, show_conversions, g)
    width = min(max((len(line.left) for line in body if line.right), default=0) + 2, 96)
    for line in body:
        if line.right:
            pad = max(2, width - len(line.left))
            lines.append(f"{line.left}{' ' * pad}{line.right:>12}")
        else:
            lines.append(line.left)
    return _plain("\n".join(lines) + "\n", g)


def _message_lines(
    messages: List[str], prefix: str, out: List[_Line], g: Glyphs
) -> None:
    for message in messages:
        wrapped = textwrap.wrap(" ".join(message.split()), 100) or [""]
        if len(wrapped) > 8:
            wrapped = wrapped[:8] + [g.dots]
        out.append(_Line(prefix + f"{g.warn} {wrapped[0]}"))
        out.extend(_Line(prefix + "  " + line) for line in wrapped[1:])


def _stream_lines(
    s: StreamRuns,
    prefix: str,
    last: bool,
    numbered: bool,
    out: List[_Line],
    show_conversions: bool,
    g: Glyphs,
) -> None:
    tag = f"#{s.index} " if numbered else ""
    # Consumers that took the stream first and were abandoned come first.
    for attempt in s.attempts:
        _node_lines(
            attempt,
            prefix,
            False,
            tag,
            [],
            out,
            show_conversions,
            g,
            None,
            "  (abandoned)",
        )
    connector = g.last if last else g.tee
    via: List[str] = []
    end = s
    while (
        not show_conversions
        and end.kind == "node"
        and end.node is not None
        and is_conversion(end.node.codec)
        and not end.node.failed
        and end.node.split is None
        and len(end.node.children) == 1
        and end.node.children[0].kind != "store"
        and not end.node.children[0].attempts
    ):
        via.append(base_name(end.node.codec))
        end = end.node.children[0]
    if end.kind == "node" and end.node is not None:
        csize = s.csize if s.kept and not s.unfinished else None
        _node_lines(end.node, prefix, last, tag, via, out, show_conversions, g, csize)
        return
    suffix = f" (via {', '.join(via)})" if via else ""
    if end.kind == "junction":
        merges = ", ".join(end.merges)
        text = f"{tag}{g.arrow} {base_name(end.junction_codec or '')}, joins {merges}"
    elif end.kind == "progress" and end.unfinished:
        text = f"{tag}{g.warn} never compressed"
    elif end.kind == "progress":
        text = f"{tag}{g.warn} graph failed before running a codec"
    elif end.kind == "end":
        text = f"{tag}(no codec took this stream)"
    else:
        text = f"{tag}stored"
    # Abandoned or unfinished streams have no compressed size (the tracer records
    # their content size), so it is left out.
    sized = s.kept and not s.unfinished
    out.append(
        _Line(prefix + connector + text + suffix, f"{_b(s.csize)} B" if sized else "")
    )
    if end.kind == "progress":
        if not end.progress_owner:
            messages = ["stopped by the same failure as another input of this graph"]
        elif end.progress_messages or not end.unfinished:
            messages = end.progress_messages
        else:
            messages = ["the trace ended before a codec took this stream"]
        _message_lines(messages, prefix + (g.blank if last else g.pipe), out, g)


def _node_lines(
    node: NodeRuns,
    prefix: str,
    last: bool,
    tag: Optional[str],
    via: List[str],
    out: List[_Line],
    show_conversions: bool,
    g: Glyphs,
    csize: Optional[int],
    note: str = "",
) -> None:
    connector = g.last if last else g.tee
    graph = base_name(node.graph) if node.graph else ""
    if base_name(node.codec) == "#in_progress":
        # The tracer's placeholder for a graph that failed before running a codec.
        label = (tag or "") + (f"graph {graph}" if graph else "a graph")
        label += " failed before running a codec"
        if node.inputs > 1:
            label += f" on {node.inputs} streams"
        out.append(_Line(prefix + connector + f"{g.warn} " + label + note))
        child_prefix = prefix + (g.blank if last else g.pipe)
        _message_lines(node.failures + node.graph_failures, child_prefix, out, g)
        return
    meta = []
    if graph and graph != base_name(node.codec):
        meta.append(f"graph {graph}")
    meta += settings(node)
    stored = [c for c in node.children if c.kind == "store"]
    if stored:
        meta.append(f"writes {sum(c.csize for c in stored):,} B")
    if node.header:
        meta.append(f"header {node.header:,} B")
    label = (tag or "") + base_name(node.codec)
    if via:
        label += f" (via {', '.join(via)})"
    if node.failed:
        label += f"  {g.warn} FAILED"
    text = prefix + connector + label + note + ("  " + " · ".join(meta) if meta else "")
    out.append(_Line(text, f"{_b(csize)} B" if csize is not None else ""))
    child_prefix = prefix + (g.blank if last else g.pipe)
    _message_lines(node.failures + node.graph_failures, child_prefix, out, g)
    _children_lines(node, child_prefix, out, show_conversions, g)


def _children_lines(
    node: NodeRuns, prefix: str, out: List[_Line], show_conversions: bool, g: Glyphs
) -> None:
    kids = [c for c in node.children if c.kind != "store" or c.attempts]
    numbered = len(node.children) + (1 if node.split else 0) > 1
    items = len(kids) + (1 if node.split else 0)
    for i, child in enumerate(kids):
        # A stored child is only listed when an abandoned attempt preceded the store.
        _stream_lines(child, prefix, i == items - 1, numbered, out, show_conversions, g)
    if node.split:
        ids = node.split.pipelines
        listed = ", ".join(ids[:8]) + (
            f" and {len(ids) - 8} more" if len(ids) > 8 else ""
        )
        count = node.split.per_run
        runs = " per run" if node.split.runs > 1 else ""
        out.append(
            _Line(
                prefix
                + g.last
                + f"{g.arrow} {count:,} outputs{runs}, each its own pipeline: {listed}",
                f"{_b(node.split.csize)} B",
            )
        )
