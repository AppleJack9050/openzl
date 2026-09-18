# Copyright (c) Meta Platforms, Inc. and affiliates.
"""JSON export of an analysis, and the self-contained interactive HTML report
(with ratio-vs-speed results when there are any)."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pareto
from pipelines import (
    allow_deep_trees,
    Analysis,
    NodeRuns,
    Pipeline,
    StreamRuns,
    analyze,
    blobs,
    chunk_summaries,
    main_chain,
    number_list,
    settings,
    short_name,
    title,
)
from trace_format import Trace

MAX_CHUNK_VIEWS = 64
TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "report_template.html"
)
TOOL = "tools/codec_reviewer"
DATA_MARKER = "__DATA__"
BENCH_MARKER = "__BENCH__"
MARKERS = ("__TITLE__", DATA_MARKER, BENCH_MARKER)


def _node(n: NodeRuns) -> Dict[str, object]:
    data: Dict[str, object] = {
        "c": short_name(n.codec),
        "g": short_name(n.graph) if n.graph else "",
        "gt": n.graph_type,
        "s": settings(n),
        "bl": blobs(n),
        "h": n.header,
        "f": n.failures + n.graph_failures,
        "k": [_stream(c) for c in n.children],
    }
    if n.split:
        data["sp"] = {
            "outputs": n.split.outputs,
            "per_run": n.split.per_run,
            "runs": n.split.runs,
            "raw": n.split.raw,
            "b": n.split.csize,
            "pipelines": n.split.pipelines,
        }
    return data


def _stream(s: StreamRuns) -> Dict[str, object]:
    data: Dict[str, object] = {
        "i": s.index,
        "t": s.type,
        "w": s.width,
        "e": s.elements,
        "r": s.raw,
        "b": s.csize,
        "kind": s.kind,
    }
    if s.node is not None:
        data["n"] = _node(s.node)
    if s.kind == "junction":
        data["j"] = short_name(s.junction_codec or "")
        data["m"] = s.merges
    if s.attempts:
        data["fa"] = [_node(a) for a in s.attempts]
    if s.progress_messages:
        data["pm"] = s.progress_messages
    if not s.progress_owner:
        data["po"] = False
    if not s.kept:
        data["x"] = True
    if s.unfinished:
        data["u"] = True
    return data


def _members(p: Pipeline, a: Analysis) -> str:
    if p.kind == "split":
        text = f"{'Outputs' if len(p.outputs) > 1 else 'Output'} {number_list(p.outputs, 14)} of {p.splitter}"
        if a.chunk_count > 1:
            text += (
                ", in every chunk"
                if len(p.chunks) == a.chunk_count
                else f", in chunks {', '.join(map(str, p.chunks))}"
            )
        return text + "."
    if p.kind == "merge":
        return f"{p.fanin} input streams." if not p.origins else ""
    if a.chunk_count > 1 and not p.top_level:
        return f"Chunks {', '.join(map(str, p.chunks))}."
    return ""


def pipeline_to_json(p: Pipeline, a: Analysis) -> Dict[str, object]:
    name, desc = title(p, a)
    return {
        "id": p.id,
        "kind": p.kind,
        "name": name,
        "desc": desc,
        "chain": main_chain(p),
        "members": _members(p, a),
        "count": p.count,
        "chunks": p.chunks,
        "raw": p.raw,
        "bytes": p.csize,
        "nodes": p.nodes,
        "stored": p.stored,
        "codecs": sorted(p.codecs),
        "failed": p.failed,
        "feeder": p.feeder,
        "splitter": p.splitter,
        "outputs": p.outputs,
        "fanin": p.fanin,
        "top_level": p.top_level,
        "chunk_count": p.chunk_count,
        "tree": _stream(p.tree) if isinstance(p.tree, StreamRuns) else _node(p.tree),
    }


def analysis_to_json(a: Analysis) -> Dict[str, object]:
    c = a.coverage
    return {
        "selection": a.selection,
        "chunk_count": a.chunk_count,
        "input": a.input,
        "bytes": a.csize,
        "codecs_run": a.codecs_run,
        "failures": a.failures,
        "unfinished": a.unfinished,
        "compression_failed": a.compression_failed,
        "distinct": sum(1 for p in a.pipelines if not p.feeder),
        "coverage": {
            "ok": c.ok,
            "nodes": c.nodes,
            "nodes_total": c.nodes_total,
            "stored": c.stored,
            "stored_total": c.stored_total,
            "split_outputs": c.split_outputs,
            "split_outputs_total": c.split_outputs_total,
            "merges": c.merges,
            "merges_total": c.merges_total,
        },
        "pipelines": [pipeline_to_json(p, a) for p in a.pipelines],
        "census": [
            {"codec": r.codec, "runs": r.runs, "header": r.header, "stored": r.stored}
            for r in a.census
        ],
    }


@allow_deep_trees
def report_data(
    name: str, trace: Trace, analysis: Optional[Analysis] = None
) -> Dict[str, object]:
    """Everything the HTML report shows: the whole trace plus per-chunk views."""
    summaries = chunk_summaries(trace)
    body = [s for s in summaries if not s.top_level]
    selections: List[Dict[str, object]] = [
        {
            "key": "all",
            "label": f"All {len(body)} chunks" if len(body) > 1 else "Whole trace",
            "analysis": analysis_to_json(analysis or analyze(trace)),
        }
    ]
    note = ""
    if len(body) > 1:
        if len(summaries) <= MAX_CHUNK_VIEWS:
            for s in summaries:
                label = (
                    "Top level (segmenter)"
                    if s.top_level
                    else f"Chunk {s.index}: {s.input:,} B {'→'} {s.csize:,} B"
                )
                selections.append(
                    {
                        "key": str(s.index),
                        "label": label,
                        "analysis": analysis_to_json(analyze(trace, s.index)),
                    }
                )
        else:
            note = f"Per-chunk views are left out: the trace has more than {MAX_CHUNK_VIEWS} chunks."
    return {
        "tool": TOOL,
        "file": name,
        "format": trace.format,
        "trace_version": trace.trace_version,
        "chunks": [
            {
                "index": s.index,
                "top_level": s.top_level,
                "input": s.input,
                "bytes": s.csize,
                "codecs": s.codecs,
            }
            for s in summaries
        ],
        "selections": selections,
        "note": note,
    }


@allow_deep_trees
def write_json(path: str, data: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # One-shot dumps uses the C encoder; streaming json.dump is quadratic in
        # nesting depth.
        f.write(json.dumps(data, separators=(",", ":")))
        f.write("\n")


def _script_json(text: str) -> str:
    # Strings from traces and results end up inside <script>; with <, > and &
    # escaped they cannot end the script or open a comment there.
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _page_title(data: Dict[str, object]) -> str:
    if data.get("bench_only"):
        title = (
            f"Ratio vs speed: {data['file']}" if data.get("file") else "Ratio vs speed"
        )
    else:
        title = (
            f"Codec review: {data['file']}" if data.get("file") else "Codec reviewer"
        )
    # A file name that is not valid text (Linux allows any bytes) still makes a page.
    return title.encode("utf-8", "replace").decode("utf-8")


@allow_deep_trees
def _fill_template(data: Dict[str, object]) -> Tuple[str, str, str]:
    """The filled-in page cut after the data and at the benchmark marker."""
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    for marker in MARKERS:
        if template.count(marker) != 1:
            raise ValueError(f"{TEMPLATE} must contain {marker} exactly once")
    page_title = _page_title(data)
    values = {
        "__TITLE__": page_title.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;"),
        "__DATA__": _script_json(json.dumps(data, separators=(",", ":"))),
    }

    def fill(part: str) -> str:
        # One pass, so a marker spelled inside a substituted value stays text.
        return re.sub("__TITLE__|__DATA__", lambda m: values[m.group(0)], part)

    # Split the template, never a filled-in page: a trace name may spell a marker.
    head, tail = template.split(BENCH_MARKER)
    if DATA_MARKER not in head:
        raise ValueError(f"{TEMPLATE} must have {DATA_MARKER} before {BENCH_MARKER}")
    before, after = head.split(DATA_MARKER)
    return fill(before) + values[DATA_MARKER], fill(after), fill(tail)


def render_parts(data: Dict[str, object]) -> Tuple[str, str]:
    """The page before and after the benchmark marker, with title and data filled in."""
    to_data, rest, tail = _fill_template(data)
    return to_data + rest, tail


def served_parts(
    data: Dict[str, object], served: Dict[str, object]
) -> Tuple[str, str, str, str]:
    """A served page cut around its ``served`` entry: (head, entry, rest, tail).

    ``head + entry + rest`` and ``tail`` are render_parts(dict(data, served=served)),
    since JSON writes the key added last just before the data's closing brace.
    Without the entry they are render_parts(data): the self-contained page. A server
    keeps one copy of the page for both.
    """
    if not data or "served" in data:
        raise ValueError("the page data must not be empty or served already")
    to_data, rest, tail = _fill_template(data)
    entry = json.dumps({"served": served}, separators=(",", ":"))[1:-1]
    return to_data[:-1], "," + _script_json(entry), "}" + rest, tail


def bench_payload(doc: Optional[Dict[str, object]]) -> str:
    """The JavaScript value that replaces the benchmark marker."""
    return "null" if doc is None else _script_json(pareto.dumps(doc))


def render_html(
    data: Dict[str, object], bench: Optional[Dict[str, object]] = None
) -> str:
    """The self-contained page; ``bench`` is a normalized benchmark document."""
    head, tail = render_parts(data)
    return head + bench_payload(bench) + tail


def write_html(
    path: str, data: Dict[str, object], bench: Optional[Dict[str, object]] = None
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(data, bench))


def bench_only_data(bench: Dict[str, object]) -> Dict[str, object]:
    """Page data for results shown without a trace."""
    return {
        "tool": TOOL,
        "file": bench["input"]["name"],
        "format": None,
        "trace_version": None,
        "chunks": [],
        "selections": [],
        "note": "",
        "empty": True,
        "bench_only": True,
    }
