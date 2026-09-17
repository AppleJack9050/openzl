# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Ratio-vs-speed results: validation, Pareto frontiers and a terminal table.

Pure functions only. A benchmark document (see bench.py) is untrusted once it
has been written to disk, so everything read back goes through normalize().
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

from text_report import UNICODE, Glyphs, _plain

FORMAT = "codec_reviewer.benchmark"
VERSION = 1
MAX_SERIES = 5
MAX_POINTS = 2000
MAX_SAMPLES = 50
MAX_TEXT = 2000
MAX_DOCUMENT = 8 * 1024 * 1024
AXES = ("c", "d")
BOTH = "cd"  # size against both speeds at once
FRONTIER_KEYS = (*AXES, BOTH)
FLAGS_AXIS = ("contended", "short")
FLAGS_BOTH = ("quick", "unpinned", "fallback")

TOOLS = ("zli", "zstd")
STATUSES = ("ok", "failed", "timeout", "skipped")
LINK_STATES = ("linked", "mismatch", "frame_mismatch", "no_point", "none")
CORE_CHOICES = ("auto", "fixed", "none")
PINS = ("taskset", "affinity", "none")
MAX_ERROR = 500
MAX_COMMAND = 64
MAX_NOTES = 20
MAX_ENV = 20
MAX_SLOT = 4
MIN_LEVEL = -50
MAX_LEVEL = 22
# Integers must survive a trip through a JavaScript number.
MAX_INT = 2**53 - 1
ZLI_DEFAULT_LEVEL = 6

_SERIES_ID = re.compile(r"[a-z0-9][a-z0-9:._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LINK_FIELDS = (
    "frame_bytes",
    "stream_bytes",
    "given_input_bytes",
    "given_stream_bytes",
)
# The terminal Frontier column, in the order of FRONTIER_KEYS.
_FRONTIER_MARKS = ("C", "D", "3D")
_FRONTIER_KEY_LINE = (
    "  Frontier: C = ratio vs compression speed, D = ratio vs decompression "
    "speed, 3D = ratio vs both speeds at once"
)

Point = dict[str, Any]
Doc = dict[str, Any]


class BenchmarkFormatError(ValueError):
    """The benchmark results are malformed or not a codec_reviewer benchmark."""


# ---------------------------------------------------------------------------
# Frontiers
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    try:
        return _is_number(value) and math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _speed_keys(axis: str) -> list[str]:
    return [k + "_speed" for k in axis]


def dominates(a: Point, b: Point, axis: str) -> bool:
    """True when a is at least as small as b and at least as fast at every speed
    the axis names ("c", "d", or "cd" for both), and strictly better in one."""
    keys = _speed_keys(axis)
    if a["bytes"] > b["bytes"] or any(a[k] < b[k] for k in keys):
        return False
    return a["bytes"] < b["bytes"] or any(a[k] > b[k] for k in keys)


def candidates(
    points: Iterable[Point], axis: str, series_ids: Iterable[str] | None = None
) -> list[Point]:
    """The points that can take part in a frontier on this axis."""
    wanted = None if series_ids is None else set(series_ids)
    keys = _speed_keys(axis)
    found = []
    for p in points:
        size = p.get("bytes")
        if (
            p.get("status") == "ok"
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            and all(_is_finite(p.get(k)) and p.get(k) > 0 for k in keys)
            and (wanted is None or p.get("series") in wanted)
        ):
            found.append(p)
    return found


def _sweep(cands: list[Point], axis: str) -> list[Point]:
    key = axis + "_speed"
    ordered = sorted(cands, key=lambda p: (p["bytes"], -p[key], p["id"]))
    kept: list[Point] = []
    best = -math.inf  # fastest speed among strictly smaller sizes
    i = 0
    while i < len(ordered):
        j = i
        size = ordered[i]["bytes"]
        while j < len(ordered) and ordered[j]["bytes"] == size:
            j += 1
        # Within one size the fastest points come first; only they can survive,
        # and only if nothing smaller is at least as fast.
        top = ordered[i][key]
        if top > best:
            kept.extend(p for p in ordered[i:j] if p[key] == top)
            best = top
        i = j
    return kept


def frontier(
    points: Iterable[Point], axis: str, series_ids: Iterable[str] | None = None
) -> list[str]:
    """Ids of the non-dominated candidates, best ratio first."""
    cands = candidates(points, axis, series_ids)
    if axis == BOTH:
        return [p["id"] for p in _sweep_both(sorted(cands, key=_order_both))]
    return [p["id"] for p in _sweep(cands, axis)]


def subset_key(doc: Doc, series_ids: Iterable[str]) -> str:
    """The key of a set of series: their ids in document order, joined by '+'."""
    wanted = set(series_ids)
    return "+".join(s["id"] for s in doc["series"] if s["id"] in wanted)


def _beaten_by(cands: list[Point], front: list[Point], axis: str) -> dict[str, str]:
    # Along a frontier sorted by size the speed strictly rises between sizes, so
    # the frontier size group with the largest size <= q's size is the fastest
    # one that is small enough; it dominates q whenever anything does. Its first
    # member has the lowest id (all members share size and speed).
    sizes = [p["bytes"] for p in front]
    first = []
    for i, p in enumerate(front):
        first.append(first[-1] if i and sizes[i - 1] == p["bytes"] else i)
    on_front = {id(p) for p in front}
    beaten: dict[str, str] = {}
    for q in sorted(cands, key=lambda p: p["id"]):
        if id(q) in on_front:
            continue
        at = bisect.bisect_right(sizes, q["bytes"]) - 1
        beaten[q["id"]] = front[first[at]]["id"]
    return beaten


def _triple(p: Point) -> tuple[int, float, float]:
    return p["bytes"], p["c_speed"], p["d_speed"]


def _order_both(p: Point) -> tuple[int, float, float, str]:
    return p["bytes"], -p["c_speed"], -p["d_speed"], p["id"]


def _sweep_both(ordered: list[Point]) -> list[Point]:
    """The three-way frontier of candidates sorted by _order_both, in that order.

    In this order a point can only be beaten by one before it. The (c, d) speeds
    of the frontier so far are kept as a staircase, c rising and d falling
    (stored negated, so both lists rise): the first step with c at least q's has
    the fastest d of all such steps. Equal points are kept or beaten together.
    """
    cs: list[float] = []
    neg_ds: list[float] = []
    kept: list[Point] = []
    for _, same in itertools.groupby(ordered, key=_triple):
        group = list(same)
        _, c, d = _triple(group[0])
        at = bisect.bisect_left(cs, c)
        if at < len(cs) and -neg_ds[at] >= d:
            continue
        kept.extend(group)
        # Steps at most as fast at both speeds are covered by this one.
        hi = bisect.bisect_right(cs, c)
        lo = bisect.bisect_left(neg_ds, -d, 0, hi)
        cs[lo:hi] = [c]
        neg_ds[lo:hi] = [-d]
    return kept


def _masks(bits: Iterable[int]) -> list[int]:
    """Running ORs of 1 << bit, after a leading 0."""
    masks = [0]
    for bit in bits:
        masks.append(masks[-1] | 1 << bit)
    return masks


def _beaten_by_both(cands: list[Point], front: list[Point]) -> dict[str, str]:
    # Of the frontier points that beat q at all three, q gets the one with the
    # largest size (the closest ratio, as in _beaten_by), then the fastest
    # compression, then the fastest decompression, then the lowest id. Each
    # frontier point gets a bit, the preferred one the highest; ANDing the
    # points no larger than q with those at least as fast at each speed leaves
    # exactly the points that beat q.
    pref = sorted(
        front, key=lambda p: (-p["bytes"], -p["c_speed"], -p["d_speed"], p["id"])
    )
    n = len(pref)
    bit = {id(p): n - 1 - i for i, p in enumerate(pref)}
    neg_sizes = [-p["bytes"] for p in pref]
    by_c = sorted(pref, key=lambda p: -p["c_speed"])
    by_d = sorted(pref, key=lambda p: -p["d_speed"])
    neg_c = [-p["c_speed"] for p in by_c]
    neg_d = [-p["d_speed"] for p in by_d]
    c_masks = _masks(bit[id(p)] for p in by_c)
    d_masks = _masks(bit[id(p)] for p in by_d)
    beaten: dict[str, str] = {}
    for q in sorted(cands, key=lambda p: p["id"]):
        if id(q) in bit:
            continue
        small = (1 << (n - bisect.bisect_left(neg_sizes, -q["bytes"]))) - 1
        fast = (
            c_masks[bisect.bisect_right(neg_c, -q["c_speed"])]
            & d_masks[bisect.bisect_right(neg_d, -q["d_speed"])]
        )
        beaten[q["id"]] = pref[n - (small & fast).bit_length()]["id"]
    return beaten


def frontiers(doc: Doc) -> dict[str, dict[str, dict[str, Any]]]:
    """Frontier and beaten-by map of every non-empty subset of the series, for each
    speed ("c", "d") and for both at once ("cd")."""
    ids = [s["id"] for s in doc["series"]]
    subsets = [
        combo
        for size in range(1, len(ids) + 1)
        for combo in itertools.combinations(ids, size)
    ]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in AXES:
        usable = candidates(doc["points"], axis)
        fronts: dict[str, list[str]] = {}
        beaten: dict[str, dict[str, str]] = {}
        for combo in subsets:
            members = set(combo)
            cands = [p for p in usable if p["series"] in members]
            front = _sweep(cands, axis)
            key = "+".join(combo)
            fronts[key] = [p["id"] for p in front]
            beaten[key] = _beaten_by(cands, front, axis)
        result[axis] = {"subsets": fronts, "beaten_by": beaten}
    # One sort for every subset: filtering keeps the order.
    usable = sorted(candidates(doc["points"], BOTH), key=_order_both)
    fronts_both: dict[str, list[str]] = {}
    beaten_both: dict[str, dict[str, str]] = {}
    for combo in subsets:
        members = set(combo)
        cands = [p for p in usable if p["series"] in members]
        front = _sweep_both(cands)
        key = "+".join(combo)
        fronts_both[key] = [p["id"] for p in front]
        beaten_both[key] = _beaten_by_both(cands, front)
    result[BOTH] = {"subsets": fronts_both, "beaten_by": beaten_both}
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _fail(message: str) -> BenchmarkFormatError:
    return BenchmarkFormatError(message)


def _valid(value: str, limit: int = MAX_TEXT) -> str:
    """value cut to limit, with lone surrogates (which cannot be written as
    UTF-8) replaced."""
    return value[:limit].encode("utf-8", "replace").decode("utf-8")


def _text(value: object, limit: int = MAX_TEXT) -> str:
    return _valid(value, limit) if isinstance(value, str) else ""


def _opt_text(value: object, limit: int = MAX_TEXT) -> str | None:
    return _valid(value, limit) if isinstance(value, str) else None


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(
    value: object, where: str, low: int = 0, high: int = MAX_INT, null: bool = True
) -> int | None:
    if value is None and null:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(f"{where} must be an integer.")
    if not low <= value <= high:
        raise _fail(f"{where} must be in {low:,}..{high:,}.")
    return value


def _float(value: object, where: str, positive: bool = False) -> float | None:
    if value is None and not positive:
        return None
    if not _is_number(value):
        raise _fail(f"{where} must be a number.")
    if not _is_finite(value):
        raise _fail(f"{where} must be a finite number.")
    if positive and value <= 0:
        raise _fail(f"{where} must be greater than 0.")
    if not positive and value < 0:
        raise _fail(f"{where} must not be negative.")
    return float(value)


def _bool(value: object, where: str, default: bool | None) -> bool | None:
    if value is None and default is not None:
        return default
    if value is None or isinstance(value, bool):
        return value
    raise _fail(f"{where} must be true or false.")


def _strings(value: object, limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        return []
    if not all(isinstance(v, str) for v in value):
        return []
    return [_valid(v) for v in value]


def _check_finite(doc: object) -> None:
    # "Non-finite numbers anywhere", including keys that are about to be dropped.
    stack = [doc]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            if not math.isfinite(item):
                raise _fail("The results contain a number that is not finite.")
        elif isinstance(item, (dict, list)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(item.values() if isinstance(item, dict) else item)


def _basename(name: str) -> str:
    return re.split(r"[\\/]", name)[-1]


def _input(value: object) -> dict[str, Any]:
    raw = _dict(value)
    sha = _text(raw.get("sha256"))
    return {
        "name": _basename(_text(raw.get("name"))),
        "bytes": _int(raw.get("bytes"), "input.bytes", null=False),
        "sha256": sha if _SHA256.fullmatch(sha) else "",
    }


def _machine(value: object) -> dict[str, Any]:
    raw = _dict(value)
    choice = raw.get("core_choice")
    pin = raw.get("pin")
    return {
        "cpu": _opt_text(raw.get("cpu")),
        "logical_cpus": _int(raw.get("logical_cpus"), "machine.logical_cpus"),
        "kernel": _text(raw.get("kernel")),
        "python": _text(raw.get("python")),
        "governor": _opt_text(raw.get("governor")),
        "core": _int(raw.get("core"), "machine.core"),
        "core_choice": choice if choice in CORE_CHOICES else "none",
        "pin": pin if pin in PINS else "none",
    }


def _tool(value: object, name: str) -> dict[str, Any]:
    raw = _dict(value)
    sha = _text(raw.get("sha256"))
    tool = {
        "path": _text(raw.get("path")),
        "sha256": sha if _SHA256.fullmatch(sha) else "",
        "bytes": _int(raw.get("bytes"), f"tools.{name}.bytes"),
        "version": _text(raw.get("version")),
    }
    if name == "zli":
        tool["build_dir"] = _opt_text(raw.get("build_dir"))
        tool["note"] = _text(raw.get("note"))
    return tool


def _settings(value: object) -> dict[str, Any]:
    raw = _dict(value)
    env = raw.get("env")
    kept_env: dict[str, str] = {}
    if isinstance(env, dict):
        for k, v in env.items():
            if len(kept_env) >= MAX_ENV:
                break
            if isinstance(k, str) and isinstance(v, str):
                kept_env[_valid(k)] = _valid(v)
    return {
        "levels": _text(raw.get("levels")),
        "rounds": _int(raw.get("rounds"), "settings.rounds", 1),
        "min_time_s": _float(raw.get("min_time_s"), "settings.min_time_s"),
        "quick": _bool(raw.get("quick"), "settings.quick", False),
        "timeout_s": _float(raw.get("timeout_s"), "settings.timeout_s"),
        "env": kept_env,
    }


def _series(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SERIES:
        raise _fail(f"The results must have 1 to {MAX_SERIES} series.")
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise _fail(f"Series {i + 1} is not an object.")
        sid = raw.get("id")
        if not isinstance(sid, str) or not _SERIES_ID.fullmatch(sid):
            raise _fail(f"Series {i + 1} has a bad id.")
        if sid in seen:
            raise _fail(f"Series {sid} appears twice.")
        seen.add(sid)
        if raw.get("tool") not in TOOLS:
            raise _fail(f"Series {sid} has an unknown tool.")
        found.append(
            {
                "id": sid,
                "tool": raw["tool"],
                "label": _text(raw.get("label")) or sid,
                "args": _strings(raw.get("args"), MAX_COMMAND),
                "slot": _int(raw.get("slot"), f"Series {sid} slot", 0, MAX_SLOT, False),
            }
        )
    return found


def _level_label(tool: str, level: int) -> str:
    if tool == "zli":
        return f"-l {level}"
    return f"--fast={-level}" if level < 0 else f"-{level}"


def _flags(value: object) -> dict[str, list[str]]:
    raw = _dict(value)
    flags = {}
    for part, known in (("c", FLAGS_AXIS), ("d", FLAGS_AXIS), ("both", FLAGS_BOTH)):
        given = raw.get(part)
        given = given if isinstance(given, list) else []
        flags[part] = [f for f in known if f in given]
    return flags


def _samples(value: object, where: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_SAMPLES:
        raise _fail(f"{where} must be a list of at most {MAX_SAMPLES} numbers.")
    return [_float(v, where, positive=True) for v in value]


def _point(raw: object, index: int, tools: dict[str, str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _fail(f"Point {index + 1} is not an object.")
    series = raw.get("series")
    if not isinstance(series, str) or series not in tools:
        raise _fail(f"Point {index + 1} belongs to an unknown series.")
    level = raw.get("level")
    if (
        not isinstance(level, int)
        or isinstance(level, bool)
        or not MIN_LEVEL <= level <= MAX_LEVEL
        or level == 0
    ):
        raise _fail(
            f"Point {index + 1} of {series} has a bad level (use 1..{MAX_LEVEL}, "
            f"or -1..{MIN_LEVEL} for zstd's fast levels)."
        )
    if tools[series] == "zli" and level < 1:
        raise _fail(f"Point {index + 1} of {series}: zli levels start at 1.")
    pid = f"{series}/{level}"
    if raw.get("id") != pid:
        raise _fail(f"Point {index + 1} should have the id {pid}.")
    status = raw.get("status")
    if status not in STATUSES:
        raise _fail(f"Point {pid} has an unknown status.")
    size = _int(raw.get("bytes"), f"Point {pid} bytes", null=True)
    c_speed = _float(raw.get("c_speed"), f"Point {pid} c_speed")
    d_speed = _float(raw.get("d_speed"), f"Point {pid} d_speed")
    c_samples = _samples(raw.get("c_samples"), f"Point {pid} c_samples")
    d_samples = _samples(raw.get("d_samples"), f"Point {pid} d_samples")
    ok = status == "ok"
    if ok:
        if size is None or size <= 0:
            raise _fail(f"Point {pid} is ok but has no compressed size.")
        _float(c_speed, f"Point {pid} c_speed", positive=True)
        _float(d_speed, f"Point {pid} d_speed", positive=True)
    return {
        "id": pid,
        "series": series,
        "level": level,
        "level_label": _text(raw.get("level_label"))
        or _level_label(tools[series], level),
        "status": status,
        "error": None if ok else _opt_text(raw.get("error"), MAX_ERROR),
        "bytes": size if ok else None,
        "c_speed": c_speed if ok else None,
        "d_speed": d_speed if ok else None,
        "c_samples": c_samples if ok else [],
        "d_samples": d_samples if ok else [],
        "flags": _flags(raw.get("flags")),
        "command": _strings(raw.get("command"), MAX_COMMAND),
    }


def _trace_link(value: object, series: set[str], ok_points: set[str]) -> dict[str, Any]:
    if value is None:
        value = {"state": "none"}
    if not isinstance(value, dict) or value.get("state") not in LINK_STATES:
        raise _fail("trace_link.state is not one of " + ", ".join(LINK_STATES) + ".")
    link: dict[str, Any] = {"state": value["state"]}
    sid = value.get("series")
    pid = value.get("point")
    link["series"] = sid if isinstance(sid, str) and sid in series else None
    link["point"] = pid if isinstance(pid, str) and pid in ok_points else None
    for field in _LINK_FIELDS:
        link[field] = _int(value.get(field), f"trace_link.{field}")
    link["verified"] = _bool(value.get("verified"), "trace_link.verified", None)
    return link


def normalize(doc: object) -> Doc:
    """Validate untrusted benchmark results; return a clean copy with fresh frontiers."""
    if not isinstance(doc, dict):
        raise _fail("The results are not a JSON object.")
    if doc.get("format") != FORMAT:
        raise _fail("This is not a codec_reviewer benchmark file.")
    version = doc.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version != VERSION:
        raise _fail(f"Unsupported benchmark version (expected {VERSION}).")
    _check_finite(doc)
    if not isinstance(doc.get("complete"), bool):
        raise _fail("complete must be true or false.")
    series = _series(doc.get("series"))
    points_raw = doc.get("points")
    if not isinstance(points_raw, list):
        raise _fail("points must be a list.")
    if len(points_raw) > MAX_POINTS:
        raise _fail(f"The results have more than {MAX_POINTS:,} points.")
    tools = {s["id"]: s["tool"] for s in series}
    points = []
    seen: set[str] = set()
    for i, raw in enumerate(points_raw):
        point = _point(raw, i, tools)
        if point["id"] in seen:
            raise _fail(f"Point {point['id']} appears twice.")
        seen.add(point["id"])
        points.append(point)
    notes = doc.get("notes")
    notes = notes if isinstance(notes, list) else []
    tools_raw = _dict(doc.get("tools"))
    clean: Doc = {
        "format": FORMAT,
        "version": VERSION,
        "complete": doc["complete"],
        "stopped": "interrupted" if doc.get("stopped") == "interrupted" else None,
        "created": _text(doc.get("created")),
        "elapsed_s": _float(doc.get("elapsed_s"), "elapsed_s"),
        "input": _input(doc.get("input")),
        "machine": _machine(doc.get("machine")),
        "tools": {name: _tool(tools_raw.get(name), name) for name in TOOLS},
        "settings": _settings(doc.get("settings")),
        "series": series,
        "points": points,
        "trace_link": _trace_link(
            doc.get("trace_link"),
            set(tools),
            {p["id"] for p in points if p["status"] == "ok"},
        ),
        "notes": [_valid(n) for n in notes if isinstance(n, str)][:MAX_NOTES],
    }
    clean["frontiers"] = frontiers(clean)
    return clean


def _reject_constant(name: str) -> None:
    raise _fail(f"The results contain {name}, which is not a finite number.")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        _reject_constant(text)
    return value


def loads(text: str | bytes | bytearray) -> Doc:
    """Parse and validate a benchmark document (UTF-8 JSON)."""
    if len(text) > MAX_DOCUMENT:
        raise _fail(f"The results are larger than {MAX_DOCUMENT:,} bytes.")
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise _fail("The results are not valid UTF-8.") from None
    try:
        doc = json.loads(
            text, parse_constant=_reject_constant, parse_float=_parse_float
        )
    except BenchmarkFormatError:
        raise
    except RecursionError:
        raise _fail("The results are nested too deeply.") from None
    except ValueError as e:
        raise _fail(f"The results are not valid JSON: {e}") from None
    return normalize(doc)


def dumps(doc: Doc) -> str:
    """Compact JSON; refuses NaN and infinities."""
    return json.dumps(doc, separators=(",", ":"), allow_nan=False)


def summary(doc: Doc) -> dict[str, Any]:
    """The short description the server lists next to a review."""
    return {
        "input": doc["input"]["name"],
        "bytes": doc["input"]["bytes"],
        "series": [s["label"] for s in doc["series"]],
        "points": sum(1 for p in doc["points"] if p["status"] == "ok"),
        "complete": doc["complete"],
    }


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    # Strings come from a file; keep escape sequences out of the terminal.
    text = " ".join(text.split())
    return "".join("?" if unicodedata.category(ch)[0] == "C" else ch for ch in text)


def _cut(text: str, n: int, g: Glyphs) -> str:
    dots = getattr(g, "dots", "...")
    return text if len(text) <= n else text[: n - len(dots)] + dots


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _speed(value: float) -> str:
    if value >= 10:
        return f"{value:,.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _facts(doc: Doc, g: Glyphs) -> list[str]:
    inp = doc["input"]
    machine = doc["machine"]
    settings_ = doc["settings"]
    facts = [_clean(inp["name"]) or "(unnamed input)", f"{inp['bytes']:,} B"]
    if machine["pin"] == "none" or machine["core"] is None:
        facts.append("unpinned")
    else:
        facts.append(f"one core (core {machine['core']})")
    if settings_["quick"]:
        facts.append("quick: one pass")
    elif settings_["rounds"]:
        up_to = "up to " if _fewer_runs(doc) else ""
        facts.append(f"best of {up_to}{_plural(settings_['rounds'], 'round')}")
    lines = ["RATIO VS SPEED  " + " · ".join(facts)]
    tools = []
    used = {s["tool"] for s in doc["series"]}
    zli = doc["tools"]["zli"]
    if "zli" in used:
        path = _clean(zli["path"]) or "zli"
        build = zli["build_dir"]
        if build and len(build) > 8 and build in path:
            path = path.replace(build, build[:4] + getattr(g, "dots", "..."))
        text = f"zli: {path}"
        if zli["sha256"]:
            text += f" (sha256 {zli['sha256'][:8]})"
        if zli["note"]:
            text += f" · {_clean(zli['note'])}"
        tools.append(text)
    zstd = doc["tools"]["zstd"]
    if "zstd" in used:
        tools.append(f"zstd {_clean(zstd['version']) or _clean(zstd['path'])}".rstrip())
    if tools:
        lines.append("   ".join(tools))
    return lines


def _notes_cell(p: Point, traced: bool) -> str:
    parts = []
    if p["status"] != "ok":
        error = _clean(p["error"] or "")
        parts.append(p["status"] + (f": {error}" if error else ""))
        return "; ".join(parts)
    if traced:
        parts.append("traced")
    flags = p["flags"]
    for flag, words in (("contended", "core busy"), ("short", "short run")):
        axes = [a.upper() for a in AXES if flag in flags[a]]
        if axes:
            parts.append(f"{words} ({' '.join(axes)})")
    words_both = {
        "fallback": "fell back to generic",
        "quick": "quick",
        "unpinned": "unpinned",
    }
    parts.extend(words_both[f] for f in flags["both"])
    return "; ".join(parts)


def _link_sentences(
    doc: Doc, labels: dict[str, str], by_id: dict[str, Point], g: Glyphs
) -> list[str]:
    link = doc["trace_link"]
    state = link["state"]
    warn = getattr(g, "warn", "!")
    series = labels.get(link["series"] or "", "the traced series")
    if state == "linked":
        point = by_id.get(link["point"] or "")
        if point is None:
            return []
        label = _clean(point["level_label"])
        lines = [f"The review above is the {label} run of {labels[point['series']]}."]
        if link["verified"] is False:
            lines.append(
                f"{warn} The traced frame did not decompress back to the input."
            )
        return lines
    if state == "mismatch":
        text = (
            f"{warn} The review above is not one of these points: the given trace does "
            f"not match what {series} produces with these settings"
        )
        given = (link["given_input_bytes"], link["given_stream_bytes"])
        if None not in given and link["stream_bytes"] is not None:
            text += (
                f" ({given[0]:,} B in, {given[1]:,} B in streams; the benchmark's "
                f"trace has {doc['input']['bytes']:,} B in, "
                f"{link['stream_bytes']:,} B in streams)"
            )
        return [text + "."]
    if state == "frame_mismatch":
        text = f"{warn} The review above is not one of these points: {series}"
        if link["frame_bytes"] is not None:
            text += f" wrote a {link['frame_bytes']:,} B frame when traced, which"
        text += (
            f" does not match its -l {ZLI_DEFAULT_LEVEL} result"
            " (the profile picks its own level)."
        )
        return [text]
    if state == "no_point":
        text = (
            f"{warn} The review above cannot be matched to a point: {series} has no "
            f"successful -l {ZLI_DEFAULT_LEVEL} result."
        )
        return [text]
    return []


def _fewer_runs(doc: Doc) -> int:
    """Measured points with fewer runs than the settings asked for."""
    settings_ = doc["settings"]
    if settings_["quick"] or not settings_["rounds"]:
        return 0
    return sum(
        1
        for p in doc["points"]
        if p["status"] == "ok" and len(p["c_samples"]) < settings_["rounds"]
    )


def render_table(doc: Doc, glyphs: Glyphs = UNICODE) -> str:
    """The results as a plain-text table with the facts a reader needs to trust them."""
    g = glyphs
    times = getattr(g, "times", "x")
    warn = getattr(g, "warn", "!")
    out = _facts(doc, g)
    series_list = doc["series"]
    labels = {s["id"]: _cut(_clean(s["label"]), 32, g) for s in series_list}
    by_id = {p["id"]: p for p in doc["points"]}
    fronts = doc.get("frontiers")
    if not fronts or any(k not in fronts for k in FRONTIER_KEYS):
        fronts = frontiers(doc)
    everything = subset_key(doc, [s["id"] for s in series_list])
    on_front = {
        key: set(fronts[key]["subsets"].get(everything, [])) for key in FRONTIER_KEYS
    }
    link = doc["trace_link"]
    traced = link["point"] if link["state"] == "linked" else None
    total = doc["input"]["bytes"]

    rows: list[tuple[str, ...]] = []
    for s in series_list:
        points = sorted(
            (p for p in doc["points"] if p["series"] == s["id"]),
            key=lambda p: p["level"],
        )
        for p in points:
            label, level = labels[s["id"]], _clean(p["level_label"])
            notes = _cut(_notes_cell(p, p["id"] == traced), 120, g)
            if p["status"] != "ok":
                rows.append((label, level, "—", "—", "—", "—", "", notes))
                continue
            speeds = []
            for axis in AXES:
                mark = "~" if p["flags"][axis] else " "
                speeds.append(_speed(p[axis + "_speed"]) + mark)
            front = " ".join(
                tok if p["id"] in on_front[key] else " " * len(tok)
                for key, tok in zip(FRONTIER_KEYS, _FRONTIER_MARKS)
            ).rstrip()
            rows.append(
                (
                    label,
                    level,
                    f"{p['bytes']:,}",
                    f"{total / p['bytes']:.2f}{times}",
                    speeds[0],
                    speeds[1],
                    front,
                    notes,
                )
            )
    head = (
        "Series",
        "Level",
        "Bytes",
        "Ratio",
        "Comp MB/s ",
        "Decomp MB/s ",
        "Frontier",
        "Notes",
    )
    # Measure what will be printed: ASCII output widens some characters.
    rows = [tuple(_plain(cell, g) for cell in r) for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(head)]
    left = (True, True, False, False, False, False, True, True)

    def line(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells[:-1]):
            parts.append(cell.ljust(widths[i]) if left[i] else cell.rjust(widths[i]))
        return ("  " + "  ".join(parts) + "  " + cells[-1]).rstrip()

    out.append(line(head))
    out.extend(line(r) for r in rows)
    if not rows:
        out.append("  (no points)")

    points = doc["points"]
    ok = [p for p in points if p["status"] == "ok"]
    if ok:
        out.append(_FRONTIER_KEY_LINE)
    banners = []
    busy = sum(1 for p in ok if any("contended" in p["flags"][a] for a in AXES))
    if busy:
        verb = "point ran" if busy == 1 else "points ran"
        banners.append(f"{busy:,} {verb} while the core was busy (marked ~)")
    short = sum(1 for p in ok if any("short" in p["flags"][a] for a in AXES))
    if short:
        banners.append(
            f"{_plural(short, 'point')} timed less than 0.05 s per run (marked ~)"
        )
    skipped = sum(1 for p in points if p["status"] == "skipped")
    if doc["stopped"] == "interrupted" or not doc["complete"]:
        lost = []
        if skipped:
            lost.append(f"{skipped:,} of {_plural(len(points), 'point')} not measured")
        fewer = _fewer_runs(doc)
        if fewer:
            runs = _plural(doc["settings"]["rounds"], "run")
            lost.append(f"{_plural(fewer, 'point')} with fewer than {runs}")
        what = "interrupted" if doc["stopped"] == "interrupted" else "incomplete"
        banners.append(what + (": " + "; ".join(lost) if lost else ""))
    failed = sum(1 for p in points if p["status"] == "failed")
    timeouts = sum(1 for p in points if p["status"] == "timeout")
    if failed or timeouts:
        parts = []
        if failed:
            parts.append(f"{_plural(failed, 'point')} failed")
        if timeouts:
            parts.append(f"{_plural(timeouts, 'point')} timed out")
        banners.append(" and ".join(parts) + " (see Notes)")
    if points and not ok:
        banners.append("no point was measured successfully")
    if doc["machine"]["pin"] == "none" or any(
        "unpinned" in p["flags"]["both"] for p in ok
    ):
        banners.append(
            "the runs were not pinned to one core, so speeds are less repeatable"
        )
    fell_back: dict[str, list[str]] = {}
    for p in ok:
        if "fallback" in p["flags"]["both"]:
            fell_back.setdefault(p["series"], []).append(_clean(p["level_label"]))
    if fell_back:
        which = "; ".join(
            f"{labels[sid]} {', '.join(levels)}" for sid, levels in fell_back.items()
        )
        banners.append(f"OpenZL fell back to generic compression for {which}")
    out.extend(f"{warn} {b}" for b in banners)
    out.extend(_link_sentences(doc, labels, by_id, g))
    out.extend(_cut(_clean(n), 300, g) for n in doc["notes"])
    return _plain("\n".join(out) + "\n", g)
