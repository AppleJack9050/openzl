#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Review the codec pipelines recorded in a zli compression trace.

Record a trace, then review it:

    zli compress FILE -p parquet -o FILE.zl --trace FILE.cbor > FILE.dot
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor --show M1 --show S3
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor --html FILE.review.html
    python3 tools/codec_reviewer/codec_reviewer.py --serve 8765 [FILE.cbor ...]

Or let it record the trace and also measure ratio vs speed against zstd:

    python3 tools/codec_reviewer/codec_reviewer.py --bench FILE -p parquet --html FILE.review.html

The trace can be the CBOR file zli writes, the DOT text it prints to stdout, or
either one gzip-compressed. Only the Python standard library is needed.
"""

from __future__ import annotations

import argparse
import errno
import os
import signal
import sys
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
import html_report  # noqa: E402
import pareto  # noqa: E402
import pipelines  # noqa: E402
import review_server  # noqa: E402
import text_report  # noqa: E402
from trace_format import TraceFormatError, load_trace  # noqa: E402

# Stack size for threads that walk trace trees (see _with_deep_stack).
STACK_SIZE = 256 * 1024 * 1024


def _glyphs(stream) -> text_report.Glyphs:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "├└│→×⚠…—".encode(encoding)
        return text_report.UNICODE
    except (UnicodeEncodeError, LookupError):
        return text_report.ASCII


def _parse_chunk(value: str):
    if value == "all":
        return "all"
    if value == "top":
        return 0
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use 'all', 'top' or a chunk number") from None


def _zli_levels(value: str) -> str:
    try:
        bench.parse_levels(value, fast=False)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return value


def _zstd_levels(value: str) -> str:
    try:
        bench.parse_levels(value, fast=True)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return value


def _core(value: str):
    if value in ("auto", "none"):
        return value
    if value.isdigit():
        return int(value)
    raise argparse.ArgumentTypeError("use 'auto', 'none' or a CPU number")


def _int_in(low: int, high: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            number = low - 1
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"use a whole number from {low} to {high}")
        return number

    return parse


def _seconds_in(low: float, high: float):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError:
            number = float("nan")
        if not low <= number <= high:  # also refuses nan
            raise argparse.ArgumentTypeError(f"use seconds from {low:g} to {high:g}")
        return number

    return parse


# Options that only mean something with --bench, and the default each gets then.
BENCH_DEFAULTS = {
    "profiles": [],
    "profile_arg": None,
    "chunk_size_mb": None,
    "zstd_long": False,
    "zli_levels": bench.DEFAULT_ZLI_LEVELS,
    "zstd_levels": bench.DEFAULT_ZSTD_LEVELS,
    "rounds": 3,
    "min_time": 0.5,
    "quick": False,
    "core": "auto",
    "bench_timeout": 900.0,
    "zli": None,
    "zstd": None,
    "build_note": "",
}
BENCH_FLAGS = {
    "profiles": "-p/--profile",
    "profile_arg": "--profile-arg",
    "chunk_size_mb": "--chunk-size-mb",
    "zstd_long": "--zstd-long",
    "zli_levels": "--zli-levels",
    "zstd_levels": "--zstd-levels",
    "rounds": "--rounds",
    "min_time": "--min-time",
    "quick": "--quick",
    "core": "--core",
    "bench_timeout": "--bench-timeout",
    "zli": "--zli",
    "zstd": "--zstd",
    "build_note": "--build-note",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codec_reviewer.py",
        description=(
            "Group the codec graph of a `zli compress --trace` trace into distinct "
            "pipelines and review them: sizes, ratios, codecs, settings, merges, "
            "failures."
        ),
        epilog=(
            "Record a trace with: zli compress FILE -p PROFILE -o FILE.zl "
            "--trace FILE.cbor"
        ),
    )
    parser.add_argument(
        "traces",
        nargs="*",
        metavar="TRACE",
        help="trace file: .cbor, DOT text, or either gzipped; --serve takes any "
        "number of them",
    )
    parser.add_argument(
        "--chunk",
        type=_parse_chunk,
        default="all",
        metavar="N",
        help="review one chunk of a segmented trace: a chunk number, 'top' for "
        "the top-level segmenter run, or 'all' (default)",
    )
    parser.add_argument(
        "--show",
        action="append",
        default=[],
        metavar="ID",
        help="print pipeline ID (E1, M2, S3, ...) as a tree; repeat for several, "
        "or 'all'",
    )
    parser.add_argument(
        "--show-conversions",
        action="store_true",
        help="keep convert_* codecs in trees instead of folding them away",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        metavar="N",
        help="rows per section in the summary; 0 shows everything (default 15)",
    )
    parser.add_argument(
        "--html", metavar="OUT", help="also write an interactive HTML report"
    )
    parser.add_argument("--json", metavar="OUT", help="also write the analysis as JSON")
    parser.add_argument(
        "--serve",
        type=int,
        metavar="PORT",
        help="serve interactive reviews at http://HOST:PORT/ until interrupted, "
        "and open more traces from the page; if a reviewer already runs on PORT, "
        "add the traces to it. 0 picks a free port",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address for --serve (default 127.0.0.1: only this machine can connect)",
    )
    parser.add_argument(
        "--ascii", action="store_true", help="draw with ASCII characters only"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing unless --show is given (for use with --html/--json)",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="exit with status 3 if the coverage check does not add up, or if a "
        "ratio-vs-speed point was not measured",
    )

    group = parser.add_argument_group(
        "ratio vs speed",
        "Measure INPUT with zli and zstd at several levels, on one CPU core, and "
        "show the Pareto frontiers of compression ratio vs compression speed, "
        "decompression speed and both at once. Without a TRACE, the trace of the "
        "first profile is recorded too (zli compress, default level).",
    )
    group.add_argument(
        "--bench",
        metavar="INPUT",
        help="the uncompressed file to measure (the file the trace compressed)",
    )
    group.add_argument(
        "-p",
        "--profile",
        dest="profiles",
        action="append",
        default=None,
        metavar="PROFILE",
        help="zli profile to measure, one series each; repeat for up to 3",
    )
    group.add_argument(
        "--profile-arg",
        metavar="ARG",
        help="passed to zli as --profile-arg for the trace and every run",
    )
    group.add_argument(
        "--chunk-size-mb",
        type=_int_in(1, 2147),
        metavar="N",
        help="passed to zli (how zli cuts INPUT into chunks); --chunk only picks "
        "which chunk of a trace to review",
    )
    group.add_argument(
        "--zstd-long",
        action="store_true",
        default=None,
        help="also measure zstd --long=27 (a 128 MB window, like the one OpenZL "
        "gives text streams)",
    )
    group.add_argument(
        "--zli-levels",
        type=_zli_levels,
        metavar="SPEC",
        help="zli levels to measure, e.g. 1-9,12,19; level 6, zli's default, is "
        f"always added (default {bench.DEFAULT_ZLI_LEVELS})",
    )
    group.add_argument(
        "--zstd-levels",
        type=_zstd_levels,
        metavar="SPEC",
        help="zstd levels to measure, e.g. 1-19 or fast=5 for --fast=5 "
        f"(default {bench.DEFAULT_ZSTD_LEVELS})",
    )
    group.add_argument(
        "--rounds",
        type=_int_in(1, 20),
        metavar="N",
        help="passes over all points; each point keeps its best run (default 3)",
    )
    group.add_argument(
        "--min-time",
        type=_seconds_in(0, 60),
        metavar="SEC",
        help="timed work per zli run, at least (default 0.5)",
    )
    group.add_argument(
        "--quick",
        action="store_true",
        default=None,
        help="one pass, one iteration per point: fast, rough speeds",
    )
    group.add_argument(
        "--core",
        type=_core,
        metavar="CPU",
        help="CPU to run on: 'auto' picks the least busy one (default), 'none' "
        "does not pin",
    )
    group.add_argument(
        "--bench-timeout",
        type=_seconds_in(1, 86400),
        metavar="SEC",
        help="give up on a run after this long and skip higher levels of that "
        "series (default 900)",
    )
    group.add_argument(
        "--zli", metavar="PATH", help="zli to use (default $ZLI, then the repo's zli)"
    )
    group.add_argument(
        "--zstd", metavar="PATH", help="zstd to use (default $ZSTD, then PATH)"
    )
    group.add_argument(
        "--build-note",
        metavar="TEXT",
        help="how zli was built, shown next to the speeds (the binary cannot tell)",
    )
    group.add_argument(
        "--bench-json",
        metavar="OUT",
        help="also write the results as JSON, to reopen with --bench-results",
    )
    group.add_argument(
        "--bench-results",
        metavar="FILE",
        help="show results saved with --bench-json instead of measuring",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    # Trace files may come before, after or between options.
    args = parser.parse_intermixed_args(argv)
    given = [key for key in BENCH_DEFAULTS if getattr(args, key) is not None]
    for key, default in BENCH_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.bench is None:
        if given:
            flags = ", ".join(BENCH_FLAGS[key] for key in given)
            parser.error(f"{flags} only apply with --bench INPUT")
        if args.bench_json and args.bench_results is None:
            parser.error("--bench-json needs --bench INPUT or --bench-results FILE")
    elif args.bench_results is not None:
        parser.error("give --bench INPUT to measure, or --bench-results FILE, not both")
    else:
        if not args.profiles:
            parser.error("--bench needs at least one zli profile: -p PROFILE")
        if len(args.profiles) > 3 or len(set(args.profiles)) != len(args.profiles):
            parser.error("give 1 to 3 different profiles with -p")

    benchmarking = args.bench is not None or args.bench_results is not None
    if benchmarking:
        if len(args.traces) > 1:
            parser.error("with --bench or --bench-results, give at most one trace")
        has_trace = bool(args.traces) or args.bench is not None
        if args.show and not has_trace:
            parser.error("--show needs a trace")
    else:
        if args.serve is None and len(args.traces) != 1:
            parser.error("give one trace file, or --serve PORT with any number of them")
        if len(args.traces) != 1 and (args.show or args.html or args.json):
            parser.error("--show, --html and --json need exactly one trace file")
    return args


def main(argv: Optional[List[str]] = None, stdout=None) -> int:
    args = parse_args(argv)
    try:
        return _run(args, stdout or sys.stdout)
    except KeyboardInterrupt:
        print("codec_reviewer: interrupted", file=sys.stderr)
        return 130


def _with_deep_stack(fn, *args):
    """Run fn in a thread with a large stack and a high recursion limit.

    Trees are walked recursively; a chain of codecs thousands deep would
    overflow the main thread's C stack on some Python versions.
    """
    result: dict = {}

    def target() -> None:
        try:
            result["value"] = pipelines.allow_deep_trees(fn)(*args)
        except BaseException as e:  # re-raised in the calling thread
            result["error"] = e

    previous = threading.stack_size(STACK_SIZE)
    try:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _run(args: argparse.Namespace, stdout) -> int:
    if args.bench is not None or args.bench_results is not None:
        return _run_bench(args, stdout)
    if args.serve is None:
        return _with_deep_stack(_review, args, args.traces[0], stdout)[0]
    # Review every trace before touching the port: all of them or none.
    store = review_server.ReviewStore()
    reviews = _with_deep_stack(_review_all, args, stdout, store)
    if reviews is None:
        print(
            "codec_reviewer: nothing was served or added, because a trace failed",
            file=sys.stderr,
        )
        return 2
    return _serve(args, store, reviews)


def _review(
    args: argparse.Namespace,
    path: str,
    stdout,
    want_data: bool = False,
    name: Optional[str] = None,
    write: bool = True,
) -> Tuple[int, Optional[dict]]:
    """Review one trace as the options say: (exit status, report data or None).

    ``name`` is shown instead of the file name; ``write=False`` leaves --html and
    --json to the caller."""
    try:
        return _review_trace(args, path, stdout, want_data, name, write)
    except RecursionError:
        print(
            f"codec_reviewer: {path}: a pipeline is nested too deeply to review",
            file=sys.stderr,
        )
        return 2, None


def _review_trace(
    args: argparse.Namespace,
    path: str,
    stdout,
    want_data: bool,
    name: Optional[str],
    write: bool,
) -> Tuple[int, Optional[dict]]:
    name = name or bench.display_name(path)
    try:
        trace = load_trace(path)
    except OSError as e:
        print(f"codec_reviewer: cannot read {path}: {e.strerror or e}", file=sys.stderr)
        return 2, None
    except TraceFormatError as e:
        print(f"codec_reviewer: {path} is not a readable trace: {e}", file=sys.stderr)
        return 2, None
    try:
        analysis = pipelines.analyze(trace, args.chunk)
    except ValueError as e:
        print(f"codec_reviewer: {name}: {e}", file=sys.stderr)
        return 2, None

    glyphs = text_report.ASCII if args.ascii else _glyphs(stdout)
    if not args.quiet:
        stdout.write(
            text_report.render_summary(name, trace, analysis, args.limit, glyphs)
        )

    wanted = [x for item in args.show for x in item.split(",") if x]
    if any(x.lower() == "all" for x in wanted):
        wanted = [p.id for p in analysis.pipelines]
    for pid in wanted:
        pipeline = analysis.get(pid)
        if pipeline is None:
            known = ", ".join(p.id for p in analysis.pipelines[:20])
            print(
                f"codec_reviewer: no pipeline {pid} in this view (IDs: {known}"
                f"{', ...' if len(analysis.pipelines) > 20 else ''})",
                file=sys.stderr,
            )
            return 2, None
        stdout.write("\n")
        stdout.write(
            text_report.render_pipeline(
                pipeline, analysis, args.show_conversions, glyphs
            )
        )

    data = None
    if args.html or args.json or want_data:
        data = html_report.report_data(
            name, trace, analysis if args.chunk == "all" else None
        )
        if write and _write_outputs(args, data):
            return 2, None

    status = 0
    if args.fail_on_incomplete and not analysis.coverage.ok:
        print(
            f"codec_reviewer: {name}: the coverage check does not add up",
            file=sys.stderr,
        )
        status = 3
    return status, data


def _write_outputs(
    args: argparse.Namespace,
    data: Dict[str, object],
    shown: Optional[Dict[str, object]] = None,
    saved: Optional[Dict[str, object]] = None,
) -> bool:
    """Write --bench-json (the results as measured), --json and --html (with the
    results as shown) when asked; the paths that could not be written."""
    outputs = []
    # The measured results first: they are the costly part.
    if saved is not None and args.bench_json:
        text = pareto.dumps(saved)
        outputs.append((args.bench_json, lambda out: _write_text(out, text)))
    if args.json:
        json_data = dict(data, bench=shown) if shown is not None else data
        outputs.append((args.json, lambda out: html_report.write_json(out, json_data)))
    if args.html:
        outputs.append(
            (args.html, lambda out: html_report.write_html(out, data, shown))
        )
    failed = []
    for out_path, write in outputs:
        try:
            write(out_path)
        except OSError as e:
            _say(f"codec_reviewer: cannot write {out_path}: {e.strerror or e}")
            failed.append(out_path)
            continue
        _say(f"Wrote {out_path}")
    return failed


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def _review_all(
    args: argparse.Namespace, stdout, store: review_server.ReviewStore
) -> Optional[List[Tuple[str, int]]]:
    """Review the traces given with --serve into store: (path, status) per trace,
    or None if one of them failed."""
    reviews = []
    for n, path in enumerate(args.traces):
        if n and not args.quiet:
            stdout.write("\n")
        code, data = _review(args, path, stdout, want_data=True)
        if code == 2:
            return None
        try:
            store.add(data)
        except RecursionError:
            print(
                f"codec_reviewer: {path}: a pipeline is nested too deeply to review",
                file=sys.stderr,
            )
            return None
        reviews.append((path, code))
    return reviews


def _bind(args: argparse.Namespace, store: review_server.ReviewStore):
    """(server, None, None) when the port is ours, (None, listing, None) when a
    reviewer already serves on it, or (None, None, exit status) on failure."""
    host, port = args.host, args.serve
    # Uploaded traces are analyzed in request threads, which get this stack size.
    threading.stack_size(STACK_SIZE)
    try:
        return review_server.make_server(store, host, port), None, None
    except (OSError, OverflowError) as e:
        in_use = isinstance(e, OSError) and e.errno == errno.EADDRINUSE
        listing = review_server.running_reviewer(host, port) if in_use else None
        if listing is not None:
            return None, listing, None
        reason = getattr(e, "strerror", None) or e
        print(
            f"codec_reviewer: cannot serve on {host}:{port}: {reason}", file=sys.stderr
        )
        return None, None, 2


def _serve_forever(args: argparse.Namespace, server, loaded: str) -> None:
    url = review_server.url_for(args.host, server.server_address[1])
    try:
        print(
            f"Codec reviewer at {url} ({loaded}; Ctrl+C to stop)",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _serve(
    args: argparse.Namespace,
    store: review_server.ReviewStore,
    reviews: List[Tuple[str, int]],
) -> int:
    status = max((code for _, code in reviews), default=0)
    server, listing, failed = _bind(args, store)
    if failed is not None:
        return failed
    if server is None:
        return max(status, _send_to_running(reviews, args.host, args.serve, listing))
    count = len(reviews)
    _serve_forever(
        args,
        server,
        f"{count} trace{'s' if count != 1 else ''}; open more in the page"
        if count
        else "no trace yet; open one in the page",
    )
    return status


def _send_to_running(
    reviews: List[Tuple[str, int]], host: str, port: int, listing: dict
) -> int:
    """Add traces to the reviewer that already serves on the port."""
    url = review_server.url_for(host, port)
    if not reviews:
        print(f"A codec reviewer is already running at {url}", file=sys.stderr)
        return 0
    status = 0
    for path, _ in reviews:
        name = bench.display_name(path)
        if _upload_trace(path, name, host, port, listing) is None:
            status = 2
    return status


def _upload_trace(
    path: str, name: str, host: str, port: int, listing: dict
) -> Optional[int]:
    """Add one trace file to the running reviewer; its review id, or None."""
    url = review_server.url_for(host, port)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        print(f"codec_reviewer: cannot read {path}: {e.strerror or e}", file=sys.stderr)
        return None
    limit = listing.get("max_upload")
    if isinstance(limit, int) and len(raw) > limit:
        ok, message, review_id = (
            False,
            f"it is {len(raw):,} bytes; the reviewer accepts traces up to {limit:,} bytes",
            None,
        )
    else:
        ok, message, review_id = review_server.add_trace(host, port, name, raw)
    if not ok:
        print(f"codec_reviewer: could not add {name}: {message}", file=sys.stderr)
        return None
    print(f"Added {name} to the reviewer at {url}: {message}", file=sys.stderr)
    return review_id


# ---------------------------------------------------------------------------
# Ratio vs speed
# ---------------------------------------------------------------------------


def _bench_config(args: argparse.Namespace) -> bench.BenchConfig:
    return bench.BenchConfig(
        input=args.bench,
        zli=bench.find_tool(args.zli, "ZLI", [os.path.join(bench.REPO_ROOT, "zli")]),
        zstd=bench.find_tool(args.zstd, "ZSTD", []),
        profiles=list(args.profiles),
        profile_arg=args.profile_arg,
        chunk_size_mb=args.chunk_size_mb,
        zstd_long=27 if args.zstd_long else None,
        zli_levels=bench.parse_levels(args.zli_levels, fast=False),
        zstd_levels=bench.parse_levels(args.zstd_levels, fast=True),
        rounds=args.rounds,
        min_time=args.min_time,
        quick=args.quick,
        core=args.core,
        timeout=args.bench_timeout,
        build_note=args.build_note,
    )


def _load_results(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "rb") as f:
            return pareto.loads(f.read(pareto.MAX_DOCUMENT + 1))
    except OSError as e:
        print(f"codec_reviewer: cannot read {path}: {e.strerror or e}", file=sys.stderr)
    except pareto.BenchmarkFormatError as e:
        print(f"codec_reviewer: {path} is not a results file: {e}", file=sys.stderr)
    return None


def _totals(data: Dict[str, object]) -> Tuple[object, object]:
    """A reviewed trace's (input bytes, stream bytes)."""
    analysis = data["selections"][0]["analysis"]
    return analysis["input"], analysis["bytes"]


def _trace_totals(path: str) -> Optional[Tuple[object, object]]:
    try:
        analysis = pipelines.analyze(load_trace(path))
    except (OSError, TraceFormatError, ValueError, RecursionError):
        return None
    return analysis.input, analysis.csize


def _progress(line: str) -> None:
    _say(line)


def _output_problem(args: argparse.Namespace) -> Optional[str]:
    """Why an output file cannot be written, found before any long work."""
    for path in (args.bench_json, args.json, args.html):
        if not path:
            continue
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(path):
            return f"cannot write {path}: it is a directory"
        if os.path.exists(path):
            if not os.access(path, os.W_OK):
                return f"cannot write {path}: it is not writable"
        elif not os.path.isdir(folder):
            return f"cannot write {path}: there is no directory {folder}"
        elif not os.access(folder, os.W_OK):
            return f"cannot write {path}: {folder} is not writable"
    return None


class _Outcome:
    """What a --bench or --bench-results run has to serve or send when its
    temporary directory is gone."""

    def __init__(self, status, data, doc, trace_raw, interrupted, saved) -> None:
        self.status = status
        self.data = data
        self.doc = doc
        self.trace_raw = trace_raw
        self.interrupted = interrupted
        self.saved = saved
        """The results are in a file already (--bench-json or a kept copy)."""


# The termination signal that stopped a --bench run, if any.
_terminated: List[int] = []


def _raise_interrupt(signum, frame) -> None:
    _terminated.append(signum)
    if signum == signal.SIGHUP:
        # The terminal is gone: writing to it would fail and lose the results.
        devnull = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            os.dup2(devnull, fd)
        os.close(devnull)
    raise KeyboardInterrupt


def _run_bench(args: argparse.Namespace, stdout) -> int:
    """--bench or --bench-results: the review of one trace (given or recorded)
    together with ratio-vs-speed results (measured or loaded)."""
    config = probe = None
    loaded = None
    if args.bench_results is not None:
        loaded = _load_results(args.bench_results)
        if loaded is None:
            return 2
    else:
        try:
            config = _bench_config(args)
            probe = bench.CpuProbe()
            bench.check_setup(config, probe)
        except bench.BenchError as e:
            print(f"codec_reviewer: {e}", file=sys.stderr)
            return 2
    problem = _output_problem(args)
    if problem:
        print(f"codec_reviewer: {problem}", file=sys.stderr)
        return 2

    # Claim the port (or find the reviewer on it) before any long work.
    store = review_server.ReviewStore()
    server = listing = None
    if args.serve is not None:
        server, listing, failed = _bind(args, store)
        if failed is not None:
            return failed
        if listing is not None and "benchmark" not in (listing.get("features") or []):
            url = review_server.url_for(args.host, args.serve)
            print(
                f"codec_reviewer: the reviewer at {url} is an older version that "
                "cannot show ratio-vs-speed results; stop it and run this again",
                file=sys.stderr,
            )
            return 2
    # A closed terminal or a kill stops the run like Ctrl+C: the running tool's
    # process group is killed and the temporary files are removed. A signal the
    # caller ignores (nohup) stays ignored.
    handlers = {}
    del _terminated[:]
    for signum in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(signum) is not signal.SIG_IGN:
            handlers[signum] = signal.signal(signum, _raise_interrupt)
    try:
        with tempfile.TemporaryDirectory(prefix="codec_reviewer-") as workdir:
            outcome = _bench_in(args, stdout, config, probe, loaded, workdir, listing)
        if isinstance(outcome, int):
            return outcome
        if args.serve is None:
            return outcome.status
        if _terminated:
            # Killed, not interrupted from the keyboard: do not start serving.
            if not outcome.saved:
                _keep_results(outcome.doc)
            return outcome.status
        if server is not None:
            return _serve_outcome(args, server, store, outcome)
        return _send_outcome(args, listing, outcome)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if server is not None:
            server.server_close()


def _bench_in(
    args: argparse.Namespace,
    stdout,
    config: Optional[bench.BenchConfig],
    probe,
    doc: Optional[Dict[str, object]],
    workdir: str,
    listing: Optional[dict],
):
    """Review, record and measure in workdir: an _Outcome, or an exit status."""
    given = args.traces[0] if args.traces else None
    status, data, totals = 0, None, None
    # A given trace is reviewed first: a bad path or --show ID fails at once.
    trace_raw = None
    if given is not None:
        status, data = _with_deep_stack(_review, args, given, stdout, True, None, False)
        if status == 2:
            return 2
        totals = _totals(data)
        if listing is not None:
            # Sent to the running reviewer at the end; check its limit now.
            trace_raw = _trace_for(args, listing, given)
            if trace_raw is None:
                return 2

    recording = None
    if config is not None:
        profile = config.profiles[0]
        input_name = bench.display_name(config.input)
        _progress(f"Recording the trace: zli compress {input_name} -p {profile}")
        try:
            recording = bench.record_trace(config, workdir, probe)
        except bench.BenchError as e:
            print(f"codec_reviewer: {e}", file=sys.stderr)
            return 2
        if recording.error:
            print(
                f"codec_reviewer: zli compress -p {profile} failed: {recording.error}",
                file=sys.stderr,
            )
        if given is None:
            if recording.trace_path is None:
                return 2
            name = f"{input_name} (-p {profile})"
            status, data = _with_deep_stack(
                _review, args, recording.trace_path, stdout, True, name, False
            )
            if status == 2:
                return 2
            totals = _totals(data)
    if listing is not None and given is None and recording is not None:
        trace_raw = _trace_for(args, listing, recording.trace_path)
        if trace_raw is None:
            return 2

    if config is not None:
        if given is None:
            recorded = totals
        elif recording.trace_path is not None:
            recorded = _with_deep_stack(_trace_totals, recording.trace_path)
        else:
            recorded = None
        try:
            raw = bench.run_benchmark(config, probe, workdir, progress=_progress)
        except bench.BenchError as e:
            print(f"codec_reviewer: {e}", file=sys.stderr)
            return 2
        bench.link_trace(
            raw,
            f"zli:{config.profiles[0]}",
            recording,
            recorded,
            totals if given is not None else None,
        )
        doc = pareto.normalize(raw)
    interrupted = doc["stopped"] == "interrupted"

    shown = review_server.fit_benchmark(doc, totals)
    if data is None:
        data = html_report.bench_only_data(shown)
    # The files first: they are what a stopped run must not lose.
    try:
        failed = _with_deep_stack(_write_outputs, args, data, shown, doc)
    except RecursionError:
        _say(
            f"codec_reviewer: {data['file']}: a pipeline is nested too deeply to review"
        )
        failed = [args.json, args.html]
    saved = bool(args.bench_json) and args.bench_json not in failed
    if failed:
        status = 2
        if not saved and config is not None:
            _keep_results(doc)
            saved = True
    if not args.quiet:
        glyphs = text_report.ASCII if args.ascii else _glyphs(stdout)
        try:
            stdout.write("\n" + pareto.render_table(shown, glyphs))
        except OSError:
            pass

    unmeasured = sum(1 for p in doc["points"] if p["status"] != "ok")
    if args.fail_on_incomplete and (unmeasured or not doc["complete"]):
        print(
            f"codec_reviewer: {unmeasured} ratio-vs-speed point"
            f"{'s were' if unmeasured != 1 else ' was'} not measured",
            file=sys.stderr,
        )
        status = max(status, 3)
    if interrupted:
        status = 130
    return _Outcome(status, data, doc, trace_raw, interrupted, saved)


def _trace_for(args: argparse.Namespace, listing: dict, path: str) -> Optional[bytes]:
    """The trace to send to the running reviewer, if it can take it."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        print(f"codec_reviewer: cannot read {path}: {e.strerror or e}", file=sys.stderr)
        return None
    limit = listing.get("max_upload")
    if isinstance(limit, int) and len(raw) > limit:
        print(
            f"codec_reviewer: the trace is {len(raw):,} bytes; the reviewer at "
            f"{review_server.url_for(args.host, args.serve)} accepts traces up to "
            f"{limit:,} bytes",
            file=sys.stderr,
        )
        return None
    return raw


def _say(text: str) -> None:
    """A message on stderr; a closed terminal must not stop the run."""
    try:
        print(text, file=sys.stderr, flush=True)
    except OSError:
        pass


def _serve_outcome(args, server, store, outcome: _Outcome) -> int:
    try:
        _with_deep_stack(store.add, outcome.data, outcome.doc)
    except RecursionError:
        print(
            f"codec_reviewer: {outcome.data['file']}: a pipeline is nested too "
            "deeply to review",
            file=sys.stderr,
        )
        return 2
    if outcome.interrupted:
        _progress("The benchmark was interrupted; serving the points it finished.")
    _serve_forever(
        args,
        server,
        f"{outcome.data['file']} with ratio vs speed; open more in the page",
    )
    return outcome.status


def _send_outcome(args, listing: dict, outcome: _Outcome) -> int:
    """Add the review and its results to the reviewer already on the port."""
    host, port = args.host, args.serve
    url = review_server.url_for(host, port)
    title = outcome.data["file"]
    review_id = None
    status = outcome.status
    if outcome.trace_raw is not None:
        ok, message, review_id = review_server.add_trace(
            host, port, title, outcome.trace_raw
        )
        if ok:
            print(f"Added {title} to the reviewer at {url}: {message}", file=sys.stderr)
        else:
            print(f"codec_reviewer: could not add {title}: {message}", file=sys.stderr)
            status = 2
    try:
        ok, message = review_server.send_benchmark(host, port, outcome.doc, review_id)
        if not ok and review_id is not None:
            # The review may have gone (evicted, or the reviewer restarted).
            ok, message = review_server.send_benchmark(host, port, outcome.doc)
    except Exception as e:  # keep the measured results whatever went wrong
        ok, message = False, f"{type(e).__name__}: {e}"
    if ok:
        print(
            f"Added the ratio-vs-speed results to the reviewer at {url}: {message}",
            file=sys.stderr,
        )
        return status
    print(f"codec_reviewer: could not add the results: {message}", file=sys.stderr)
    if not outcome.saved:
        _keep_results(outcome.doc)
    return 2


def _keep_results(doc: Dict[str, object]) -> None:
    """Save results that could not go anywhere else, and say where."""
    try:
        fd, path = tempfile.mkstemp(prefix="codec_reviewer-", suffix=".bench.json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(pareto.dumps(doc))
            f.write("\n")
    except OSError as e:
        print(
            f"codec_reviewer: the results could not be saved either: {e.strerror or e}",
            file=sys.stderr,
        )
        return
    print(
        f"The results are saved in {path}; show them with --bench-results {path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
