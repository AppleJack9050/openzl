#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Review the codec pipelines recorded in a zli compression trace.

Record a trace, then review it:

    zli compress FILE -p parquet -o FILE.zl --trace FILE.cbor > FILE.dot
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor --show M1 --show S3
    python3 tools/codec_reviewer/codec_reviewer.py FILE.cbor --html FILE.review.html
    python3 tools/codec_reviewer/codec_reviewer.py --serve 8765 [FILE.cbor ...]

The trace can be the CBOR file zli writes, the DOT text it prints to stdout, or
either one gzip-compressed. Only the Python standard library is needed.
"""

from __future__ import annotations

import argparse
import errno
import os
import sys
import threading
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import html_report  # noqa: E402
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
        help="exit with status 3 if the coverage check does not add up",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    # Trace files may come before, after or between options.
    args = parser.parse_intermixed_args(argv)
    if args.serve is None and len(args.traces) != 1:
        parser.error("give one trace file, or --serve PORT with any number of them")
    if len(args.traces) != 1 and (args.show or args.html or args.json):
        parser.error("--show, --html and --json need exactly one trace file")
    return args


def main(argv: Optional[List[str]] = None, stdout=None) -> int:
    args = parse_args(argv)
    return _run(args, stdout or sys.stdout)


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
    args: argparse.Namespace, path: str, stdout, want_data: bool = False
) -> Tuple[int, Optional[dict]]:
    """Review one trace as the options say: (exit status, report data or None)."""
    try:
        return _review_trace(args, path, stdout, want_data)
    except RecursionError:
        print(
            f"codec_reviewer: {path}: a pipeline is nested too deeply to review",
            file=sys.stderr,
        )
        return 2, None


def _review_trace(
    args: argparse.Namespace, path: str, stdout, want_data: bool
) -> Tuple[int, Optional[dict]]:
    name = os.path.basename(path)
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
        for out_path, write in (
            (args.json, html_report.write_json),
            (args.html, html_report.write_html),
        ):
            if not out_path:
                continue
            try:
                write(out_path, data)
            except OSError as e:
                print(
                    f"codec_reviewer: cannot write {out_path}: {e.strerror or e}",
                    file=sys.stderr,
                )
                return 2, None
            print(f"Wrote {out_path}", file=sys.stderr)

    status = 0
    if args.fail_on_incomplete and not analysis.coverage.ok:
        print(
            f"codec_reviewer: {name}: the coverage check does not add up",
            file=sys.stderr,
        )
        status = 3
    return status, data


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


def _serve(
    args: argparse.Namespace,
    store: review_server.ReviewStore,
    reviews: List[Tuple[str, int]],
) -> int:
    host, port = args.host, args.serve
    status = max((code for _, code in reviews), default=0)
    # Uploaded traces are analyzed in request threads, which get this stack size.
    threading.stack_size(STACK_SIZE)
    try:
        server = review_server.make_server(store, host, port)
    except (OSError, OverflowError) as e:
        in_use = isinstance(e, OSError) and e.errno == errno.EADDRINUSE
        listing = review_server.running_reviewer(host, port) if in_use else None
        if listing is not None:
            return max(status, _send_to_running(reviews, host, port, listing))
        reason = getattr(e, "strerror", None) or e
        print(
            f"codec_reviewer: cannot serve on {host}:{port}: {reason}", file=sys.stderr
        )
        return 2
    try:
        count = len(reviews)
        loaded = (
            f"{count} trace{'s' if count != 1 else ''}; open more in the page"
            if count
            else "no trace yet; open one in the page"
        )
        print(
            f"Codec reviewer at {review_server.url_for(host, server.server_address[1])}"
            f" ({loaded}; Ctrl+C to stop)",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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
    limit = listing.get("max_upload")
    for path, _ in reviews:
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as e:
            print(
                f"codec_reviewer: cannot read {path}: {e.strerror or e}",
                file=sys.stderr,
            )
            status = 2
            continue
        if isinstance(limit, int) and len(raw) > limit:
            ok, message = (
                False,
                (
                    f"it is {len(raw):,} bytes; the reviewer accepts traces up to {limit:,} bytes"
                ),
            )
        else:
            ok, message = review_server.send_trace(host, port, name, raw)
        if ok:
            print(f"Added {name} to the reviewer at {url}: {message}", file=sys.stderr)
        else:
            print(f"codec_reviewer: could not add {name}: {message}", file=sys.stderr)
            status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())
