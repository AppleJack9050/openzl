# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Serve interactive reviews over HTTP, and review traces opened in the page.

Each review lives at ``/r/<id>`` and ``/`` goes to the newest one;
``/r/<id>/download`` saves a review as the self-contained page ``--html`` writes.
The page posts a trace file to ``/api/reviews``; the server analyzes it with the
same code as the command line and answers with the new review's address.

A review can also carry ratio-vs-speed results (``pareto.py``): the command line
measures them and posts the JSON document to ``/api/reviews/<id>/benchmark``, or
to ``/api/reviews`` for a page with results only. The server never runs a tool;
it validates the document and recomputes its frontiers.

The server is meant for one person on their own machine. When it listens on a
loopback address it answers only requests addressed to a loopback name, which
keeps other websites out through DNS rebinding. Uploads must carry the
``X-Codec-Reviewer`` header and, from a browser, come from the server's own
page, so other websites cannot post traces to it either.
"""

from __future__ import annotations

import collections
import http.client
import http.server
import ipaddress
import json
import re
import socket
import sys
import threading
import traceback
import urllib.parse
from typing import Dict, List, Optional, Tuple

import html_report
import pareto
from trace_format import load_trace_bytes, TraceFormatError

TOOL = "tools/codec_reviewer"
MAX_UPLOAD = 512 * 1024 * 1024
MAX_BENCH_UPLOAD = pareto.MAX_DOCUMENT
MAX_REVIEWS = 50
UPLOAD_HEADER = "X-Codec-Reviewer"
NAME_HEADER = "X-Trace-Name"
# What the listing advertises; the command line checks it before measuring.
FEATURES = ["benchmark"]
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}
# The page has inline script and style and needs nothing else except its own API.
_PAGE_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_wildcard(host: str) -> bool:
    return host in ("", "0.0.0.0", "::")


def trace_name(value: str) -> str:
    """A display name from the upload's (percent-encoded) file name."""
    name = urllib.parse.unquote(value or "")
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:200] or "uploaded trace"


def download_name(name: str, trace: bool = True) -> str:
    """The file name of a review's self-contained page: the trace's name without
    .gz and then .cbor or .dot (a page of results alone keeps its input's name),
    plus .review.html."""
    if trace:
        name = re.sub(r"\.gz$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\.(cbor|dot)$", "", name, flags=re.IGNORECASE)
    # Control and format characters, lone surrogates, path separators and what
    # Windows refuses in file names.
    name = "".join(
        ch if ch.isprintable() and ch not in '/\\:*?"<>|' else "_" for ch in name
    )
    return (name.strip(" .")[:200] or "review") + ".review.html"


def attachment(name: str) -> str:
    """A Content-Disposition that saves the response as ``name``."""
    # The quoted name is a fallback for clients without filename*: printable
    # ASCII only, and no % since some of them decode %XX in it.
    fallback = "".join(
        ch if " " <= ch <= "~" and ch not in '"\\%' else "_" for ch in name
    )
    encoded = urllib.parse.quote(name, safe="", errors="replace")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def fit_benchmark(
    doc: Dict[str, object], totals: Optional[Tuple[object, object]]
) -> Dict[str, object]:
    """The results as a page with this trace may show them.

    ``totals`` is the reviewed trace's (input bytes, stream bytes), or None for a
    page without a trace. Only the recorded trace is "the traced run": results
    measured on another input, or shown without a trace, must not mark a point.
    """
    link = dict(doc["trace_link"])
    notes = list(doc["notes"])
    recorded = (doc["input"]["bytes"], link["stream_bytes"])
    if totals is None:
        link["state"] = "none"
    elif (
        link["state"] == "mismatch"
        and link["point"]
        and link["stream_bytes"] is not None
        and tuple(totals) == recorded
    ):
        # Measured next to another trace, shown with the recorded one (the frame
        # already matched the point when the results were measured).
        link["state"] = "linked"
    elif link["state"] in ("linked", "mismatch") and tuple(totals) != recorded:
        shown_before = (link["given_input_bytes"], link["given_stream_bytes"])
        link["state"] = "mismatch"
        # The sentences compare the reviewed trace with the recorded one.
        link["given_input_bytes"], link["given_stream_bytes"] = totals
        if tuple(totals) != shown_before:
            notes.append(
                f"This review's trace ({_count(totals[0])} B in, {_count(totals[1])} B "
                f"in streams) is not the trace recorded with these results "
                f"({_count(recorded[0])} B in, {_count(recorded[1])} B in streams), "
                "so no point is marked as the traced run."
            )
    return dict(doc, trace_link=link, notes=notes)


def _count(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) else "?"


class _Review:
    """One kept review: the page around its results, and the results."""

    def __init__(
        self,
        entry: Dict[str, object],
        parts: Tuple[bytes, ...],
        totals: Optional[Tuple[object, object]],
    ) -> None:
        self.entry = entry
        self.parts = parts
        """The page around its results: (head, served entry, rest of head, tail)."""
        self.totals = totals
        self.bench: Optional[Dict[str, object]] = None
        """The results as they were sent; the page shows them fitted to the trace."""
        self.payload = b"null"

    def set_bench(self, doc: Optional[Dict[str, object]]) -> None:
        self.bench = doc
        shown = fit_benchmark(doc, self.totals) if doc is not None else None
        self.payload = html_report.bench_payload(shown).encode("utf-8")
        self.entry = dict(self.entry, bench=pareto.summary(doc) if doc else None)

    def page_parts(self, served: bool) -> Tuple[bytes, ...]:
        """The served page, or without its served entry the page --html writes."""
        head, entry, rest, tail = self.parts
        if served:
            return head, entry, rest, self.payload, tail
        return head, rest, self.payload, tail


def _render(data: Dict[str, object], review_id: int) -> Tuple[bytes, ...]:
    # One copy serves both pages: the served entry is a part of its own.
    parts = html_report.served_parts(data, {"id": review_id, "root": "../"})
    return tuple(part.encode("utf-8") for part in parts)


def _bench_only_entry(review_id: int, doc: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": review_id,
        "name": doc["input"]["name"],
        "format": None,
        "input": doc["input"]["bytes"],
        "bytes": None,
        "failed": False,
    }


class ReviewStore:
    """The reviews a server keeps, oldest first. Past ``limit`` the oldest go."""

    def __init__(self, limit: int = MAX_REVIEWS) -> None:
        self.limit = limit
        self._reviews: "collections.OrderedDict[int, _Review]" = (
            collections.OrderedDict()
        )
        self._next_id = 1
        self._lock = threading.Lock()
        # Analyses are CPU- and memory-heavy; run one at a time.
        self._analysis_lock = threading.Lock()

    def add(
        self, data: Dict[str, object], bench: Optional[Dict[str, object]] = None
    ) -> int:
        """Keep the report of one trace (from ``html_report.report_data``), or of
        results only (``html_report.bench_only_data``), with normalized results."""
        with self._lock:
            review_id = self._next_id
            self._next_id += 1
        parts = _render(data, review_id)
        if data.get("bench_only"):
            totals = None
            entry = _bench_only_entry(review_id, bench)
        else:
            analysis = data["selections"][0]["analysis"]
            totals = (analysis["input"], analysis["bytes"])
            entry = {
                "id": review_id,
                "name": data["file"],
                "format": data["format"],
                "input": analysis["input"],
                "bytes": analysis["bytes"],
                "failed": analysis["compression_failed"],
            }
        review = _Review(entry, parts, totals)
        review.set_bench(bench)
        with self._lock:
            self._reviews[review_id] = review
            while len(self._reviews) > self.limit:
                self._reviews.popitem(last=False)
        return review_id

    def add_trace_bytes(self, name: str, raw: bytes) -> int:
        """Analyze a trace file's contents and keep its review."""
        with self._analysis_lock:
            data = html_report.report_data(name, load_trace_bytes(raw))
            return self.add(data)

    def add_bench(self, doc: Dict[str, object]) -> int:
        """Keep a page with normalized results and no trace."""
        return self.add(html_report.bench_only_data(doc), doc)

    def attach_bench(self, review_id: int, doc: Dict[str, object]) -> bool:
        """Show normalized results on a kept review; False if it is not kept."""
        with self._lock:
            review = self._reviews.get(review_id)
            bench_only = review is not None and review.totals is None
        if bench_only:
            # A page of results alone is named after their input.
            parts = _render(html_report.bench_only_data(doc), review_id)
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return False
            if bench_only:
                review.parts = parts
                review.entry = _bench_only_entry(review_id, doc)
            review.set_bench(doc)
            return True

    def bench(self, review_id: int) -> Optional[Dict[str, object]]:
        """The results of a kept review as they were sent, or None."""
        with self._lock:
            review = self._reviews.get(review_id)
            return review.bench if review else None

    def page(self, review_id: int) -> Optional[bytes]:
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return None
            parts = review.page_parts(served=True)
        return b"".join(parts)

    def download(self, review_id: int) -> Optional[Tuple[str, bytes]]:
        """A kept review as one self-contained page, the one --html writes for its
        trace and results: (file name, page), or None."""
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return None
            name = download_name(review.entry["name"], review.totals is not None)
            parts = review.page_parts(served=False)
        return name, b"".join(parts)

    def issued(self, review_id: int) -> bool:
        """Whether this server ever handed out review_id."""
        with self._lock:
            return 0 < review_id < self._next_id

    def latest(self) -> Optional[int]:
        with self._lock:
            return next(reversed(self._reviews), None)

    def entries(self) -> List[Dict[str, object]]:
        """Kept reviews, newest first."""
        with self._lock:
            return [review.entry for review in reversed(self._reviews.values())]


def landing_page(
    root: str = "./", missing: Optional[Dict[str, object]] = None
) -> bytes:
    """The page shown before any trace is open, or for a review that isn't kept."""
    served: Dict[str, object] = {"id": None, "root": root}
    if missing:
        served["missing"] = missing
    data = {
        "tool": TOOL,
        "file": "",
        "format": None,
        "trace_version": None,
        "chunks": [],
        "selections": [],
        "note": "",
        "empty": True,
        "served": served,
    }
    return html_report.render_html(data).encode("utf-8")


def make_server(
    store: ReviewStore,
    host: str = "127.0.0.1",
    port: int = 0,
    max_upload: int = MAX_UPLOAD,
    max_bench_upload: int = MAX_BENCH_UPLOAD,
) -> http.server.ThreadingHTTPServer:
    """An HTTP server for the reviews in ``store``. Port 0 picks a free port."""
    allowed_hosts = _LOOPBACK_NAMES | {host.lower()} if _is_loopback(host) else None

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "codec_reviewer"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            headers: Tuple[Tuple[str, str], ...] = (),
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if content_type.startswith("text/html"):
                self.send_header("Content-Security-Policy", _PAGE_POLICY)
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            body = json.dumps(value).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _text(self, status: int, message: str) -> None:
            self._send(
                status, (message + "\n").encode("utf-8"), "text/plain; charset=utf-8"
            )

        def _host_allowed(self) -> bool:
            if allowed_hosts is None:
                return True
            try:
                name = urllib.parse.urlsplit(
                    "//" + self.headers.get("Host", "")
                ).hostname
            except ValueError:
                return False
            return name in allowed_hosts

        def _upload_refusal(self, kind: str) -> Optional[str]:
            """Why an upload is refused, or None if it came from the page."""
            what = "Traces" if kind == "upload" else "Results"
            if self.headers.get(UPLOAD_HEADER) != kind:
                return f"{what} can only be opened from the review page."
            # Browsers set Sec-Fetch-Site and pages cannot change it. A client that
            # is not a browser (codec_reviewer.py, curl) could send anything anyway.
            site = self.headers.get("Sec-Fetch-Site")
            if site in ("same-origin", "none"):
                return None
            if site is not None:
                return f"{what} can only be opened from the review page."
            origin = self.headers.get("Origin")
            if origin is None:
                return None
            parts = urllib.parse.urlsplit(origin)
            host = self.headers.get("Host", "")
            if (
                parts.scheme in ("http", "https")
                and parts.netloc.lower() == host.lower()
            ):
                return None
            return (
                f"The page at {origin} does not match this server's address {host}; "
                "a proxy may be rewriting the Host header."
            )

        def _read_body(self, limit: int, what: str) -> Optional[bytes]:
            """The request body, or None after answering with an error."""
            length = self.headers.get("Content-Length", "")
            if not re.fullmatch(r"[0-9]{1,19}", length):
                self.close_connection = True
                self._json(411, {"error": "The upload needs a Content-Length."})
                return None
            size = int(length)
            if size > limit:
                self.close_connection = True
                self._json(
                    413,
                    {
                        "error": f"The {what} is {size:,} bytes; this server accepts "
                        f"{what}s up to {limit:,} bytes."
                    },
                )
                return None
            raw = self.rfile.read(size)
            if len(raw) != size:
                self.close_connection = True
                name = trace_name(self.headers.get(NAME_HEADER, ""))
                self._json(400, {"error": f"The upload of {name} was cut short."})
                return None
            return raw

        def do_GET(self) -> None:
            if not self._host_allowed():
                self._text(403, "This server only answers requests to localhost.")
                return
            path = urllib.parse.urlsplit(self.path).path
            if path == "/":
                latest = store.latest()
                if latest is None:
                    self._send(200, landing_page(), "text/html; charset=utf-8")
                else:
                    # Relative, so the page also works behind a path prefix.
                    self._send(
                        303,
                        b"",
                        "text/plain; charset=utf-8",
                        (("Location", f"r/{latest}"),),
                    )
                return
            match = re.fullmatch(r"/r/(\d{1,9})", path)
            if match:
                review_id = int(match.group(1))
                page = store.page(review_id)
                if page is None:
                    # The landing page, so Open trace and Reviews stay at hand.
                    reason = "evicted" if store.issued(review_id) else "unknown"
                    missing = {"id": review_id, "reason": reason, "limit": store.limit}
                    body = landing_page("../", missing)
                    self._send(404, body, "text/html; charset=utf-8")
                else:
                    self._send(200, page, "text/html; charset=utf-8")
                return
            match = re.fullmatch(r"/r/(\d{1,9})/download", path)
            if match:
                self._download(int(match.group(1)))
                return
            if path == "/api/reviews":
                self._json(
                    200,
                    {
                        "tool": TOOL,
                        "features": FEATURES,
                        "reviews": store.entries(),
                        "max_upload": max_upload,
                        "max_bench_upload": max_bench_upload,
                    },
                )
                return
            match = re.fullmatch(r"/api/reviews/(\d{1,9})/benchmark", path)
            if match:
                doc = store.bench(int(match.group(1)))
                if doc is None:
                    self._json(404, {"error": "This review has no results."})
                    return
                name = f"{doc['input']['name']}.bench.json"
                disposition = "attachment; filename*=UTF-8''" + urllib.parse.quote(
                    name, safe=""
                )
                self._send(
                    200,
                    pareto.dumps(doc).encode("utf-8"),
                    "application/json; charset=utf-8",
                    (("Content-Disposition", disposition),),
                )
                return
            self._text(404, "Not found. Reviews are served at /.")

        do_HEAD = do_GET

        def _download(self, review_id: int) -> None:
            found = store.download(review_id)
            if found is None:
                if store.issued(review_id):
                    message = (
                        f"Review {review_id} is no longer kept; this server keeps "
                        f"the newest {store.limit} reviews."
                    )
                else:
                    message = f"There is no review {review_id}."
                self._text(404, message)
                return
            name, page = found
            self._send(
                200,
                page,
                "text/html; charset=utf-8",
                (("Content-Disposition", attachment(name)),),
            )

        def do_POST(self) -> None:
            if not self._host_allowed():
                self._json(
                    403, {"error": "This server only answers requests to localhost."}
                )
                return
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/reviews":
                kind = self.headers.get(UPLOAD_HEADER)
                if kind == "benchmark":
                    self._post_bench(None)
                else:
                    self._post_trace()
                return
            match = re.fullmatch(r"/api/reviews/(\d{1,9})/benchmark", path)
            if match:
                self._post_bench(int(match.group(1)))
                return
            self._json(404, {"error": "Not found."})

        def _post_trace(self) -> None:
            refusal = self._upload_refusal("upload")
            if refusal:
                self._json(403, {"error": refusal})
                return
            raw = self._read_body(max_upload, "trace")
            if raw is None:
                return
            name = trace_name(self.headers.get(NAME_HEADER, ""))
            try:
                review_id = store.add_trace_bytes(name, raw)
            except TraceFormatError as e:
                self._json(400, {"error": f"{name} is not a readable trace: {e}"})
                return
            except RecursionError:
                self._json(
                    400,
                    {"error": f"A pipeline in {name} is nested too deeply to review."},
                )
                return
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self._json(
                    500, {"error": f"Reviewing {name} failed: {type(e).__name__}: {e}"}
                )
                return
            self._json(201, {"id": review_id, "url": f"r/{review_id}"})

        def _post_bench(self, review_id: Optional[int]) -> None:
            refusal = self._upload_refusal("benchmark")
            if refusal:
                self._json(403, {"error": refusal})
                return
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";")[0].strip().lower() != "application/json":
                self.close_connection = True
                self._json(415, {"error": "Results must be sent as application/json."})
                return
            if review_id is not None and store.page(review_id) is None:
                self.close_connection = True
                self._json(
                    404,
                    {
                        "error": f"Review {review_id} is no longer kept; open its "
                        "trace again, then add the results."
                    },
                )
                return
            raw = self._read_body(max_bench_upload, "results file")
            if raw is None:
                return
            name = trace_name(self.headers.get(NAME_HEADER, ""))
            try:
                doc = pareto.loads(raw)
            except pareto.BenchmarkFormatError as e:
                self._json(400, {"error": f"{name} is not a results file: {e}"})
                return
            try:
                if review_id is None:
                    review_id = store.add_bench(doc)
                    status = 201
                elif store.attach_bench(review_id, doc):
                    status = 200
                else:
                    self._json(404, {"error": f"Review {review_id} is no longer kept."})
                    return
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self._json(
                    500, {"error": f"Showing {name} failed: {type(e).__name__}: {e}"}
                )
                return
            self._json(status, {"id": review_id, "url": f"r/{review_id}"})

    class Server(http.server.ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        daemon_threads = True

        def handle_error(self, request, client_address) -> None:
            # A browser that goes away mid-response is not worth a traceback.
            if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
                super().handle_error(request, client_address)

    return Server((host, port), Handler)


def url_for(host: str, port: int) -> str:
    shown = "127.0.0.1" if _is_wildcard(host) else host
    return f"http://[{shown}]:{port}/" if ":" in shown else f"http://{shown}:{port}/"


def _connect(host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(
        "127.0.0.1" if _is_wildcard(host) else host, port, timeout=timeout
    )


def running_reviewer(
    host: str, port: int, timeout: float = 3.0
) -> Optional[Dict[str, object]]:
    """The review listing of a codec reviewer at host:port, or None if there is none."""
    conn = _connect(host, port, timeout)
    try:
        conn.request("GET", "/api/reviews")
        response = conn.getresponse()
        listing = json.loads(response.read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()
    if (
        response.status == 200
        and isinstance(listing, dict)
        and listing.get("tool") == TOOL
    ):
        return listing
    return None


def _post(
    host: str,
    port: int,
    path: str,
    body: bytes,
    headers: Dict[str, str],
    timeout: float,
) -> Tuple[Optional[int], Dict[str, object], str]:
    """POST to a running reviewer: (status or None, JSON answer, error text)."""
    conn = _connect(host, port, timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
    except (BrokenPipeError, ConnectionResetError) as e:
        # The server may have answered (say, 413) and closed before reading it all.
        sent_error: Optional[Exception] = e
    except (OSError, http.client.HTTPException) as e:
        conn.close()
        return None, {}, f"cannot reach {url_for(host, port)}: {e}"
    else:
        sent_error = None
    try:
        response = conn.getresponse()
        try:
            answer = json.loads(response.read())
        except ValueError:
            answer = None
        if not isinstance(answer, dict):
            answer = {}
        error = answer.get("error")
        if not isinstance(error, str):
            error = f"the server answered {response.status}"
        return response.status, answer, error
    except (OSError, http.client.HTTPException) as e:
        return None, {}, f"cannot reach {url_for(host, port)}: {sent_error or e}"
    finally:
        conn.close()


def _added(
    host: str, port: int, result: Tuple[Optional[int], Dict[str, object], str], ok: int
) -> Tuple[bool, str, Optional[int]]:
    status, answer, error = result
    review_id, url = answer.get("id"), answer.get("url")
    if status == ok and isinstance(url, str) and isinstance(review_id, int):
        return True, url_for(host, port) + url, review_id
    return False, error, None


def add_trace(
    host: str, port: int, name: str, raw: bytes, timeout: float = 600.0
) -> Tuple[bool, str, Optional[int]]:
    """Add a trace to a running reviewer: (True, review URL, id) or (False, error, None)."""
    headers = {
        "Content-Type": "application/octet-stream",
        UPLOAD_HEADER: "upload",
        NAME_HEADER: urllib.parse.quote(name, errors="replace"),
    }
    result = _post(host, port, "/api/reviews", raw, headers, timeout)
    return _added(host, port, result, 201)


def send_trace(
    host: str, port: int, name: str, raw: bytes, timeout: float = 600.0
) -> Tuple[bool, str]:
    """Add a trace to a running reviewer: (True, review URL) or (False, error)."""
    ok, message, _ = add_trace(host, port, name, raw, timeout)
    return ok, message


def send_benchmark(
    host: str,
    port: int,
    doc: Dict[str, object],
    review_id: Optional[int] = None,
    timeout: float = 60.0,
) -> Tuple[bool, str]:
    """Show results on review ``review_id`` of a running reviewer, or on a page of
    their own: (True, review URL) or (False, error)."""
    headers = {
        "Content-Type": "application/json",
        UPLOAD_HEADER: "benchmark",
        NAME_HEADER: urllib.parse.quote(
            f"{doc['input']['name']}.bench.json", errors="replace"
        ),
    }
    body = pareto.dumps(doc).encode("utf-8")
    if review_id is None:
        result = _post(host, port, "/api/reviews", body, headers, timeout)
        ok, message, _ = _added(host, port, result, 201)
    else:
        path = f"/api/reviews/{review_id}/benchmark"
        result = _post(host, port, path, body, headers, timeout)
        ok, message, _ = _added(host, port, result, 200)
    return ok, message
