# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Serve interactive reviews over HTTP, and review traces opened in the page.

Each review lives at ``/r/<id>`` and ``/`` goes to the newest one. The page
posts a trace file to ``/api/reviews``; the server analyzes it with the same
code as the command line and answers with the new review's address.

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
from trace_format import load_trace_bytes, TraceFormatError

TOOL = "tools/codec_reviewer"
MAX_UPLOAD = 512 * 1024 * 1024
MAX_REVIEWS = 50
UPLOAD_HEADER = "X-Codec-Reviewer"
NAME_HEADER = "X-Trace-Name"
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


class ReviewStore:
    """The reviews a server keeps, oldest first. Past ``limit`` the oldest go."""

    def __init__(self, limit: int = MAX_REVIEWS) -> None:
        self.limit = limit
        self._reviews: "collections.OrderedDict[int, Tuple[Dict[str, object], bytes]]" = collections.OrderedDict()
        self._next_id = 1
        self._lock = threading.Lock()
        # Analyses are CPU- and memory-heavy; run one at a time.
        self._analysis_lock = threading.Lock()

    def add(self, data: Dict[str, object]) -> int:
        """Keep the report of one trace (from ``html_report.report_data``)."""
        with self._lock:
            review_id = self._next_id
            self._next_id += 1
        page = html_report.render_html(
            dict(data, served={"id": review_id, "root": "../"})
        )
        analysis = data["selections"][0]["analysis"]
        entry = {
            "id": review_id,
            "name": data["file"],
            "format": data["format"],
            "input": analysis["input"],
            "bytes": analysis["bytes"],
            "failed": analysis["compression_failed"],
        }
        with self._lock:
            self._reviews[review_id] = (entry, page.encode("utf-8"))
            while len(self._reviews) > self.limit:
                self._reviews.popitem(last=False)
        return review_id

    def add_trace_bytes(self, name: str, raw: bytes) -> int:
        """Analyze a trace file's contents and keep its review."""
        with self._analysis_lock:
            data = html_report.report_data(name, load_trace_bytes(raw))
            return self.add(data)

    def page(self, review_id: int) -> Optional[bytes]:
        with self._lock:
            item = self._reviews.get(review_id)
        return item[1] if item else None

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
            return [entry for entry, _ in reversed(self._reviews.values())]


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

        def _upload_refusal(self) -> Optional[str]:
            """Why an upload is refused, or None if it came from the page."""
            if self.headers.get(UPLOAD_HEADER) != "upload":
                return "Traces can only be opened from the review page."
            # Browsers set Sec-Fetch-Site and pages cannot change it. A client that
            # is not a browser (codec_reviewer.py, curl) could send anything anyway.
            site = self.headers.get("Sec-Fetch-Site")
            if site in ("same-origin", "none"):
                return None
            if site is not None:
                return "Traces can only be opened from the review page."
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
            if path == "/api/reviews":
                self._json(
                    200,
                    {
                        "tool": TOOL,
                        "reviews": store.entries(),
                        "max_upload": max_upload,
                    },
                )
                return
            self._text(404, "Not found. Reviews are served at /.")

        do_HEAD = do_GET

        def do_POST(self) -> None:
            if not self._host_allowed():
                self._json(
                    403, {"error": "This server only answers requests to localhost."}
                )
                return
            if urllib.parse.urlsplit(self.path).path != "/api/reviews":
                self._json(404, {"error": "Not found."})
                return
            refusal = self._upload_refusal()
            if refusal:
                self._json(403, {"error": refusal})
                return
            length = self.headers.get("Content-Length", "")
            if not re.fullmatch(r"[0-9]{1,19}", length):
                self.close_connection = True
                self._json(411, {"error": "The upload needs a Content-Length."})
                return
            size = int(length)
            if size > max_upload:
                self.close_connection = True
                self._json(
                    413,
                    {
                        "error": f"The trace is {size:,} bytes; this server accepts "
                        f"traces up to {max_upload:,} bytes."
                    },
                )
                return
            raw = self.rfile.read(size)
            name = trace_name(self.headers.get(NAME_HEADER, ""))
            if len(raw) != size:
                self.close_connection = True
                self._json(400, {"error": f"The upload of {name} was cut short."})
                return
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


def send_trace(
    host: str, port: int, name: str, raw: bytes, timeout: float = 600.0
) -> Tuple[bool, str]:
    """Add a trace to a running reviewer: (True, review URL) or (False, error)."""
    conn = _connect(host, port, timeout)
    try:
        conn.request(
            "POST",
            "/api/reviews",
            body=raw,
            headers={
                "Content-Type": "application/octet-stream",
                UPLOAD_HEADER: "upload",
                NAME_HEADER: urllib.parse.quote(name),
            },
        )
    except (BrokenPipeError, ConnectionResetError) as e:
        # The server may have answered (say, 413) and closed before reading it all.
        sent_error = e
    except (OSError, http.client.HTTPException) as e:
        conn.close()
        return False, f"cannot reach {url_for(host, port)}: {e}"
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
        if response.status == 201 and isinstance(answer.get("url"), str):
            return True, url_for(host, port) + answer["url"]
        error = answer.get("error")
        return False, error if isinstance(error, str) else (
            f"the server answered {response.status}"
        )
    except (OSError, http.client.HTTPException) as e:
        return False, f"cannot reach {url_for(host, port)}: {sent_error or e}"
    finally:
        conn.close()
