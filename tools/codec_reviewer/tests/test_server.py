# Copyright (c) Meta Platforms, Inc. and affiliates.

import http.client
import http.server
import io
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path[:0] = [TOOL, HERE]

import codec_reviewer  # noqa: E402
import review_server  # noqa: E402
import trace_builder as tb  # noqa: E402

DATA = os.path.join(HERE, "data")
SENSORS = os.path.join(DATA, "sensors_chunks.cbor.gz")
SERIAL_DOT = os.path.join(DATA, "serial.dot")


def read(path):
    with open(path, "rb") as f:
        return f.read()


class Server:
    """A review server on a free port, running in a thread."""

    def __init__(self, **kwargs):
        self.store = kwargs.pop("store", None) or review_server.ReviewStore()
        self.httpd = review_server.make_server(self.store, "127.0.0.1", 0, **kwargs)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def upload(self, name, raw, **headers):
        sent = {review_server.UPLOAD_HEADER: "upload", "X-Trace-Name": name}
        sent.update(headers)
        status, _, body = self.request("POST", "/api/reviews", raw, sent)
        return status, json.loads(body)


class ServerTest(unittest.TestCase):
    def test_open_traces_in_the_page(self):
        with Server() as server:
            status, headers, body = server.request("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn(b"<title>Codec reviewer</title>", body)
            self.assertIn(b'"empty":true', body)

            status, answer = server.upload("sensors_chunks.cbor.gz", read(SENSORS))
            self.assertEqual((status, answer), (201, {"id": 1, "url": "r/1"}))
            status, headers, _ = server.request("GET", "/")
            self.assertEqual((status, headers["Location"]), (303, "r/1"))
            status, headers, body = server.request("GET", "/r/1")
            self.assertEqual(status, 200)
            self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
            self.assertIn(b"<title>Codec review: sensors_chunks.cbor.gz</title>", body)
            self.assertIn(b'"served":{"id":1,"root":"../"}', body)

            # DOT text works too, and names are reduced to a file name.
            status, answer = server.upload("..%2Fdir%2Fserial.dot", read(SERIAL_DOT))
            self.assertEqual((status, answer["id"]), (201, 2))
            _, _, body = server.request("GET", "/api/reviews")
            listing = json.loads(body)
            self.assertEqual(listing["tool"], review_server.TOOL)
            self.assertEqual(
                [(r["id"], r["name"]) for r in listing["reviews"]],
                [(2, "serial.dot"), (1, "sensors_chunks.cbor.gz")],
            )
            self.assertEqual(listing["reviews"][1]["bytes"], 560_780)

            status, answer = server.upload("notes.txt", b"not a trace")
            self.assertEqual(status, 400)
            self.assertIn("notes.txt is not a readable trace", answer["error"])
            self.assertEqual(server.request("GET", "/r/3")[0], 404)
            self.assertEqual(server.request("GET", "/r/1/")[0], 404)
            self.assertEqual(server.request("GET", "/elsewhere")[0], 404)

    def test_uploads_only_come_from_the_page(self):
        raw = read(SERIAL_DOT)
        with Server() as server:
            own = f"http://127.0.0.1:{server.port}"
            status, _, _ = server.request(
                "POST", "/api/reviews", raw, {"X-Trace-Name": "serial.dot"}
            )
            self.assertEqual(status, 403)
            self.assertEqual(
                server.upload("t", raw, Origin="http://evil.example")[0], 403
            )
            self.assertEqual(server.upload("t", raw, Origin="null")[0], 403)
            self.assertEqual(
                server.upload("t", raw, **{"Sec-Fetch-Site": "cross-site"})[0], 403
            )
            self.assertEqual(
                server.upload(
                    "t", raw, Origin=own, **{"Sec-Fetch-Site": "same-origin"}
                )[0],
                201,
            )
            # DNS rebinding: a request addressed to another name is refused...
            evil = {"Host": f"evil.example:{server.port}"}
            self.assertEqual(
                server.request("GET", "/api/reviews", headers=evil)[0], 403
            )
            self.assertEqual(server.upload("t", raw, **evil)[0], 403)
            # ...while a forwarded port (another port number on localhost) works.
            forwarded = {"Host": "localhost:18765"}
            self.assertEqual(
                server.request("GET", "/api/reviews", headers=forwarded)[0], 200
            )

    def test_limits(self):
        store = review_server.ReviewStore(limit=2)
        raw = read(SERIAL_DOT)
        with Server(store=store, max_upload=len(raw)) as server:
            status, answer = server.upload("big.dot", raw + b" ")
            self.assertEqual(status, 413)
            self.assertIn("up to", answer["error"])
            for _ in range(3):
                self.assertEqual(server.upload("serial.dot", raw)[0], 201)
            self.assertEqual([r["id"] for r in store.entries()], [3, 2])
            self.assertEqual(server.request("GET", "/r/1")[0], 404)

            # An upload without Content-Length is refused.
            with socket.create_connection(("127.0.0.1", server.port)) as s:
                s.sendall(
                    b"POST /api/reviews HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    b"X-Codec-Reviewer: upload\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
                )
                reply = b""
                try:
                    while chunk := s.recv(4096):
                        reply += chunk
                except ConnectionResetError:
                    pass
                self.assertTrue(reply.startswith(b"HTTP/1.0 411"), reply[:40])

    def test_errors_keep_the_server_running(self):
        packed = read(SENSORS)
        damaged = packed[:20] + bytes(b ^ 0xFF for b in packed[20:60]) + packed[60:]
        chain = tb.TraceBuilder()
        ch = chain.chunk()
        s = ch.start(1000)
        for _ in range(25_000):
            s = ch.one("zl.delta_int", s, 1000, tb.NUMERIC, 4)
        ch.store_rest()
        with Server() as server:
            for name, raw, message in (
                ("damaged.cbor.gz", damaged, "gzip data is damaged"),
                ("odd.cbor", tb.encode_cbor({"chunks": [{"streams": [1]}]}), "shaped"),
                ("deep.cbor", chain.to_cbor(), "nested too deeply"),
            ):
                status, answer = server.upload(name, raw)
                self.assertEqual(status, 400, answer)
                self.assertIn(message, answer["error"])
            self.assertEqual(server.upload("serial.dot", read(SERIAL_DOT))[0], 201)

    def test_http_details(self):
        with Server() as server:
            server.upload("serial.dot", read(SERIAL_DOT))

            def raw_request(head, body=b""):
                with socket.create_connection(("127.0.0.1", server.port)) as s:
                    s.sendall(head + body)
                    s.shutdown(socket.SHUT_WR)
                    reply = b""
                    try:
                        while chunk := s.recv(65536):
                            reply += chunk
                    except ConnectionResetError:
                        pass
                return reply

            # HEAD answers with the page's headers and no body.
            reply = raw_request(b"HEAD /r/1 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            head, _, body = reply.partition(b"\r\n\r\n")
            self.assertTrue(head.startswith(b"HTTP/1.0 200"), head[:60])
            self.assertRegex(head.decode(), r"Content-Length: \d{4,}")
            self.assertEqual(body, b"")

            post = b"POST /api/reviews HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Codec-Reviewer: upload\r\n"
            # A body shorter than its Content-Length.
            reply = raw_request(post + b"Content-Length: 5000\r\n\r\n", b"digraph")
            self.assertTrue(reply.startswith(b"HTTP/1.0 400"), reply[:60])
            self.assertIn(b"cut short", reply)
            # A Content-Length written with non-ASCII digits.
            reply = raw_request(post + b"Content-Length: \xb2\r\n\r\n")
            self.assertTrue(reply.startswith(b"HTTP/1.0 411"), reply[:60])

    def test_missing_reviews_get_the_landing_page(self):
        store = review_server.ReviewStore(limit=1)
        with Server(store=store) as server:
            server.upload("serial.dot", read(SERIAL_DOT))
            server.upload("serial.dot", read(SERIAL_DOT))
            for path, reason in (
                ("/r/1", '"reason":"evicted"'),
                ("/r/9", '"reason":"unknown"'),
            ):
                status, headers, body = server.request("GET", path)
                self.assertEqual(status, 404)
                self.assertTrue(headers["Content-Type"].startswith("text/html"))
                self.assertIn(reason.encode(), body)
                self.assertIn(b'"root":"../"', body)

    def test_uploads_through_a_proxy_that_rewrites_host(self):
        raw = read(SERIAL_DOT)
        with Server() as server:
            page = (
                "http://127.0.0.1:9999"  # the proxy's address, as the browser sees it
            )
            ok = server.upload(
                "t", raw, Origin=page, **{"Sec-Fetch-Site": "same-origin"}
            )
            self.assertEqual(ok[0], 201)
            status, answer = server.upload("t", raw, Origin=page)
            self.assertEqual(status, 403)
            self.assertIn("127.0.0.1:9999", answer["error"])
            self.assertIn("proxy", answer["error"])


class CommandLineTest(unittest.TestCase):
    def run_cli(self, *args):
        err = io.StringIO()
        with redirect_stderr(err):
            code = codec_reviewer.main(list(args), stdout=io.StringIO())
        return code, err.getvalue()

    def test_trace_arguments(self):
        for argv in (
            [],
            [SENSORS, SERIAL_DOT],
            ["--serve", "0", SENSORS, SERIAL_DOT, "--html", "x"],
            ["--serve", "0", "--show", "E1"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    codec_reviewer.parse_args(argv)
        # Trace files can sit between options.
        args = codec_reviewer.parse_args(
            [SENSORS, "--serve", "0", SERIAL_DOT, "--quiet", SERIAL_DOT]
        )
        self.assertEqual(args.traces, [SENSORS, SERIAL_DOT, SERIAL_DOT])
        self.assertEqual((args.serve, args.quiet), (0, True))

    def test_serve_traces_given_as_arguments(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(TOOL, "codec_reviewer.py"),
                "--quiet",
                "--serve",
                "0",
                SENSORS,
                SERIAL_DOT,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            ready, _, _ = select.select([proc.stderr], [], [], 60)
            self.assertTrue(ready, "the server did not start")
            line = proc.stderr.readline().decode()
            self.assertIn("(2 traces;", line)
            port = int(line.split("http://127.0.0.1:")[1].split("/")[0])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("GET", "/api/reviews")
            reviews = json.loads(conn.getresponse().read())["reviews"]
            conn.close()
            self.assertEqual(
                [r["name"] for r in reviews], ["serial.dot", "sensors_chunks.cbor.gz"]
            )
        finally:
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.wait(timeout=30), 0)
            proc.stderr.close()

    def test_add_traces_to_a_running_reviewer(self):
        with Server() as server:
            code, err = self.run_cli("--serve", str(server.port), SERIAL_DOT)
            self.assertEqual(code, 0, err)
            self.assertIn(f"http://127.0.0.1:{server.port}/r/1", err)
            self.assertEqual(
                [r["name"] for r in server.store.entries()], ["serial.dot"]
            )

            code, err = self.run_cli(
                "--serve", str(server.port), SERIAL_DOT + ".missing"
            )
            self.assertEqual(code, 2)
            self.assertIn("cannot read", err)

        # A port held by something else is an error, not a traceback.
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            port = str(taken.getsockname()[1])
            code, err = self.run_cli("--serve", port, SERIAL_DOT)
        self.assertEqual(code, 2)
        self.assertIn(f"cannot serve on 127.0.0.1:{port}", err)

    def test_options_apply_when_adding_to_a_running_reviewer(self):
        with Server() as server, tempfile.TemporaryDirectory() as tmp:
            html = os.path.join(tmp, "serial.html")
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stderr(err):
                code = codec_reviewer.main(
                    ["--serve", str(server.port), SERIAL_DOT, "--html", html],
                    stdout=out,
                )
            self.assertEqual(code, 0, err.getvalue())
            self.assertTrue(os.path.getsize(html) > 1000)
            self.assertIn("Codec review: serial.dot", out.getvalue())
            self.assertEqual(len(server.store.entries()), 1)

            code, err = self.run_cli(
                "--serve", str(server.port), SERIAL_DOT, "--show", "Z9"
            )
            self.assertEqual(code, 2)
            self.assertIn("no pipeline Z9", err)
            self.assertIn("nothing was served or added", err)
            self.assertEqual(len(server.store.entries()), 1)

    def test_oversized_trace_for_a_running_reviewer(self):
        with Server(max_upload=100) as server:
            code, err = self.run_cli("--quiet", "--serve", str(server.port), SENSORS)
        self.assertEqual(code, 2)
        # Checked before sending, from the limit the reviewer lists.
        self.assertIn("the reviewer accepts traces up to 100 bytes", err)
        # Sent anyway, a large body that the server refuses early still gets its message.
        with Server(max_upload=100) as server:
            ok, message = review_server.send_trace(
                "127.0.0.1", server.port, "big.dot", b"x" * (32 << 20)
            )
        self.assertFalse(ok)
        self.assertIn("up to 100 bytes", message)

    def test_port_held_by_another_http_service(self):
        class Other(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"[]"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        other = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Other)
        thread = threading.Thread(target=other.serve_forever, daemon=True)
        thread.start()
        try:
            port = str(other.server_address[1])
            code, err = self.run_cli("--quiet", "--serve", port, SERIAL_DOT)
        finally:
            other.shutdown()
            other.server_close()
        self.assertEqual(code, 2)
        self.assertIn(f"cannot serve on 127.0.0.1:{port}", err)


if __name__ == "__main__":
    unittest.main()
