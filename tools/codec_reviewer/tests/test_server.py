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
import html_report  # noqa: E402
import pareto  # noqa: E402
import review_server  # noqa: E402
import trace_builder as tb  # noqa: E402

DATA = os.path.join(HERE, "data")
SENSORS = os.path.join(DATA, "sensors_chunks.cbor.gz")
SERIAL_DOT = os.path.join(DATA, "serial.dot")
# Measured on the input of SENSORS (sensors.parquet); its trace link says so.
BENCH = os.path.join(DATA, "bench_sensors.json")


def bench_doc():
    with open(BENCH, "rb") as f:
        return pareto.loads(f.read())


def page_bench(body):
    """The results embedded in a served page (None for null)."""
    text = body.decode("utf-8")
    start = text.index("const BENCH = ") + len("const BENCH = ")
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


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

    def send_results(self, raw, review_id=None, **headers):
        sent = {
            review_server.UPLOAD_HEADER: "benchmark",
            "Content-Type": "application/json",
            "X-Trace-Name": "sensors.parquet.bench.json",
        }
        sent.update(headers)
        path = (
            "/api/reviews"
            if review_id is None
            else f"/api/reviews/{review_id}/benchmark"
        )
        status, _, body = self.request("POST", path, raw, sent)
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


class ResultsTest(unittest.TestCase):
    def test_results_on_reviews(self):
        raw = pareto.dumps(bench_doc()).encode()
        with Server() as server:
            server.upload("sensors_chunks.cbor.gz", read(SENSORS))
            server.upload("serial.dot", read(SERIAL_DOT))
            _, _, before = server.request("GET", "/r/1")
            self.assertIsNone(page_bench(before))

            self.assertEqual(
                server.send_results(raw, 1), (200, {"id": 1, "url": "r/1"})
            )
            _, _, page = server.request("GET", "/r/1")
            shown = page_bench(page)
            # The results measured on this trace's input mark its traced run.
            self.assertEqual(shown["trace_link"]["state"], "linked")
            self.assertEqual(shown["frontiers"], bench_doc()["frontiers"])
            # Only the results changed in the page.
            payload = html_report.bench_payload(shown).encode()
            self.assertEqual(page.replace(payload, b"null", 1), before)

            # On another trace they are shown, but mark nothing.
            self.assertEqual(server.send_results(raw, 2)[0], 200)
            shown = page_bench(server.request("GET", "/r/2")[2])
            self.assertEqual(shown["trace_link"]["state"], "mismatch")
            self.assertIn(
                "not the trace recorded with these results", shown["notes"][-1]
            )

            # Results alone get a page of their own.
            self.assertEqual(server.send_results(raw), (201, {"id": 3, "url": "r/3"}))
            _, _, page = server.request("GET", "/r/3")
            self.assertIn(b'"bench_only":true', page)
            self.assertIn(b"<title>Ratio vs speed: sensors.parquet</title>", page)
            self.assertEqual(page_bench(page)["trace_link"]["state"], "none")

            _, _, body = server.request("GET", "/api/reviews")
            listing = json.loads(body)
            self.assertIn("benchmark", listing["features"])
            self.assertEqual(
                listing["max_bench_upload"], review_server.MAX_BENCH_UPLOAD
            )
            self.assertEqual(
                [(r["id"], r["name"], bool(r["bench"])) for r in listing["reviews"]],
                [
                    (3, "sensors.parquet", True),
                    (2, "serial.dot", True),
                    (1, "sensors_chunks.cbor.gz", True),
                ],
            )
            self.assertEqual(
                listing["reviews"][0]["bench"], pareto.summary(bench_doc())
            )

            status, headers, body = server.request("GET", "/api/reviews/1/benchmark")
            self.assertEqual(status, 200)
            self.assertIn("sensors.parquet.bench.json", headers["Content-Disposition"])
            self.assertTrue(headers["Content-Disposition"].startswith("attachment;"))
            self.assertEqual(json.loads(body)["trace_link"]["state"], "linked")
            # The download opens again as results.
            self.assertEqual(pareto.loads(body)["points"], bench_doc()["points"])

    def test_results_keep_their_link_when_moved(self):
        raw = pareto.dumps(bench_doc()).encode()
        with Server() as server:
            server.upload("serial.dot", read(SERIAL_DOT))
            self.assertEqual(server.send_results(raw, 1)[0], 200)
            self.assertEqual(server.send_results(raw)[0], 201)
            server.upload("sensors_chunks.cbor.gz", read(SENSORS))

            # On another trace the sentence compares that trace with the recorded one.
            link = page_bench(server.request("GET", "/r/1")[2])["trace_link"]
            self.assertEqual(link["state"], "mismatch")
            totals = server.store.entries()[2]
            self.assertEqual(
                (link["given_input_bytes"], link["given_stream_bytes"]),
                (totals["input"], totals["bytes"]),
            )
            # Downloads return the results as sent, so they mark the traced run
            # again when opened on their own trace.
            for review_id in (1, 2):
                _, _, body = server.request(
                    "GET", f"/api/reviews/{review_id}/benchmark"
                )
                self.assertEqual(json.loads(body)["trace_link"]["state"], "linked")
                self.assertEqual(server.send_results(body, 3)[0], 200)
                shown = page_bench(server.request("GET", "/r/3")[2])
                self.assertEqual(shown["trace_link"]["state"], "linked")
                self.assertEqual(shown["notes"], bench_doc()["notes"])

    def test_fit_results_measured_next_to_another_trace(self):
        doc = bench_doc()
        recorded = (doc["input"]["bytes"], doc["trace_link"]["stream_bytes"])
        given = (2_889_011, 681_395)
        measured = dict(
            doc,
            trace_link=dict(
                doc["trace_link"],
                state="mismatch",
                given_input_bytes=given[0],
                given_stream_bytes=given[1],
            ),
        )
        # Shown with the given trace: unchanged.
        shown = review_server.fit_benchmark(measured, given)
        self.assertEqual(shown["trace_link"], measured["trace_link"])
        self.assertEqual(shown["notes"], doc["notes"])
        # Shown with the recorded trace: that is the traced run.
        shown = review_server.fit_benchmark(measured, recorded)
        self.assertEqual(shown["trace_link"]["state"], "linked")
        # Shown with a third trace: the sentence names that trace.
        third = (2_889_011, 12_345)
        shown = review_server.fit_benchmark(measured, third)
        link = shown["trace_link"]
        self.assertEqual(link["state"], "mismatch")
        self.assertEqual((link["given_input_bytes"], link["given_stream_bytes"]), third)
        self.assertIn("12,345 B in streams", shown["notes"][-1])
        # Without recorded stream bytes nothing becomes the traced run.
        blind = dict(
            measured, trace_link=dict(measured["trace_link"], stream_bytes=None)
        )
        shown = review_server.fit_benchmark(blind, (recorded[0], None))
        self.assertEqual(shown["trace_link"]["state"], "mismatch")

    def test_new_results_rename_a_results_page(self):
        doc = bench_doc()
        other = json.loads(pareto.dumps(doc))
        other["input"]["name"] = "other.parquet"
        with Server() as server:
            server.send_results(pareto.dumps(doc).encode())
            self.assertEqual(server.send_results(json.dumps(other).encode(), 1)[0], 200)
            _, _, page = server.request("GET", "/r/1")
            self.assertIn(b"<title>Ratio vs speed: other.parquet</title>", page)
            self.assertIn(b'"file":"other.parquet"', page)
            self.assertNotIn(b'"file":"sensors.parquet"', page)
            (entry,) = server.store.entries()
            self.assertEqual(entry["name"], "other.parquet")
            self.assertEqual(entry["bench"]["input"], "other.parquet")

    def test_results_with_names_that_are_not_text(self):
        raw = pareto.dumps(bench_doc()).replace(
            '"name":"sensors.parquet"', '"name":"caf\\udce9.parquet"'
        )
        with Server() as server:
            status, answer = server.send_results(raw.encode())
            self.assertEqual(status, 201, answer)
            status, headers, body = server.request("GET", "/api/reviews/1/benchmark")
            self.assertEqual(status, 200)
            self.assertIn("caf%3F.parquet.bench.json", headers["Content-Disposition"])
            self.assertEqual(server.request("GET", "/r/1")[0], 200)

    def test_results_uploads_are_checked(self):
        doc = bench_doc()
        raw = pareto.dumps(doc).encode()
        store = review_server.ReviewStore(limit=1)
        limit = len(raw) + 64
        with Server(store=store, max_bench_upload=limit) as server:
            server.upload("serial.dot", read(SERIAL_DOT))
            self.assertEqual(server.request("GET", "/api/reviews/1/benchmark")[0], 404)
            for headers, status in (
                ({review_server.UPLOAD_HEADER: "upload"}, 403),
                ({review_server.UPLOAD_HEADER: ""}, 403),
                ({"Sec-Fetch-Site": "cross-site"}, 403),
                ({"Origin": "http://evil.example"}, 403),
                ({"Content-Type": "text/plain"}, 415),
            ):
                with self.subTest(headers=headers):
                    answer = server.send_results(raw, 1, **headers)
                    self.assertEqual(answer[0], status, answer)
            status, answer = server.send_results(raw + b" " * 65, 1)
            self.assertEqual(status, 413)
            self.assertIn("results file", answer["error"])

            tampered = json.loads(raw)
            tampered["frontiers"] = {
                "c": {"subsets": {}},
                "cd": {"subsets": {}, "beaten_by": {"x": "y"}},
            }
            tampered["points"][0]["c_speed"] = 1e9
            self.assertEqual(
                server.send_results(json.dumps(tampered).encode(), 1)[0], 200
            )
            # Frontiers sent along are ignored and computed again.
            shown = store.bench(1)
            everything = pareto.subset_key(shown, [x["id"] for x in shown["series"]])
            fastest = shown["frontiers"]["c"]["subsets"][everything]
            self.assertIn(tampered["points"][0]["id"], fastest)
            self.assertEqual(set(shown["frontiers"]), {"c", "d", "cd"})
            self.assertEqual(
                shown["frontiers"], pareto.frontiers(shown), "recomputed frontiers"
            )

            for body, message in (
                (b"{not json", "not a results file"),
                (raw.replace(b'"version":1', b'"version":2'), "not a results file"),
                (
                    raw.replace(b'"elapsed_s":', b'"elapsed_s":NaN,"x":', 1),
                    "not a results file",
                ),
                (b"\xff\xfe", "not a results file"),
            ):
                with self.subTest(body=body[:20]):
                    status, answer = server.send_results(body, 1)
                    self.assertEqual(status, 400, answer)
                    self.assertIn(message, answer["error"])

            # A review that is no longer kept cannot take results.
            server.upload("serial.dot", read(SERIAL_DOT))
            status, answer = server.send_results(raw, 1)
            self.assertEqual(status, 404)
            self.assertIn("no longer kept", answer["error"])
            self.assertEqual(server.send_results(raw, 99)[0], 404)
            self.assertEqual(server.request("GET", "/api/reviews/99/benchmark")[0], 404)

    def test_send_results_from_the_command_line_helpers(self):
        doc = bench_doc()
        with Server() as server:
            ok, url, review_id = review_server.add_trace(
                "127.0.0.1", server.port, "sensors_chunks.cbor.gz", read(SENSORS)
            )
            self.assertTrue(ok, url)
            self.assertEqual(review_id, 1)
            self.assertTrue(url.endswith("/r/1"))
            ok, url = review_server.send_benchmark("127.0.0.1", server.port, doc, 1)
            self.assertEqual((ok, url), (True, f"http://127.0.0.1:{server.port}/r/1"))
            ok, url = review_server.send_benchmark("127.0.0.1", server.port, doc)
            self.assertEqual((ok, url), (True, f"http://127.0.0.1:{server.port}/r/2"))
            ok, message = review_server.send_benchmark("127.0.0.1", server.port, doc, 7)
            self.assertFalse(ok)
            self.assertIn("no longer kept", message)


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
