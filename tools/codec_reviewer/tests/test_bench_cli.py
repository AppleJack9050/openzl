# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The command line's ratio-vs-speed options, run against fake zli and zstd."""

import http.client
import http.server
import io
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path[:0] = [TOOL, HERE]

import codec_reviewer  # noqa: E402
import fake_tools as ft  # noqa: E402
import pareto  # noqa: E402
import review_server  # noqa: E402
from test_bench import gone  # noqa: E402
from test_server import page_bench, Server  # noqa: E402

DATA = os.path.join(HERE, "data")
# A real trace of sensors.parquet (2,889,011 bytes) and one of another input.
SENSORS = os.path.join(DATA, "sensors_chunks.cbor.gz")
SERIAL_DOT = os.path.join(DATA, "serial.dot")
SENSORS_BYTES = 2_889_011


def parse_error(argv):
    err = io.StringIO()
    with redirect_stderr(err), unittest.TestCase().assertRaises(SystemExit):
        codec_reviewer.parse_args(argv)
    return err.getvalue()


class OptionsTest(unittest.TestCase):
    def test_refused_combinations(self):
        for argv, message in (
            (["t.cbor", "-p", "parquet"], "-p/--profile only apply with --bench INPUT"),
            (
                ["t.cbor", "--zstd-levels", "3", "--quick"],
                "--zstd-levels, --quick only apply with --bench",
            ),
            (["t.cbor", "--bench-json", "o.json"], "--bench-json needs --bench"),
            (["--bench", "in"], "at least one zli profile: -p PROFILE"),
            (["--bench", "in", "-p", "a", "-p", "b", "-p", "c", "-p", "d"], "1 to 3"),
            (["--bench", "in", "-p", "a", "-p", "a"], "1 to 3 different profiles"),
            (
                ["--bench", "in", "-p", "a", "--bench-results", "r.json"],
                "not both",
            ),
            (["--bench", "in", "-p", "a", "x.cbor", "y.cbor"], "at most one trace"),
            (["--bench-results", "r.json", "--show", "E1"], "--show needs a trace"),
            (["--bench", "in", "-p", "a", "--zli-levels", "fast=3"], "no fast levels"),
            (["--bench", "in", "-p", "a", "--zstd-levels", "0"], "name the level"),
            (["--bench", "in", "-p", "a", "--zstd-levels", "23"], "go up to 22"),
            (["--bench", "in", "-p", "a", "--core", "fast"], "'auto', 'none'"),
            (["--bench", "in", "-p", "a", "--rounds", "0"], "from 1 to 20"),
            (["--bench", "in", "-p", "a", "--min-time", "nan"], "seconds from 0"),
            (["--bench", "in", "-p", "a", "--chunk-size-mb", "0"], "from 1 to"),
        ):
            with self.subTest(argv=argv):
                self.assertIn(message, parse_error(argv))

    def test_defaults(self):
        args = codec_reviewer.parse_args(["--bench", "in", "-p", "parquet"])
        self.assertEqual(args.traces, [])
        self.assertEqual(args.zstd_levels, "3,6,9,12")
        self.assertEqual(args.zli_levels, "1-9,12,15,19,22")
        self.assertEqual((args.rounds, args.min_time, args.core), (3, 0.5, "auto"))
        self.assertIs(args.zstd_long, False)
        self.assertIs(args.quick, False)
        # A trace, options and profiles in any order.
        args = codec_reviewer.parse_args(
            ["-p", "parquet", "t.cbor", "--bench", "in", "-p", "serial", "--core", "3"]
        )
        self.assertEqual(
            (args.traces, args.profiles, args.core),
            (["t.cbor"], ["parquet", "serial"], 3),
        )
        # Plain reviews are unchanged.
        args = codec_reviewer.parse_args(["t.cbor"])
        self.assertIsNone(args.bench)
        self.assertEqual(args.profiles, [])


class BenchCommandTest(unittest.TestCase):
    """--bench and --bench-results end to end, with fake tools."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="bench-cli-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        bin_dir = os.path.join(self.dir, "bin")
        os.mkdir(bin_dir)
        self.tools = ft.write_tools(bin_dir)
        self.log = os.path.join(self.dir, "calls.jsonl")
        # The input the SENSORS trace compressed has this size; its content
        # does not matter to the fakes.
        self.input = os.path.join(self.dir, "sensors.parquet")
        with open(self.input, "wb") as f:
            f.truncate(SENSORS_BYTES)
        env = mock.patch.dict(
            os.environ,
            ft.tool_env(
                os.path.dirname(self.tools["zli"]), log=self.log, FAKE_TRACE=SENSORS
            ),
        )
        env.start()
        self.addCleanup(env.stop)
        for name in ("ZLI", "ZSTD", "FAKE_MODE", "FAKE_ZLI_MODE", "FAKE_ZSTD_MODE"):
            os.environ.pop(name, None)

    def bench_args(self, *extra):
        return [
            "--bench",
            self.input,
            "-p",
            "parquet",
            "--zli",
            self.tools["zli"],
            "--zstd",
            self.tools["zstd"],
            "--zli-levels",
            "1,6",
            "--zstd-levels",
            "3,6",
            "--core",
            "none",
            "--quick",
            *extra,
        ]

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stderr(err):
            code = codec_reviewer.main(list(argv), stdout=out)
        return code, out.getvalue(), err.getvalue()

    def out(self, name):
        return os.path.join(self.dir, name)

    def calls(self, tool=None):
        return ft.read_log(self.log, tool)

    def test_record_measure_and_write(self):
        html, js, results = self.out("r.html"), self.out("r.json"), self.out("b.json")
        code, out, err = self.run_cli(
            *self.bench_args("--html", html, "--json", js, "--bench-json", results)
        )
        self.assertEqual(code, 0, err)
        self.assertIn(
            "Recording the trace: zli compress sensors.parquet -p parquet", err
        )
        self.assertIn("Codec review: sensors.parquet (-p parquet)", out)
        self.assertIn("RATIO VS SPEED", out)
        for path in (html, js, results):
            self.assertIn(f"Wrote {path}", err)

        with open(results, "rb") as f:
            saved = pareto.loads(f.read())
        self.assertEqual(
            [p["id"] for p in saved["points"]],
            ["zli:parquet/1", "zstd/3", "zli:parquet/6", "zstd/6"],
        )
        self.assertEqual(saved["settings"]["levels"], "zli 1,6; zstd 3,6")
        self.assertEqual(saved["trace_link"]["state"], "linked")
        self.assertEqual(saved["trace_link"]["stream_bytes"], 560_780)
        self.assertIs(saved["trace_link"]["verified"], True)
        self.assertEqual(saved["input"]["name"], "sensors.parquet")

        with open(html, encoding="utf-8") as f:
            page = f.read().encode()
        self.assertIn(
            b"<title>Codec review: sensors.parquet (-p parquet)</title>", page
        )
        self.assertEqual(page_bench(page)["trace_link"]["state"], "linked")
        with open(js, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["file"], "sensors.parquet (-p parquet)")
        self.assertEqual(data["bench"]["points"], saved["points"])

        # zstd ran only the levels asked for; zli ran compress, decompress and
        # one quick benchmark per level.
        zstd = [c for c in self.calls("zstd") if c.get("level") is not None]
        self.assertEqual(sorted(c["level"] for c in zstd), [3, 6])
        commands = [c["argv"][0] for c in self.calls("zli")]
        self.assertEqual(commands.count("compress"), 1)
        self.assertEqual(commands.count("decompress"), 1)
        self.assertEqual(commands.count("benchmark"), 2)

        # The saved results show again without measuring.
        before = len(self.calls())
        again = self.out("again.html")
        code, _, err = self.run_cli(
            SENSORS, "--quiet", "--bench-results", results, "--html", again
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.calls()), before)
        with open(again, encoding="utf-8") as f:
            self.assertEqual(
                page_bench(f.read().encode())["trace_link"]["state"], "linked"
            )
        # Alone, they get a page of their own.
        alone = self.out("alone.html")
        code, out, err = self.run_cli("--bench-results", results, "--html", alone)
        self.assertEqual(code, 0, err)
        self.assertIn("RATIO VS SPEED", out)
        with open(alone, encoding="utf-8") as f:
            page = f.read().encode()
        self.assertIn(b'"bench_only":true', page)
        self.assertEqual(page_bench(page)["trace_link"]["state"], "none")

    def test_given_traces(self):
        code, out, err = self.run_cli(
            SENSORS, *self.bench_args("--json", self.out("s.json"))
        )
        self.assertEqual(code, 0, err)
        self.assertIn("Codec review: sensors_chunks.cbor.gz", out)
        with open(self.out("s.json")) as f:
            self.assertEqual(json.load(f)["bench"]["trace_link"]["state"], "linked")
        # The trace was still recorded, to compare it with the given one.
        self.assertEqual([c["argv"][0] for c in self.calls("zli")].count("compress"), 1)

        code, out, err = self.run_cli(
            SERIAL_DOT, *self.bench_args("--json", self.out("d.json"))
        )
        self.assertEqual(code, 0, err)
        with open(self.out("d.json")) as f:
            link = json.load(f)["bench"]["trace_link"]
        self.assertEqual(link["state"], "mismatch")
        self.assertIn("so no point is marked as the traced run", out)

    def test_problems_found_before_measuring(self):
        missing = os.path.join(self.dir, "no-such-dir", "r.html")
        code, _, err = self.run_cli(*self.bench_args("--html", missing))
        self.assertEqual(code, 2)
        self.assertIn(f"cannot write {missing}: there is no directory", err)
        code, _, err = self.run_cli(*self.bench_args("--bench-json", self.dir))
        self.assertEqual(code, 2)
        self.assertIn("it is a directory", err)
        # A given trace is reviewed before anything is recorded.
        code, _, err = self.run_cli(SENSORS + ".missing", *self.bench_args())
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)
        code, _, err = self.run_cli(SENSORS, "--show", "Z9", *self.bench_args())
        self.assertEqual(code, 2)
        self.assertIn("no pipeline Z9", err)
        self.assertEqual({c["argv"][0] for c in self.calls("zli")}, {"list-profiles"})

    def test_an_output_that_fails_keeps_the_others(self):
        results, html = self.out("b.json"), self.out("r.html")
        with mock.patch.object(
            codec_reviewer.html_report,
            "write_html",
            side_effect=OSError(28, "No space left on device"),
        ):
            code, _, err = self.run_cli(
                *self.bench_args("--quiet", "--html", html, "--bench-json", results)
            )
        self.assertEqual(code, 2)
        self.assertIn(f"cannot write {html}: No space left on device", err)
        self.assertIn(f"Wrote {results}", err)
        self.assertTrue(os.path.getsize(results) > 1000)

    def test_given_trace_without_a_recorded_one(self):
        # zli wrote a frame but no trace: the given trace cannot be matched.
        os.environ["FAKE_ZLI_MODE"] = "notrace"
        js = self.out("s.json")
        code, _, err = self.run_cli(SENSORS, *self.bench_args("--json", js))
        self.assertEqual(code, 0, err)
        with open(js) as f:
            link = json.load(f)["bench"]["trace_link"]
        self.assertEqual(link["state"], "mismatch")
        self.assertIsNone(link["stream_bytes"])

    def test_terminated_runs_clean_up(self):
        tmp = self.out("tmp")
        os.mkdir(tmp)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TMPDIR=tmp, FAKE_ZLI_MODE="hang@6")
        results = self.out("partial.json")
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(TOOL, "codec_reviewer.py"),
                "--quiet",
                *self.bench_args("--bench-json", results),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            line = ""
            while not line.startswith("[1/"):
                ready, _, _ = select.select([proc.stderr], [], [], 60)
                self.assertTrue(ready, "the benchmark did not start")
                line = proc.stderr.readline().decode()
                self.assertTrue(line, "the command ended early")
            # The next zli run hangs; stop the command like a closed terminal would.
            while not any(c["tool"] == "helper" for c in self.calls()):
                select.select([], [], [], 0.05)
            proc.send_signal(signal.SIGHUP)
            self.assertEqual(proc.wait(timeout=30), 130)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stderr.close()
        self.assertEqual(os.listdir(tmp), [])
        (hung,) = [c for c in self.calls("zli") if c.get("level") == 6]
        (helper,) = [c for c in self.calls() if c["tool"] == "helper"]
        self.assertTrue(gone(hung["pid"]))
        self.assertTrue(gone(helper["pid"]))
        with open(results, "rb") as f:
            doc = pareto.loads(f.read())
        self.assertEqual(doc["stopped"], "interrupted")

    def start_cli(self, *argv, env=None, **popen):
        return subprocess.Popen(
            [sys.executable, os.path.join(TOOL, "codec_reviewer.py"), "--quiet", *argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})),
            **popen,
        )

    def wait_for_hang(self, proc):
        """Wait until the fake zli hangs at level 6 (its helper is logged)."""
        line = ""
        while not line.startswith("[1/"):
            ready, _, _ = select.select([proc.stderr], [], [], 60)
            self.assertTrue(ready, "the benchmark did not start")
            line = proc.stderr.readline().decode()
            self.assertTrue(line, "the command ended early")
        while not any(c["tool"] == "helper" for c in self.calls()):
            select.select([], [], [], 0.05)

    def test_nohup_keeps_running(self):
        results = self.out("r.json")
        proc = self.start_cli(
            *self.bench_args("--bench-json", results, "--bench-timeout", "3"),
            env={"FAKE_ZLI_MODE": "hang@6"},
            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
        )
        try:
            self.wait_for_hang(proc)
            proc.send_signal(signal.SIGHUP)
            self.assertEqual(proc.wait(timeout=60), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stderr.close()
        with open(results, "rb") as f:
            doc = pareto.loads(f.read())
        self.assertIsNone(doc["stopped"])
        self.assertEqual(
            [p["status"] for p in doc["points"]], ["ok", "ok", "timeout", "ok"]
        )

    def test_terminated_server_does_not_start(self):
        tmp = self.out("tmp")
        os.mkdir(tmp)
        proc = self.start_cli(
            *self.bench_args("--serve", "0"),
            env={"FAKE_ZLI_MODE": "hang@6", "TMPDIR": tmp},
        )
        try:
            self.wait_for_hang(proc)
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=30), 130)
            err = proc.stderr.read().decode()
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stderr.close()
        self.assertNotIn("Codec reviewer at", err)
        (saved,) = os.listdir(tmp)
        self.assertIn(f"The results are saved in {os.path.join(tmp, saved)}", err)

    def test_closed_terminal_keeps_the_results(self):
        import pty

        results, html = self.out("r.json"), self.out("r.html")
        env = dict(os.environ, PYTHONUNBUFFERED="1", FAKE_ZLI_MODE="hang@6")
        argv = [sys.executable, os.path.join(TOOL, "codec_reviewer.py")]
        argv += self.bench_args("--bench-json", results, "--html", html)
        pid, fd = pty.fork()
        if pid == 0:
            os.execve(argv[0], argv, env)
        seen = b""
        try:
            while b"[1/" not in seen:
                ready, _, _ = select.select([fd], [], [], 60)
                self.assertTrue(ready, "the benchmark did not start")
                seen += os.read(fd, 4096)
            while not any(c["tool"] == "helper" for c in self.calls()):
                select.select([], [], [], 0.05)
        finally:
            os.close(fd)  # the terminal hangs up
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 130)
        with open(results, "rb") as f:
            self.assertEqual(pareto.loads(f.read())["stopped"], "interrupted")
        self.assertGreater(os.path.getsize(html), 1000)

    def test_outputs_that_exist(self):
        code, _, err = self.run_cli(*self.bench_args("--bench-json", os.devnull))
        self.assertEqual(code, 0, err)
        self.assertIn(f"Wrote {os.devnull}", err)
        locked = self.out("locked.html")
        with open(locked, "w"):
            pass
        os.chmod(locked, 0o444)
        if os.access(locked, os.W_OK):
            self.skipTest("running as root")
        code, _, err = self.run_cli(*self.bench_args("--html", locked))
        self.assertEqual(code, 2)
        self.assertIn(f"cannot write {locked}: it is not writable", err)

    def test_results_are_kept_when_an_output_fails(self):
        tmp = self.out("tmp")
        os.mkdir(tmp)
        html = self.out("r.html")
        with (
            mock.patch.object(tempfile, "tempdir", tmp),
            mock.patch.object(
                codec_reviewer.html_report,
                "write_html",
                side_effect=OSError(28, "full"),
            ),
        ):
            code, _, err = self.run_cli(*self.bench_args("--quiet", "--html", html))
        self.assertEqual(code, 2)
        (saved,) = os.listdir(tmp)
        self.assertIn(f"The results are saved in {os.path.join(tmp, saved)}", err)

    def test_given_trace_too_large_for_the_reviewer(self):
        with Server(max_upload=100) as small:
            code, _, err = self.run_cli(
                SENSORS, *self.bench_args("--serve", str(small.port))
            )
        self.assertEqual(code, 2)
        self.assertIn("accepts traces up to 100 bytes", err)
        self.assertEqual({c["argv"][0] for c in self.calls("zli")}, {"list-profiles"})

    def test_trace_names_that_are_not_text(self):
        odd = os.path.join(self.dir, "caf\udce9.cbor.gz")
        shutil.copy(SENSORS, odd)
        with Server() as server:
            code, _, err = self.run_cli(
                odd, *self.bench_args("--quiet", "--serve", str(server.port))
            )
            self.assertEqual(code, 0, err)
            (entry,) = server.store.entries()
            self.assertEqual(entry["name"], "caf\ufffd.cbor.gz")
            self.assertIsNotNone(entry["bench"])

    def test_setup_errors_stop_before_any_run(self):
        code, _, err = self.run_cli(
            *self.bench_args()[:3], "nope", *self.bench_args()[4:]
        )
        self.assertEqual(code, 2)
        self.assertIn("zli has no profile 'nope'", err)
        self.assertEqual({c["argv"][0] for c in self.calls("zli")}, {"list-profiles"})

        code, _, err = self.run_cli(
            *[a if a != self.input else self.dir for a in self.bench_args()]
        )
        self.assertEqual(code, 2)
        self.assertIn("is a directory; --bench needs a file", err)

        os.environ["ZSTD"] = os.path.join(self.dir, "missing-zstd")
        args = self.bench_args()
        del args[args.index("--zstd") : args.index("--zstd") + 2]
        code, _, err = self.run_cli(*args)
        self.assertEqual(code, 2)
        self.assertIn("not an executable file", err)

        code, _, err = self.run_cli("--bench-results", self.out("none.json"))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)
        with open(self.out("bad.json"), "w") as f:
            f.write('{"format": "something else"}')
        code, _, err = self.run_cli("--bench-results", self.out("bad.json"))
        self.assertEqual(code, 2)
        self.assertIn("is not a results file", err)

    def test_recording_failure(self):
        # zli writes the trace of a failed compression: it is reviewed, and the
        # sweep still shows how zstd does.
        os.environ["FAKE_ZLI_MODE"] = "strict"
        os.environ["FAKE_TRACE"] = os.path.join(DATA, "compressed_parquet_failure.cbor")
        results = self.out("failed.json")
        code, out, err = self.run_cli(*self.bench_args("--bench-json", results))
        self.assertEqual(code, 0, err)
        self.assertIn("zli compress -p parquet failed", err)
        self.assertIn("Stream parameter invalid", err)
        self.assertIn("compression failed", out)
        with open(results, "rb") as f:
            doc = pareto.loads(f.read())
        self.assertEqual(
            {p["series"]: p["status"] for p in doc["points"]},
            {"zli:parquet": "failed", "zstd": "ok"},
        )
        self.assertEqual(doc["trace_link"]["state"], "no_point")

        # Without a trace there is nothing to review: stop before measuring.
        os.environ["FAKE_ZLI_MODE"] = "crash"
        code, _, err = self.run_cli(*self.bench_args())
        self.assertEqual(code, 2)
        self.assertIn("zli compress -p parquet failed", err)
        self.assertIn("Segmentation fault", err)
        self.assertEqual(
            [c["argv"][0] for c in self.calls("zli")].count("benchmark"), 2
        )

    def test_unmeasured_points(self):
        os.environ["FAKE_ZSTD_MODE"] = "fail@6"
        code, out, err = self.run_cli(*self.bench_args())
        self.assertEqual(code, 0, err)
        self.assertIn("failed", out)
        code, out, err = self.run_cli(*self.bench_args("--fail-on-incomplete"))
        self.assertEqual(code, 3)
        self.assertIn("1 ratio-vs-speed point was not measured", err)

    def test_interrupt_keeps_measured_points(self):
        results = self.out("partial.json")
        seen = []

        def progress(line):
            seen.append(line)
            if line.startswith("[2/"):
                raise KeyboardInterrupt

        with mock.patch.object(codec_reviewer, "_progress", progress):
            code, out, err = self.run_cli(*self.bench_args("--bench-json", results))
        self.assertEqual(code, 130)
        with open(results, "rb") as f:
            doc = pareto.loads(f.read())
        self.assertIs(doc["complete"], False)
        self.assertEqual(doc["stopped"], "interrupted")
        self.assertEqual(
            [p["status"] for p in doc["points"]], ["ok", "ok", "skipped", "skipped"]
        )

    def test_add_to_a_running_reviewer(self):
        with Server() as server:
            port = str(server.port)
            code, _, err = self.run_cli(*self.bench_args("--quiet", "--serve", port))
            self.assertEqual(code, 0, err)
            self.assertIn(f"http://127.0.0.1:{port}/r/1", err)
            self.assertIn("Added the ratio-vs-speed results", err)
            (entry,) = server.store.entries()
            self.assertEqual(entry["name"], "sensors.parquet (-p parquet)")
            self.assertEqual(server.store.bench(1)["trace_link"]["state"], "linked")

            results = self.out("b.json")
            with open(results, "w") as f:
                f.write(pareto.dumps(server.store.bench(1)))
            code, _, err = self.run_cli("--serve", port, "--bench-results", results)
            self.assertEqual(code, 0, err)
            self.assertEqual(server.store.entries()[0]["name"], "sensors.parquet")
            # Kept as sent; the page without a trace marks no point.
            self.assertEqual(server.store.bench(2)["trace_link"]["state"], "linked")
            page = page_bench(server.store.page(2))
            self.assertEqual(page["trace_link"]["state"], "none")

        # A trace larger than the reviewer takes is refused before measuring.
        with Server(max_upload=100) as small:
            before = len(self.calls("zli"))
            code, _, err = self.run_cli(*self.bench_args("--serve", str(small.port)))
            self.assertEqual(code, 2)
            self.assertIn("the reviewer at", err)
            self.assertIn("accepts traces up to 100 bytes", err)
            benchmarks = [c["argv"][0] for c in self.calls("zli")[before:]]
            self.assertNotIn("benchmark", benchmarks)

    def test_results_are_kept_when_the_reviewer_refuses_them(self):
        tmp = self.out("tmp")
        os.mkdir(tmp)
        with mock.patch.object(tempfile, "tempdir", tmp):
            with Server(max_bench_upload=100) as server:
                code, _, err = self.run_cli(
                    *self.bench_args("--quiet", "--serve", str(server.port))
                )
        self.assertEqual(code, 2)
        self.assertIn("could not add the results", err)
        (saved,) = [n for n in os.listdir(tmp) if n.endswith(".bench.json")]
        path = os.path.join(tmp, saved)
        self.assertIn(f"The results are saved in {path}", err)
        with open(path, "rb") as f:
            self.assertEqual(len(pareto.loads(f.read())["points"]), 4)

    def test_older_reviewer_on_the_port(self):
        listing = json.dumps({"tool": review_server.TOOL, "reviews": []}).encode()

        class Old(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(listing)))
                self.end_headers()
                self.wfile.write(listing)

            def log_message(self, *args):
                pass

        old = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Old)
        thread = threading.Thread(target=old.serve_forever, daemon=True)
        thread.start()
        try:
            port = str(old.server_address[1])
            code, _, err = self.run_cli(*self.bench_args("--serve", port))
        finally:
            old.shutdown()
            old.server_close()
        self.assertEqual(code, 2)
        self.assertIn("older version that cannot show ratio-vs-speed results", err)
        # Nothing was recorded or measured.
        self.assertEqual({c["argv"][0] for c in self.calls("zli")}, {"list-profiles"})

    def test_serve_results(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(TOOL, "codec_reviewer.py"),
                "--quiet",
                *self.bench_args("--serve", "0"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            line = ""
            while "Codec reviewer at" not in line:
                ready, _, _ = select.select([proc.stderr], [], [], 60)
                self.assertTrue(ready, "the server did not start")
                line = proc.stderr.readline().decode()
                self.assertTrue(line, "the command ended before serving")
            self.assertIn("with ratio vs speed", line)
            port = int(line.split("http://127.0.0.1:")[1].split("/")[0])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("GET", "/api/reviews/1/benchmark")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            doc = json.loads(response.read())
            conn.close()
            self.assertEqual(doc["trace_link"]["state"], "linked")
        finally:
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.wait(timeout=30), 0)
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
