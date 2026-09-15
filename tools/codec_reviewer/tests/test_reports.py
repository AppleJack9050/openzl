# Copyright (c) Meta Platforms, Inc. and affiliates.

import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import codec_reviewer  # noqa: E402
import html_report  # noqa: E402
import text_report  # noqa: E402
import trace_builder as tb  # noqa: E402
from pipelines import analyze  # noqa: E402
from trace_format import load_trace, load_trace_bytes  # noqa: E402

DATA = os.path.join(HERE, "data")
SENSORS = os.path.join(DATA, "sensors_chunks.cbor.gz")


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stderr(err):
        code = codec_reviewer.main(list(args), stdout=out)
    return code, out.getvalue(), err.getvalue()


class TextReportTest(unittest.TestCase):
    def setUp(self):
        self.trace = load_trace(SENSORS)
        self.analysis = analyze(self.trace)

    def test_summary_sections(self):
        text = text_report.render_summary("sensors.cbor", self.trace, self.analysis)
        self.assertIn("Codec review: sensors.cbor", text)
        self.assertIn("all 4 chunks", text)
        self.assertIn("2,889,011 B in → 560,780 B in streams (5.15×)", text)
        self.assertIn("Coverage OK: 595 of 595 codec nodes", text)
        for heading in (
            "START (5)",
            "MERGES (8)",
            "OUTPUTS OF dispatchN_byTag",
            "CODECS",
        ):
            self.assertIn(heading, text)
        self.assertNotIn("FAILURES", text)

    def test_ascii_output_is_ascii(self):
        text = text_report.render_summary(
            "x", self.trace, self.analysis, 0, text_report.ASCII
        )
        for p in self.analysis.pipelines:
            text += text_report.render_pipeline(
                p, self.analysis, glyphs=text_report.ASCII
            )
        text.encode("ascii")

    def test_pipeline_tree(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(9000)
        outs = ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 3000)] * 3)
        converted = [
            ch.one("zl.convert_serial_to_num_le32", o, 3000, tb.NUMERIC, 4)
            for o in outs
        ]
        (joined,) = ch.codec("zl.concat_num", converted, [(tb.NUMERIC, 4, 9000)])
        a_out, b_out = ch.codec(
            "zl.field_lz",
            [joined],
            [(tb.NUMERIC, 4, 2000), (tb.SERIAL, 1, 500)],
            header=4,
            ints=[(181, 3)],
        )
        ch.one("zl.private.zstd", a_out, 800, ints=[(100, 19), (160, 1)])
        ch.store_rest()
        a = analyze(load_trace_bytes(trace.to_cbor()))
        tree = text_report.render_pipeline(a.get("m1"), a)
        self.assertIn("3 streams in (9,000 B)", tree)
        self.assertIn("└── concat_num", tree)
        self.assertIn("field_lz  level 3 · writes 500 B · header 4 B", tree)
        self.assertIn("#0 zstd  level 19 · long matching on · writes 800 B", tree)
        feeder = text_report.render_pipeline(a.get("S1"), a)
        self.assertIn("→ concat_num, joins M1", feeder)
        folded = text_report.render_pipeline(a.get("S1"), a, show_conversions=False)
        shown = text_report.render_pipeline(a.get("S1"), a, show_conversions=True)
        self.assertIn(
            "└── → concat_num, joins M1 (via convert_serial_to_num_le32)", folded
        )
        self.assertNotIn("── convert_serial_to_num_le32", folded)
        self.assertIn("└── convert_serial_to_num_le32", shown)

    def test_failures_are_listed(self):
        trace = load_trace(os.path.join(DATA, "compressed_parquet_failure.cbor"))
        a = analyze(trace)
        text = text_report.render_summary("bad.cbor", trace, a)
        self.assertIn("nothing written: compression failed", text)
        self.assertIn("FAILURES (1)", text)
        self.assertIn("Found compressed chunk", text)
        tree = text_report.render_pipeline(a.pipelines[0], a)
        self.assertIn("segmenter  ⚠ FAILED  chunk size 20 MB", tree)

    def test_abandoned_attempts_and_folding(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(4000)
        n = ch.one("zl.convert_serial_to_num_le32", s, 4000, tb.NUMERIC, 4)
        ch.codec("zl.field_lz", [n], [], failure="Message: field_lz failed on column 7")
        ch.one("zl.private.zstd", n, 900)
        ch.store_rest()
        a = analyze(load_trace_bytes(trace.to_cbor()))
        tree = text_report.render_pipeline(a.pipelines[0], a)
        # The conversion is not folded away, so the failure under it stays visible.
        self.assertIn("└── convert_serial_to_num_le32", tree)
        self.assertIn("├── field_lz  ⚠ FAILED  (abandoned)", tree)
        self.assertIn("⚠ Message: field_lz failed on column 7", tree)
        self.assertIn("└── zstd  writes 900 B", tree)

    def test_split_at_the_start(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        ch.codecs.append(
            dict(
                name="zl.#start",
                header=0,
                ints=[],
                copies=[],
                inputs=[],
                outputs=[],
                failure=None,
                standard=True,
            )
        )
        for i in range(30):
            sid = ch._new_stream(i, tb.SERIAL, 1, 1000, 1000)
            ch.codecs[0]["outputs"].append(sid)
        for sid in list(ch.codecs[0]["outputs"]):
            ch.one("zl.private.zstd", sid, 100)
        ch.store_rest()
        a = analyze(load_trace_bytes(trace.to_cbor()))
        tree = text_report.render_pipeline(a.get("E1"), a)
        self.assertIn("└── → 30 outputs, each its own pipeline: S1", tree)


class HtmlReportTest(unittest.TestCase):
    def test_report_embeds_all_views(self):
        trace = load_trace(SENSORS)
        data = html_report.report_data("sensors.cbor", trace)
        self.assertEqual(
            [s["key"] for s in data["selections"]], ["all", "0", "1", "2", "3", "4"]
        )
        page = html_report.render_html(data)
        self.assertNotIn("__DATA__", page)
        self.assertIn("<title>Codec review: sensors.cbor</title>", page)
        payload = re.search(r"const DATA = (.*);\n", page).group(1)
        self.assertEqual(json.loads(payload), json.loads(json.dumps(data)))

    def test_trace_text_cannot_escape_the_script(self):
        for name in (
            "zl.x--><img src=x onerror=alert(1)>",
            "zl.<!--<script>",
            "zl.</script><b>",
        ):
            with self.subTest(name=name):
                trace = tb.TraceBuilder()
                ch = trace.chunk()
                s = ch.start(100)
                ch.one(name, s, 50)
                ch.store_rest()
                data = html_report.report_data(
                    "t.cbor", load_trace_bytes(trace.to_cbor())
                )
                page = html_report.render_html(data)
                # The page is the template with the payload at its one marker, and
                # the payload holds nothing the HTML parser acts on.
                with open(html_report.TEMPLATE, encoding="utf-8") as f:
                    template = f.read()
                head, tail = template.replace(
                    "__TITLE__", "Codec review: t.cbor"
                ).split("__DATA__")
                self.assertTrue(page.startswith(head) and page.endswith(tail))
                payload = page[len(head) : len(page) - len(tail)]
                for text in ("<", ">", "&"):
                    self.assertNotIn(text, payload)
                self.assertIn(
                    name[3:], json.dumps(json.loads(payload), ensure_ascii=False)
                )

    def test_markers_are_substituted_once(self):
        trace = load_trace_bytes(tb.serial_trace().to_cbor())
        page = html_report.render_html(html_report.report_data("x__DATA__.cbor", trace))
        self.assertIn("<title>Codec review: x__DATA__.cbor</title>", page)
        self.assertEqual(page.count('"tool":'), 1)

        # A template that names a marker twice (say, in a comment) is refused.
        with open(html_report.TEMPLATE, encoding="utf-8") as f:
            template = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "template.html")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("<!-- __DATA__ -->" + template)
            saved, html_report.TEMPLATE = html_report.TEMPLATE, bad
            try:
                with self.assertRaises(ValueError):
                    html_report.render_html(html_report.report_data("t", trace))
            finally:
                html_report.TEMPLATE = saved

    def test_trace_text_cannot_close_the_script(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(100)
        ch.one("zl.</script><script>alert(1)</script>", s, 50)
        ch.store_rest()
        data = html_report.report_data("<evil>.cbor", load_trace_bytes(trace.to_cbor()))
        page = html_report.render_html(data)
        script = page[page.index("const DATA") :]
        self.assertNotIn("</script><script>", script.split("\n", 1)[0])
        self.assertIn("<title>Codec review: &lt;evil&gt;.cbor</title>", page)


class CliTest(unittest.TestCase):
    def test_summary_show_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            html = os.path.join(tmp, "r.html")
            js = os.path.join(tmp, "r.json")
            code, out, err = run_cli(
                SENSORS, "--show", "M1,E1", "--html", html, "--json", js, "--limit", "3"
            )
            self.assertEqual(code, 0, err)
            self.assertIn("Codec review: sensors_chunks.cbor.gz", out)
            self.assertIn("\nM1  ", out)
            self.assertIn("\nE1  Top level", out)
            self.assertTrue(os.path.getsize(html) > 10_000)
            with open(js) as f:
                self.assertEqual(
                    json.load(f)["selections"][0]["analysis"]["coverage"]["ok"], True
                )

    def test_one_chunk(self):
        code, out, _ = run_cli(SENSORS, "--chunk", "2", "--quiet", "--show", "all")
        self.assertEqual(code, 0)
        self.assertIn("E1  Input", out)
        self.assertEqual(run_cli(SENSORS, "--chunk", "9")[0], 2)

    def test_unwritable_output_and_deep_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = run_cli(
                SENSORS, "--quiet", "--html", os.path.join(tmp, "missing", "r.html")
            )
            self.assertEqual(code, 2)
            self.assertIn("cannot write", err)

            trace = tb.TraceBuilder()
            ch = trace.chunk()
            s = ch.start(1000)
            for _ in range(1500):
                s = ch.one("zl.delta_int", s, 1000)
            ch.store_rest()
            path = os.path.join(tmp, "deep.cbor")
            with open(path, "wb") as f:
                f.write(trace.to_cbor())
            html = os.path.join(tmp, "deep.html")
            code, out, err = run_cli(
                path, "--show", "E1", "--html", html, "--json", html + ".json"
            )
            self.assertEqual(code, 0, err)
            self.assertIn("1501 codecs per run", out)

    def test_failed_compression(self):
        code, out, _ = run_cli(
            os.path.join(DATA, "compressed_parquet_failure.cbor"),
            "--fail-on-incomplete",
        )
        self.assertEqual(code, 0)
        self.assertIn("nothing written: compression failed", out)
        self.assertIn("Found compressed chunk", out)

    def test_errors(self):
        code, _, err = run_cli(os.path.join(DATA, "missing.cbor"))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)
        with tempfile.NamedTemporaryFile(suffix=".cbor") as f:
            f.write(b"not a trace")
            f.flush()
            code, _, err = run_cli(f.name)
        self.assertEqual(code, 2)
        self.assertIn("not a readable trace", err)
        code, _, err = run_cli(SENSORS, "--quiet", "--show", "Z9")
        self.assertEqual(code, 2)
        self.assertIn("no pipeline Z9", err)


if __name__ == "__main__":
    unittest.main()
