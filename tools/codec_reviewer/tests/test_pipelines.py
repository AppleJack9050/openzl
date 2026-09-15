# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import text_report  # noqa: E402
import trace_builder as tb  # noqa: E402
from pipelines import analyze, main_chain, settings, title, walk_nodes  # noqa: E402
from trace_format import load_trace, load_trace_bytes, trace_from_cbor  # noqa: E402

DATA = os.path.join(HERE, "data")


def load(builder, fmt="cbor"):
    data = builder.to_cbor() if fmt == "cbor" else builder.to_dot().encode()
    return load_trace_bytes(data)


def by_kind(analysis, kind):
    return [p for p in analysis.pipelines if p.kind == kind]


def column_trace():
    """#start -> dispatchN_byTag -> 5 outputs.

    #0 goes straight to the frame, #1 and #2 run the same delta_int ->
    bitpack_int pipeline, #3 runs zstd, and #4 runs delta_int -> bitpack_int
    with a different delta_int parameter.
    """
    trace = tb.TraceBuilder()
    ch = trace.chunk()
    s = ch.start(10_000)
    outs = ch.codec(
        "zl.dispatchN_byTag",
        [s],
        [
            (tb.SERIAL, 1, 500),
            (tb.NUMERIC, 4, 4000),
            (tb.NUMERIC, 4, 4000),
            (tb.SERIAL, 1, 1000),
            (tb.NUMERIC, 4, 500),
        ],
        header=20,
    )
    for sid, content in ((outs[1], 900), (outs[2], 700)):
        d = ch.one("zl.delta_int", sid, 4000, tb.NUMERIC, 4)
        ch.one("zl.bitpack_int", d, content)
    ch.one("zl.private.zstd", outs[3], 300, ints=[(100, 3)])
    d = ch.one("zl.delta_int", outs[4], 500, tb.NUMERIC, 4, ints=[(1, 2)])
    ch.one("zl.bitpack_int", d, 100)
    ch.store_rest()
    return trace


class GroupingTest(unittest.TestCase):
    def test_serial_trace_is_one_entry_pipeline(self):
        a = analyze(load(tb.serial_trace(1000, 400)))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual([p.id for p in a.pipelines], ["E1"])
        (p,) = a.pipelines
        self.assertEqual((p.raw, p.csize, p.nodes, p.count), (1000, 400, 2, 1))
        self.assertEqual((a.input, a.csize, a.codecs_run), (1000, 400, 1))
        self.assertEqual(main_chain(p), "zstd")
        self.assertIn("level 7", settings(p.tree.children[0].node))

    def test_identical_column_pipelines_are_grouped(self):
        for fmt in ("cbor", "dot"):
            with self.subTest(fmt=fmt):
                a = analyze(load(column_trace(), fmt))
                self.assertTrue(a.coverage.ok, a.coverage)
                splits = by_kind(a, "split")
                self.assertEqual(
                    [(p.id, p.outputs, p.count) for p in splits],
                    [("S1", [1, 2], 2), ("S2", [3], 1), ("S3", [4], 1)],
                )
                pair = splits[0]
                self.assertEqual((pair.raw, pair.csize, pair.nodes), (8000, 1600, 2))
                self.assertEqual(pair.splitter, "dispatchN_byTag")
                self.assertEqual(title(pair, a), ("2 outputs", "32-bit numbers"))
                # The stored output stays with the dispatch node as bytes it writes.
                (entry,) = by_kind(a, "entry")
                dispatch = entry.tree.children[0].node
                self.assertEqual([c.kind for c in dispatch.children], ["store"])
                self.assertEqual(entry.stored, 500)
                self.assertEqual(dispatch.split.outputs, 4)
                self.assertEqual(dispatch.split.pipelines, ["S1", "S2", "S3"])
                self.assertEqual(entry.csize, 20 + 500 + 1600 + 300 + 100)

    def test_merges_and_the_outputs_feeding_them(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(9000)
        outs = ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 3000)] * 3)
        converted = [
            ch.one("zl.convert_serial_to_num_le32", o, 3000, tb.NUMERIC, 4)
            for o in outs
        ]
        (joined,) = ch.codec(
            "zl.concat_num", converted, [(tb.NUMERIC, 4, 9000)], header=6
        )
        ch.one("zl.private.zstd", joined, 1200)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        (merge,) = by_kind(a, "merge")
        self.assertEqual(
            (merge.id, merge.fanin, merge.raw, merge.csize, merge.nodes),
            ("M1", 3, 9000, 1206, 2),
        )
        self.assertEqual(
            merge.origins,
            [("dispatchN_byTag", 0), ("dispatchN_byTag", 1), ("dispatchN_byTag", 2)],
        )
        (feeder,) = by_kind(a, "split")
        self.assertTrue(feeder.feeder)
        self.assertEqual((feeder.count, feeder.nodes, feeder.stored), (3, 1, 0))
        self.assertEqual(feeder.tree.node.children[0].kind, "junction")
        self.assertEqual(feeder.tree.node.children[0].merges, ["M1"])
        self.assertEqual(
            main_chain(feeder), "convert_serial_to_num_le32 → concat_num (M1)"
        )
        # A merge's cost is split evenly across its inputs.
        self.assertEqual(feeder.csize, 3 * (1206 // 3))

    def test_children_follow_output_index_not_stream_id(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(2000)
        outs = ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 1000)] * 3)
        for n, sid in enumerate(outs):
            if n == 1:
                # Same codec, but its streams were created in the other order.
                lens, toks = ch.codec(
                    "zl.tokenize_numeric",
                    [sid],
                    [(tb.SERIAL, 1, 100), (tb.NUMERIC, 4, 800)],
                    output_indices=[1, 0],
                )
            else:
                toks, lens = ch.codec(
                    "zl.tokenize_numeric",
                    [sid],
                    [(tb.NUMERIC, 4, 800), (tb.SERIAL, 1, 100)],
                )
            ch.one("zl.private.zstd", toks, 200)
        ch.store_rest()
        for fmt in ("cbor", "dot"):
            with self.subTest(fmt=fmt):
                a = analyze(load(trace, fmt))
                (split,) = by_kind(a, "split")
                self.assertEqual(split.count, 3)
                kids = split.tree.node.children
                self.assertEqual(
                    [(c.index, c.type, c.kind) for c in kids],
                    [(0, "numeric", "node"), (1, "serial", "store")],
                )

    def test_wide_transforms_stay_inside_their_pipeline(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(8000)
        (n,) = ch.codec("zl.convert_serial_to_num_le64", [s], [(tb.NUMERIC, 8, 8000)])
        for sid in ch.codec("zl.transpose_split", [n], [(tb.SERIAL, 1, 1000)] * 8):
            ch.one("zl.private.huffman_v2", sid, 100)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertEqual([p.kind for p in a.pipelines], ["entry"])
        self.assertEqual(a.pipelines[0].nodes, 11)

        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(2400)
        for sid in ch.codec("zl.custom_split", [s], [(tb.SERIAL, 1, 100)] * 24):
            ch.one("zl.private.zstd", sid, 10)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertEqual(
            sorted((p.kind, p.count) for p in a.pipelines),
            [("entry", 1), ("split", 24)],
        )

    def test_graph_instance_numbers_are_ignored_but_settings_are_not(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(4000)
        outs = ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 1000)] * 4)
        for i, (sid, level) in enumerate(zip(outs, (7, 7, 7, 9))):
            ch.begin_graph(f"zl.ace#{i}", gtype=5, ints=[(100, level)])
            ch.one("zl.private.zstd", sid, 100)
            ch.end_graph()
        ch.store_rest()
        a = analyze(load(trace))
        splits = by_kind(a, "split")
        self.assertEqual(sorted(p.count for p in splits), [1, 3])
        levels = {p.count: settings(p.tree.node) for p in splits}
        self.assertEqual(levels, {3: ["level 7"], 1: ["level 9"]})

    def test_codec_instance_numbers_are_ignored(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(3000)
        outs = ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 1000)] * 3)
        for i, sid in enumerate(outs):
            ch.one(f"zl.private.field_lz#{i + 1}", sid, 100)
        ch.store_rest()
        (split,) = by_kind(analyze(load(trace)), "split")
        self.assertEqual((split.count, split.codecs), (3, {"field_lz"}))

    def test_segmented_trace(self):
        trace = tb.TraceBuilder()
        top = trace.chunk()
        s = top.start(2000)
        top.codec("segmenter", [s], [], ints=[(2, 1_000_000)], standard=False)
        for compressed in (300, 200):
            ch = trace.chunk()
            s = ch.start(1000)
            ch.one("zl.private.zstd", s, compressed)
            ch.store_rest()
        loaded = load(trace)
        a = analyze(loaded)
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.chunk_count, a.input, a.csize), (2, 2000, 500))
        top_p, every = by_kind(a, "entry")
        self.assertTrue(top_p.top_level)
        self.assertEqual((top_p.id, top_p.csize, top_p.chunk_count), ("E1", 500, 2))
        self.assertEqual(title(top_p, a), ("Top level", "segmenter → 2 chunks"))
        self.assertIn("chunk size 1 MB", settings(top_p.tree.children[0].node))
        self.assertEqual((every.count, title(every, a)[0]), (2, "Every chunk"))

        one = analyze(loaded, 2)
        self.assertEqual((one.chunk_count, one.input, one.csize), (1, 1000, 200))
        top_only = analyze(loaded, 0)
        self.assertEqual((top_only.input, top_only.csize), (2000, None))
        with self.assertRaises(ValueError):
            analyze(loaded, 7)

    def test_failed_attempts_and_unfinished_streams(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(1000)
        a_out, b_out = ch.codec(
            "zl.tokenize_numeric", [s], [(tb.NUMERIC, 4, 400), (tb.NUMERIC, 4, 600)]
        )
        ch.codec("zl.field_lz", [a_out], [], failure="Message: offsets too large")
        ch.one("zl.private.zstd", a_out, 150)
        ch.store_rest()
        # b_out never got compressed: finalizeTrace marks it in progress.
        store = next(c for c in ch.codecs if c["inputs"] == [b_out])
        store["name"] = "zl.#in_progress"
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.failures, a.unfinished, a.compression_failed), (1, 1, True))
        (p,) = a.pipelines
        self.assertTrue(p.failed)
        self.assertEqual(p.nodes, 5)  # start, tokenize, field_lz, zstd, #in_progress
        tok = p.tree.children[0].node
        first, second = tok.children
        (attempt,) = first.attempts
        self.assertEqual(
            (attempt.codec, attempt.failures),
            ("zl.field_lz", ["Message: offsets too large"]),
        )
        self.assertEqual(first.node.codec, "zl.private.zstd")
        self.assertEqual(second.kind, "progress")


class FailureTest(unittest.TestCase):
    """How the tracer records failures (ChunkTrace::on_migraphEncode_end, finalizeTrace)."""

    def test_failed_graph_placeholder_then_fallback(self):
        # Permissive mode: a graph fails before running a codec, the tracer adds an
        # #in_progress placeholder carrying the graph's error, then zstd takes over.
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(1000)
        ch.begin_graph("sddl#8", gtype=3, failure="Message: description does not match")
        ch.codec("zl.#in_progress", [s], [])
        ch.end_graph()
        ch.one("zl.private.zstd", s, 600)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual(
            (a.failures, a.unfinished, a.compression_failed), (1, 0, False)
        )
        (p,) = a.pipelines
        self.assertEqual((p.nodes, p.csize, p.failed), (3, 600, True))
        stream = p.tree.children[0]
        (placeholder,) = stream.attempts
        self.assertEqual(
            placeholder.graph_failures, ["Message: description does not match"]
        )
        self.assertEqual(stream.node.codec, "zl.private.zstd")

    def test_strict_failure_before_any_codec(self):
        obj = {
            "traceVersion": 1,
            "chunks": [
                {
                    "streams": [],
                    "codecs": [
                        {"name": "zl.#start", "inputStreams": [], "outputStreams": []},
                        {
                            "name": "zl.#in_progress",
                            "inputStreams": [0],
                            "outputStreams": [],
                        },
                    ],
                    "graphs": [
                        {
                            "gName": "sddl#8",
                            "gType": 3,
                            "gFailureString": "Message: boom",
                            "codecIDs": [1],
                        }
                    ],
                }
            ],
        }
        a = analyze(trace_from_cbor(obj))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.failures, a.unfinished, a.compression_failed), (1, 1, True))
        (p,) = a.pipelines
        self.assertIsNone(p.csize)
        self.assertEqual(p.tree.children[0].progress_messages, ["Message: boom"])

    def test_strict_failure_of_the_only_consumer(self):
        # zli --strict: the conversion fails, nothing else runs and no placeholder
        # is added, because the stream was consumed.
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(100_001)
        ch.codec("zl.convert_serial_to_num_le16", [s], [], failure="Message: odd size")
        for fmt in ("cbor", "dot"):
            a = analyze(load(trace, fmt))
            self.assertTrue(a.coverage.ok, a.coverage)
            self.assertEqual(
                (a.failures, a.unfinished, a.compression_failed), (1, 1, True)
            )
            self.assertIsNone(a.pipelines[0].csize)
            self.assertIsNone(a.csize)
            summary = text_report.render_summary("t", load(trace, fmt), a)
            self.assertIn("100,001 B in, nothing written: compression failed", summary)
            # The failed stream has no compressed size to show.
            tree = text_report.render_pipeline(a.pipelines[0], a)
            self.assertNotIn("100,001 B\n", tree)
            self.assertNotIn("1.00×", summary)

    def test_placeholder_in_abandoned_work_is_not_a_failed_compression(self):
        # A conversion succeeds, the graph it leads to fails before running a codec
        # (so the tracer adds a placeholder), and the fallback compresses the
        # original stream.
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(4000)
        n = ch.one("zl.convert_serial_to_num_le32", s, 4000, tb.NUMERIC, 4)
        ch.begin_graph("my_graph#4", gtype=3, failure="Message: unsupported")
        ch.codec("zl.#in_progress", [n], [])
        ch.end_graph()
        ch.one("zl.private.zstd", s, 900)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual(
            (a.failures, a.unfinished, a.compression_failed, a.csize),
            (1, 0, False, 900),
        )
        tree = text_report.render_pipeline(a.pipelines[0], a)
        self.assertIn("graph failed before running a codec", tree)
        self.assertNotIn("never compressed", tree)
        self.assertIn("4.44×", text_report.render_summary("t", load(trace), a))

    def test_placeholder_with_several_inputs(self):
        # A graph taking two streams fails before running a codec.
        def build(fallback):
            trace = tb.TraceBuilder()
            ch = trace.chunk()
            s = ch.start(2000)
            a_, b_ = ch.codec(
                "zl.split", [s], [(tb.SERIAL, 1, 1000), (tb.SERIAL, 1, 1000)]
            )
            ch.begin_graph("cluster#2", gtype=4, failure="Message: bad config")
            ch.codec("zl.#in_progress", [a_, b_], [])
            ch.end_graph()
            if fallback:
                ch.one("zl.private.zstd", a_, 300)
                ch.one("zl.private.zstd", b_, 300)
            ch.store_rest()
            return load(trace), analyze(load(trace))

        # Permissive: each stream is compressed on its own afterwards. The
        # placeholder is an abandoned attempt on its first input, not a merge.
        _, a = build(fallback=True)
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.coverage.merges_total, len(a.pipelines)), (0, 1))
        self.assertEqual((a.unfinished, a.compression_failed, a.csize), (0, False, 600))
        first, second = a.pipelines[0].tree.children[0].node.children
        self.assertEqual([n.codec for n in first.attempts], ["zl.#in_progress"])
        self.assertEqual(second.attempts, [])

        # Strict: both streams stop there; the error is reported once.
        trace, a = build(fallback=False)
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.coverage.merges_total, a.unfinished), (0, 2))
        self.assertTrue(a.compression_failed)
        summary = text_report.render_summary("t", trace, a)
        self.assertEqual(summary.count("bad config"), 1)
        tree = text_report.render_pipeline(a.pipelines[0], a)
        self.assertEqual(tree.count("never compressed"), 2)
        self.assertIn("same failure as another input", tree)
        streams = a.pipelines[0].tree.children[0].node.children
        self.assertEqual([s.kind for s in streams], ["progress", "progress"])
        self.assertEqual(
            [s.progress_messages for s in streams], [["Message: bad config"], []]
        )

    def test_work_rolled_back_by_permissive_mode_is_kept(self):
        # A conversion succeeds, the codec after it fails, and the fallback graph
        # compresses the original stream again.
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(4000)
        ch.begin_graph("zl.ace#1", gtype=5)
        n = ch.one("zl.convert_serial_to_num_le32", s, 4000, tb.NUMERIC, 4)
        ch.codec("zl.delta_int", [n], [], failure="Message: overflow")
        ch.end_graph()
        ch.one("zl.private.zstd", s, 900)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        self.assertEqual((a.failures, a.compression_failed), (1, False))
        stream = a.pipelines[0].tree.children[0]
        (attempt,) = stream.attempts
        self.assertEqual(
            [(n.codec, n.failures) for n in walk_nodes(attempt)],
            [
                ("zl.convert_serial_to_num_le32", []),
                ("zl.delta_int", ["Message: overflow"]),
            ],
        )
        self.assertEqual(stream.node.codec, "zl.private.zstd")

        # A function graph ran tokenize and then failed; tokenize's outputs were
        # never consumed, so the tracer sent them to store.
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(4000)
        ch.begin_graph("my_graph#3", gtype=3, failure="Message: graph failed")
        ch.codec("zl.tokenize", [s], [(tb.NUMERIC, 2, 800), (tb.SERIAL, 1, 100)])
        ch.end_graph()
        ch.one("zl.private.zstd", s, 900)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        # tokenize's stores were rolled back with it: only zstd's output is written.
        self.assertEqual((a.coverage.stored, a.coverage.stored_total), (900, 900))
        self.assertEqual(a.pipelines[0].stored, 900)
        stream = a.pipelines[0].tree.children[0]
        (attempt,) = stream.attempts
        self.assertEqual(
            (attempt.codec, attempt.graph_failures),
            ("zl.tokenize", ["Message: graph failed"]),
        )

    def test_graph_error_is_reported_once(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(1000)
        ch.begin_graph("my_graph#1", gtype=3, failure="Message: graph failed")
        a_out = ch.one("zl.delta_int", s, 1000, tb.NUMERIC, 4)
        ch.one("zl.bitpack_int", a_out, 200)
        ch.end_graph()
        ch.store_rest()
        a = analyze(load(trace))
        messages = [
            m for n in walk_nodes(a.pipelines[0].tree) for m in n.graph_failures
        ]
        self.assertEqual(messages, ["Message: graph failed"])
        self.assertEqual(a.failures, 1)

    def test_segmented_compression_that_failed_part_way(self):
        trace = tb.TraceBuilder()
        top = trace.chunk()
        s = top.start(2000)
        top.codec(
            "segmenter",
            [s],
            [],
            ints=[(100, 1000)],
            standard=False,
            failure="Message: bad row",
        )
        ch = trace.chunk()
        s = ch.start(1000)
        ch.one("zl.private.zstd", s, 300)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.compression_failed)
        top_p = a.pipelines[0]
        self.assertTrue(top_p.top_level)
        self.assertIsNone(top_p.csize)
        self.assertEqual((a.input, a.csize), (2000, 300))
        self.assertIn("chunk size 1,000 B", settings(top_p.tree.children[0].node))
        summary = text_report.render_summary("t", load(trace), a)
        self.assertIn("2,000 B in; compression failed after 300 B in streams", summary)
        # The chunk that finished keeps its ratio; the whole file gets none.
        rows = [line for line in summary.splitlines() if "×" in line]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].startswith("  E2            300    3.33×"), rows)


class CutTest(unittest.TestCase):
    def test_outputs_of_different_splitters_stay_apart(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(10_000)
        outs = ch.codec(
            "zl.dispatchN_byTag",
            [s],
            [(tb.SERIAL, 1, 1000)] * 3 + [(tb.STRING, 0, 7000)],
        )
        for sid in outs[:3]:
            ch.one("zl.private.zstd", sid, 100)
        for sid in ch.codec(
            "zl.dispatch_string", [outs[3]], [(tb.SERIAL, 1, 1000)] * 7
        ):
            ch.one("zl.private.zstd", sid, 50)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertTrue(a.coverage.ok, a.coverage)
        splits = {(p.splitter, tuple(p.outputs), p.count) for p in by_kind(a, "split")}
        self.assertEqual(
            splits,
            {
                ("dispatchN_byTag", (0, 1, 2), 3),
                ("dispatchN_byTag", (3,), 1),
                ("dispatch_string", tuple(range(7)), 7),
            },
        )

    def test_transpose_is_never_a_split_point(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(3200)
        (n,) = ch.codec("zl.convert_serial_to_struct", [s], [(tb.STRUCT, 32, 3200)])
        for sid in ch.codec("zl.transpose_split", [n], [(tb.SERIAL, 1, 100)] * 32):
            ch.one("zl.private.huffman_v2", sid, 10)
        ch.store_rest()
        a = analyze(load(trace))
        self.assertEqual([p.kind for p in a.pipelines], ["entry"])

    def test_split_output_counts_are_per_run(self):
        trace = tb.TraceBuilder()
        top = trace.chunk()
        s = top.start(2000)
        top.codec("segmenter", [s], [], standard=False)
        for _ in range(2):
            ch = trace.chunk()
            s = ch.start(1000)
            for sid in ch.codec("zl.dispatchN_byTag", [s], [(tb.SERIAL, 1, 250)] * 4):
                ch.one("zl.private.zstd", sid, 50)
            ch.store_rest()
        a = analyze(load(trace))
        every = next(p for p in a.pipelines if p.kind == "entry" and not p.top_level)
        split = every.tree.children[0].node.split
        self.assertEqual((split.outputs, split.runs, split.per_run), (8, 2, 4))
        self.assertEqual(main_chain(every), "dispatchN_byTag → 4 outputs")


class RealTraceTest(unittest.TestCase):
    def test_sensor_table_in_four_chunks(self):
        for name in ("sensors_chunks.cbor.gz", "sensors_chunks.dot.gz"):
            with self.subTest(name=name):
                trace = load_trace(os.path.join(DATA, name))
                a = analyze(trace)
                self.assertTrue(a.coverage.ok, a.coverage)
                self.assertEqual(
                    (a.chunk_count, a.input, a.csize), (4, 2_889_011, 560_780)
                )
                self.assertEqual(
                    (a.coverage.nodes_total, a.coverage.stored_total), (595, 560_435)
                )
                self.assertEqual(
                    (a.coverage.split_outputs_total, a.coverage.merges_total), (36, 8)
                )
                kinds = [p.kind for p in a.pipelines]
                self.assertEqual(
                    (kinds.count("entry"), kinds.count("merge"), kinds.count("split")),
                    (5, 8, 12),
                )
                for chunk in trace.chunks:
                    self.assertTrue(analyze(trace, chunk.index).coverage.ok)

    def test_compression_that_failed(self):
        trace = load_trace(os.path.join(DATA, "compressed_parquet_failure.cbor"))
        a = analyze(trace)
        self.assertEqual(a.failures, 1)
        self.assertEqual(a.coverage.stored_total, 0)
        segmenter = a.pipelines[0].tree.children[0].node
        self.assertIn("Found compressed chunk", segmenter.failures[0])
        # The segmenter never produced a chunk: no size, no ratio.
        self.assertEqual((a.unfinished, a.compression_failed), (1, True))
        self.assertIsNone(a.pipelines[0].csize)
        self.assertNotIn("×", text_report.render_summary("t", trace, a))


if __name__ == "__main__":
    unittest.main()
