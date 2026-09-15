# Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import trace_builder as tb  # noqa: E402
from trace_format import (  # noqa: E402
    TraceFormatError,
    decode_cbor,
    load_trace,
    load_trace_bytes,
    trace_from_cbor,
)

DATA = os.path.join(HERE, "data")


def normalized(trace):
    """Order-independent view of a trace, for comparing CBOR and DOT readings."""
    out = []
    for chunk in trace.chunks:
        out.append(
            (
                [
                    (s.type, s.output_index, s.width, s.elements, s.csize)
                    for s in chunk.streams
                ],
                [
                    (
                        c.name,
                        c.header_size,
                        c.params.ints,
                        [p for p, _ in c.params.copies],
                        sorted(c.inputs),
                        sorted(c.outputs),
                        bool(c.failure),
                        None if c.graph is None else chunk.graphs[c.graph].name,
                    )
                    for c in chunk.codecs
                ],
                [(g.name, g.type, g.params.ints, g.codecs) for g in chunk.graphs],
            )
        )
    return out


class DecodeCborTest(unittest.TestCase):
    def test_round_trips_encoder_values(self):
        value = {
            "small": 5,
            "byte": 200,
            "short": 60000,
            "long": 1 << 40,
            "negative": -1000,
            "text": "zl.private.field_lz",
            "bytes": b"\x00\x01",
            "list": [True, False, None, 1.5],
        }
        self.assertEqual(decode_cbor(tb.encode_cbor(value)), value)

    def test_half_and_single_floats(self):
        self.assertEqual(decode_cbor(b"\xf9\x34\x00"), 0.25)
        self.assertEqual(decode_cbor(b"\xf9\x3e\x00"), 1.5)
        self.assertEqual(decode_cbor(b"\xfa" + struct.pack(">f", 2.5)), 2.5)

    def test_indefinite_lengths_and_tags(self):
        self.assertEqual(decode_cbor(b"\x9f\x01\x02\xff"), [1, 2])
        self.assertEqual(decode_cbor(b"\xbf\x61a\x01\xff"), {"a": 1})
        self.assertEqual(decode_cbor(b"\x7f\x61a\x61b\xff"), "ab")
        self.assertEqual(decode_cbor(b"\x5f\x41\x01\x41\x02\xff"), b"\x01\x02")
        self.assertEqual(decode_cbor(b"\xc1\x1a\x00\x00\x00\x07"), 7)

    def test_truncated_data_is_an_error(self):
        with self.assertRaises(TraceFormatError):
            decode_cbor(b"\x82\x01")
        with self.assertRaises(TraceFormatError):
            decode_cbor(b"\x63ab")


class LoadTraceTest(unittest.TestCase):
    def build(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s0 = ch.start(1000)
        ch.begin_graph("zl.ace#3", gtype=5, ints=[(100, 9)])
        a, b = ch.codec(
            "zl.tokenize_numeric",
            [s0],
            [(tb.NUMERIC, 4, 400), (tb.NUMERIC, 2, 300)],
            header=3,
            ints=[(0, 1)],
        )
        ch.end_graph()
        ch.one("zl.private.zstd", a, 120, copies=[(7, 12)])
        ch.store_rest()
        return trace

    def test_cbor_and_dot_read_the_same_graph(self):
        trace = self.build()
        from_cbor = load_trace_bytes(trace.to_cbor())
        from_dot = load_trace_bytes(trace.to_dot().encode())
        self.assertEqual(from_cbor.format, "cbor")
        self.assertEqual(from_dot.format, "dot")
        self.assertEqual(normalized(from_cbor), normalized(from_dot))
        chunk = from_cbor.chunks[0]
        self.assertEqual(chunk.codecs[1].graph, 0)
        self.assertEqual(chunk.graphs[0].type, "Parameterized")
        self.assertEqual(chunk.streams[1].raw_size, 400)
        # DOT has no content size; fixed-width streams derive it.
        self.assertEqual(from_dot.chunks[0].streams[2].raw_size, 300)

    def test_gzip_and_leading_whitespace(self):
        trace = self.build()
        self.assertEqual(
            normalized(load_trace_bytes(gzip.compress(trace.to_cbor()))),
            normalized(load_trace_bytes(b"\n  " + trace.to_dot().encode())),
        )

    def test_dot_puts_top_level_first(self):
        trace = tb.TraceBuilder()
        top = trace.chunk()
        s = top.start(5000)
        top.codec("segmenter", [s], [], ints=[(2, 20_000_000)], standard=False)
        body = trace.chunk()
        s = body.start(5000)
        body.one("zl.private.zstd", s, 900)
        body.store_rest()
        loaded = load_trace_bytes(trace.to_dot().encode())
        self.assertEqual([c.index for c in loaded.chunks], [0, 1])
        self.assertEqual(loaded.chunks[0].codecs[1].name, "segmenter")
        self.assertEqual(
            normalized(loaded), normalized(load_trace_bytes(trace.to_cbor()))
        )

    def test_failures_are_read(self):
        trace = tb.TraceBuilder()
        ch = trace.chunk()
        s = ch.start(100)
        ch.codec("zl.field_lz", [s], [], failure="Message: bad input")
        ch.one("zl.private.zstd", s, 60)
        ch.store_rest()
        cbor = load_trace_bytes(trace.to_cbor())
        dot = load_trace_bytes(trace.to_dot().encode())
        self.assertEqual(cbor.chunks[0].codecs[1].failure, "Message: bad input")
        self.assertEqual(dot.chunks[0].codecs[1].failure, "[PLACEHOLDER]")

    def test_trace_format_version_0(self):
        obj = {
            "libraryVersion": 1,
            "frameVersion": 1,
            "streams": [
                {
                    "type": 1,
                    "outputIdx": 0,
                    "eltWidth": 1,
                    "numElts": 10,
                    "cSize": 4,
                    "contentSize": 10,
                },
                {
                    "type": 1,
                    "outputIdx": 0,
                    "eltWidth": 1,
                    "numElts": 4,
                    "cSize": 4,
                    "contentSize": 4,
                },
            ],
            "codecs": [
                {"name": "zl.private.zstd", "inputStreams": [0], "outputStreams": [1]},
                {"name": "zl.store", "inputStreams": [1], "outputStreams": []},
            ],
            "graphs": [{"gType": 0, "gName": "zl.zstd", "codecIDs": [0]}],
        }
        trace = trace_from_cbor(obj)
        self.assertEqual(trace.trace_version, 0)
        names = [c.name for c in trace.chunks[0].codecs]
        self.assertEqual(names, ["zl.#start", "zl.private.zstd", "zl.store"])
        self.assertEqual(trace.chunks[0].codecs[1].graph, 0)

    def test_rejects_what_is_not_a_trace(self):
        for data in (
            b"",
            b'{"chunks": []}',
            tb.encode_cbor([1, 2]),
            tb.encode_cbor({"a": 1}),
        ):
            with self.assertRaises(TraceFormatError, msg=repr(data)):
                load_trace_bytes(data)
        huge = {
            "streams": [],
            "codecs": [{"name": "zl.#start", "outputStreams": [1 << 20]}],
        }
        with self.assertRaisesRegex(TraceFormatError, "refers to stream"):
            trace_from_cbor({"chunks": [huge]})
        dangling = {
            "streams": [],
            "codecs": [],
            "graphs": [{"gName": "g", "codecIDs": [2]}],
        }
        with self.assertRaisesRegex(TraceFormatError, "codec 2"):
            trace_from_cbor({"chunks": [dangling]})

    def test_damaged_and_oddly_shaped_input_is_a_trace_error(self):
        good = tb.serial_trace().to_cbor()
        packed = gzip.compress(good)
        damaged = packed[:12] + bytes(b ^ 0xFF for b in packed[12:40]) + packed[40:]
        chunk = {"streams": [], "codecs": [{"name": "zl.#start"}]}
        for data in (
            damaged,
            packed[:-12],  # ends too early
            tb.encode_cbor({"chunks": [{"streams": ["serial"], "codecs": []}]}),
            tb.encode_cbor({"chunks": [dict(chunk, codecs=[{"inputStreams": 5}])]}),
            bytes([0xA1, 0x66]) + b"chunks" + bytes([0x3F]),  # indefinite negative int
        ):
            with self.assertRaises(TraceFormatError, msg=repr(data[:40])):
                load_trace_bytes(data)

    def test_gzip_output_is_bounded(self):
        dot = tb.serial_trace().to_dot().encode()
        packed = gzip.compress(dot)
        self.assertEqual(len(load_trace_bytes(packed, limit=len(dot)).chunks), 1)
        with self.assertRaisesRegex(TraceFormatError, "larger than"):
            load_trace_bytes(packed, limit=len(dot) - 1)
        with self.assertRaisesRegex(TraceFormatError, "larger than"):
            load_trace_bytes(gzip.compress(b"\0" * (1 << 20)), limit=1 << 16)
        with self.assertRaisesRegex(TraceFormatError, "larger than"):
            load_trace_bytes(dot, limit=len(dot) - 1)
        # Several gzip members, as `cat a.gz b.gz` makes, are read as one file.
        half = len(dot) // 2
        joined = gzip.compress(dot[:half]) + gzip.compress(dot[half:])
        self.assertEqual(len(load_trace_bytes(joined).chunks), 1)

    def test_streams_the_tracer_never_recorded(self):
        # Strict mode, graph failed before any codec ran: no streams recorded,
        # but the failed graph's placeholder consumes stream 0.
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
                            "gName": "sddl",
                            "gType": 3,
                            "gFailureString": "Message: bad description",
                            "codecIDs": [1],
                        }
                    ],
                }
            ],
        }
        trace = trace_from_cbor(obj)
        chunk = trace.chunks[0]
        self.assertEqual(len(chunk.streams), 1)
        self.assertEqual(chunk.streams[0].raw_size, None)
        self.assertEqual(chunk.codecs[0].outputs, [0])
        self.assertEqual(chunk.codecs[1].graph, 0)

    def test_real_zli_traces_agree(self):
        cbor = load_trace(os.path.join(DATA, "sensors_chunks.cbor.gz"))
        dot = load_trace(os.path.join(DATA, "sensors_chunks.dot.gz"))
        self.assertEqual(len(cbor.chunks), 5)
        self.assertEqual(cbor.trace_version, 1)
        self.assertEqual(normalized(cbor), normalized(dot))
        serial_cbor = load_trace(os.path.join(DATA, "serial.cbor"))
        serial_dot = load_trace(os.path.join(DATA, "serial.dot"))
        self.assertEqual(normalized(serial_cbor), normalized(serial_dot))


if __name__ == "__main__":
    unittest.main()
