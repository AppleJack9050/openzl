# Copyright (c) Meta Platforms, Inc. and affiliates.

import hashlib
import json
import math
import os
import platform
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench  # noqa: E402
import fake_tools as ft  # noqa: E402

INPUT_BYTES = 102400
FAKE_VARS = ("FAKE_MODE", "FAKE_ZLI_MODE", "FAKE_ZSTD_MODE", "FAKE_LOG")
FAKE_VARS += ("FAKE_SIZES", "FAKE_TIME", "FAKE_FRAME_BYTES", "FAKE_CORE")

ZLI_CSV = (
    "srcSize,compressedSize,compressionRatio,ctimeMs,dtimeMs,iters,path\n"
    "2889011,562193,5.13882,77.1544,8.68004,3\n"
)
ZLI_OUTPUT = (
    "Chunking is not currently implemented for all profiles. Ignoring size "
    "parameter if unimplemented.\n"
    "Chunking is implemented for the following profiles: csv, parquet\n"
    "\r1 files: 2889011 -> 790453 (3.65),  291.09 MB/s  964.51 MB/s\x1b[K\n"
)
ZSTD_OUTPUT = (
    "bench 1.5.7 : input 2889011 bytes, 1 seconds, 0 KB blocks\n"
    "-3       713069 (4.052) 321.16 MB/s 1275.3 MB/s  sensors.parquet\n"
)


def gone(pid: int, wait: float = 3.0) -> bool:
    """The process has exited (a zombie waiting for its reaper counts)."""
    deadline = time.monotonic() + wait
    while True:
        try:
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state in ("Z", "X"):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)


class LevelTest(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            "3": [3],
            "1-3": [1, 2, 3],
            "fast=5": [-5],
            "fast=1-3": [-3, -2, -1],
            "all": list(range(1, 23)),
            " 19 , 3 ": [3, 19],
            "1-3,2,3": [1, 2, 3],
            "22,fast=50": [-50, 22],
            "FAST=2,All": [-2] + list(range(1, 23)),
            "7-7": [7],
        }
        for spec, levels in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(bench.parse_levels(spec), levels)

    def test_default_levels(self):
        config = bench.BenchConfig("i", "z", "s", ["parquet"])
        self.assertEqual(config.zli_levels, [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 19, 22])
        self.assertEqual(config.zstd_levels, [3, 6, 9, 12])
        self.assertEqual(
            bench.parse_levels(bench.DEFAULT_ZSTD_LEVELS, fast=False), [3, 6, 9, 12]
        )

    def test_zli_has_no_fast_levels(self):
        with self.assertRaises(ValueError) as caught:
            bench.parse_levels("1,fast=5", fast=False)
        self.assertIn("'fast=5': zli has no fast levels", str(caught.exception))
        self.assertEqual(bench.parse_levels("1,fast=5"), [-5, 1])

    def test_errors(self):
        cases = [
            ("", "no levels given"),
            ("  ", "no levels given"),
            ("1,,2", "empty item in the level list '1,,2'"),
            ("1,", "empty item"),
            (
                "0",
                "0 is zli's default level (6) and zstd's default (3); name the level",
            ),
            ("0-5", "0 is zli's default level"),
            ("-3", "use fast=K for zstd's fast levels (fast=3 is zstd --fast=3)"),
            ("fast=-3", "use fast=K for zstd's fast levels"),
            ("23", "levels go up to 22"),
            ("1-23", "level 23 is too high: levels go up to 22"),
            ("fast=51", "zstd's fast levels go from fast=1 to fast=50"),
            ("fast=0", "zstd's fast levels go from fast=1 to fast=50"),
            ("abc", "cannot read level 'abc'; use N, A-B, fast=K, fast=A-B or all"),
            ("1-", "cannot read level '1-'"),
            ("fast=", "cannot read level 'fast='"),
            ("1-2-3", "cannot read level '1-2-3'"),
            ("3.5", "cannot read level '3.5'"),
            ("9-1", "range '9-1' goes backwards; write 1-9"),
            ("fast=5-1", "range 'fast=5-1' goes backwards; write fast=1-5"),
        ]
        for spec, message in cases:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as caught:
                    bench.parse_levels(spec)
                self.assertIn(message, str(caught.exception))

    def test_format_round_trip(self):
        for spec in (
            bench.DEFAULT_ZLI_LEVELS,
            bench.DEFAULT_ZSTD_LEVELS,
            "1",
            "1,2",
            "fast=3-5,fast=1,1-22",
            "fast=2,fast=1",
            "fast=48-50,20-22",
        ):
            with self.subTest(spec=spec):
                levels = bench.parse_levels(spec)
                self.assertEqual(bench.format_levels(levels), spec)
                self.assertEqual(
                    bench.parse_levels(bench.format_levels(levels)), levels
                )

    def test_tool_levels(self):
        levels = bench.parse_levels(bench.DEFAULT_ZLI_LEVELS, fast=False)
        self.assertEqual(
            bench.zli_levels(levels), [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 19, 22]
        )
        self.assertEqual(bench.zli_levels([-5, 1, 19]), [1, 6, 19])
        self.assertEqual(bench.zli_levels([6, 6]), [6])
        self.assertEqual(bench.zstd_levels([19, -5, 1, 1]), [-5, 1, 19])
        self.assertEqual(bench.zstd_level_label(-5), "--fast=5")
        self.assertEqual(bench.zstd_level_label(3), "-3")
        self.assertEqual(bench.zli_level_label(6), "-l 6")

    def test_iterations_for(self):
        # max(ceil(S/tc), min(ceil(S/td), floor(4S/tc))), clamped to 1..10000
        self.assertEqual(bench.iterations_for(0.5, 0.1, 0.1), 5)
        self.assertEqual(bench.iterations_for(0.5, 0.1, 0.01), 20)
        self.assertEqual(bench.iterations_for(0.5, 0.1, 0.04), 13)
        self.assertEqual(bench.iterations_for(0.3, 0.1, 0.1), 3)
        self.assertEqual(bench.iterations_for(0.5, 2.0, 0.5), 1)
        self.assertEqual(bench.iterations_for(0.5, 1e-9, 1e-9), 10000)
        self.assertEqual(bench.iterations_for(0.0, 1e-9, 1e-9), 1)
        self.assertEqual(bench.iterations_for(0.05, 0.010, 0.002), 20)


class ParserTest(unittest.TestCase):
    def test_zli_csv_six_fields(self):
        parsed = bench.parse_zli_csv(ZLI_CSV, 2889011, 3)
        self.assertEqual(parsed["bytes"], 562193)
        self.assertAlmostEqual(parsed["c_speed"], 2.889011 / (0.0771544 / 3))
        self.assertAlmostEqual(parsed["d_speed"], 2.889011 / (0.00868004 / 3))
        self.assertAlmostEqual(parsed["c_seconds"], 0.0771544)
        self.assertAlmostEqual(parsed["d_seconds"], 0.00868004)

    def test_zli_csv_seven_fields(self):
        text = ZLI_CSV.rstrip("\n") + ",/data/sensors.parquet\n"
        self.assertEqual(bench.parse_zli_csv(text, 2889011, 3)["bytes"], 562193)

    def test_zli_csv_exponents(self):
        text = ZLI_CSV.splitlines()[0] + "\n100001,25781,3.87886,0.00011,2e-05,1\n"
        parsed = bench.parse_zli_csv(text, 100001, 1)
        self.assertAlmostEqual(parsed["d_seconds"], 2e-08)
        self.assertAlmostEqual(parsed["d_speed"], 0.100001 / 2e-08)
        text = ZLI_CSV.splitlines()[0] + "\n1e+06,2.5e3,400,1.5E2,20,2\n"
        parsed = bench.parse_zli_csv(text, 1000000, 2)
        self.assertEqual(parsed["bytes"], 2500)
        self.assertAlmostEqual(parsed["c_speed"], 1.0 / 0.075)

    def test_zli_csv_refusals(self):
        head = ZLI_CSV.splitlines()[0] + "\n"
        cases = [
            (
                head + "100001,25781,3.87886,0.00011,3e-05,0\n",
                100001,
                1,
                "0 iterations",
            ),
            (ZLI_CSV, 2889012, 3, "srcSize 2,889,011, but the input is 2,889,012"),
            (ZLI_CSV, 2889011, 4, "iters 3, but -n 4 was given"),
            (ZLI_CSV.replace("iters,", "runs,"), 2889011, 3, "has no iters column"),
            ("", 1, 1, "has no srcSize, compressedSize, ctimeMs, dtimeMs, iters"),
            ("garbage\nnot,a,benchmark\n", 1, 1, "has no srcSize"),
            (head, 2889011, 3, "has 0 data rows instead of 1"),
            (ZLI_CSV + ZLI_CSV.splitlines()[1] + "\n", 2889011, 3, "2 data rows"),
            (
                head + "2889011,562193,5.1,77.1,8.6,3,p,extra\n",
                2889011,
                3,
                "8 values for 7 columns",
            ),
            (head + "2889011,562193,5.1,77.1,8.6\n", 2889011, 3, "5 values for 7"),
            (head + "2889011,562193,5.1,fast,8.6,3\n", 2889011, 3, "ctimeMs='fast'"),
            (head + "2889011,562193,5.1,nan,8.6,3\n", 2889011, 3, "ctimeMs='nan'"),
            (head + "2889011,562193,5.1,0,8.6,3\n", 2889011, 3, "not positive"),
            (head + "2889011,562193,5.1,7,-1,3\n", 2889011, 3, "not positive"),
            (head + "2889011,0,5.1,7,1,3\n", 2889011, 3, "compressedSize 0"),
            (head + "2889011,12.5,5.1,7,1,3\n", 2889011, 3, "compressedSize 12.5"),
        ]
        for text, size, iterations, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    bench.parse_zli_csv(text, size, iterations)
                self.assertIn(message, str(caught.exception))

    def test_zli_output(self):
        self.assertEqual(bench.parse_zli_output(ZLI_OUTPUT), 790453)
        progress = (
            "\r0 files: 0 -> 0 (0.00)\x1b[K\r1 files: 10 -> 5 (2.00),  1 MB/s\x1b[K"
            "\r1 files: 10 -> 4 (2.50),  1.00 MB/s  2.00 MB/s\x1b[K"
        )
        self.assertEqual(bench.parse_zli_output(progress), 4)
        self.assertEqual(bench.parse_zli_output("2 files: 20 -> 7 (2.86)\n"), 7)
        self.assertIsNone(bench.parse_zli_output("Chunking is not ...\n"))
        self.assertIsNone(bench.parse_zli_output(""))

    def test_zli_csv_with_a_nul_byte(self):
        # Python 3.10's csv module raises csv.Error for it; 3.13 reads on.
        with self.assertRaises(ValueError) as caught:
            bench.parse_zli_csv("\x00garbage\nnot,a,benchmark\n", 1, 1)
        self.assertRegex(str(caught.exception), "garbled|has no srcSize")

    def test_zstd_output(self):
        self.assertEqual(
            bench.parse_zstd_output(ZSTD_OUTPUT, 2889011, 3), (713069, 321.16, 1275.3)
        )
        fast = ZSTD_OUTPUT.replace(
            "-3       713069 (4.052) 321.16 MB/s 1275.3 MB/s",
            "--5     1171693 (2.466) 586.55 MB/s 1758.3 MB/s",
        )
        self.assertEqual(
            bench.parse_zstd_output(fast, 2889011, -5), (1171693, 586.55, 1758.3)
        )
        ultra = ZSTD_OUTPUT.replace(
            "-3       713069 (4.052) 321.16 MB/s",
            "-20      632836 (4.565)   5.70 MB/s",
        )
        self.assertEqual(bench.parse_zstd_output(ultra, 2889011, 20)[:2], (632836, 5.7))
        several = ZSTD_OUTPUT + (
            "-4       700000 (4.127) 300.00 MB/s 1200.0 MB/s  sensors.parquet\r\n"
        )
        self.assertEqual(bench.parse_zstd_output(several, 2889011, 4)[0], 700000)

    def test_zstd_output_refusals(self):
        cases = [
            (ZSTD_OUTPUT, -3, "no result for level --3 (only for -3)"),
            (
                ZSTD_OUTPUT.replace("-3   ", "--3  "),
                3,
                "no result for level -3 (only for --3)",
            ),
            (ZSTD_OUTPUT, 13, "no result for level -13"),
            (ZSTD_OUTPUT.splitlines()[0] + "\n", 3, "no result for level -3"),
            (ZSTD_OUTPUT.splitlines()[1] + "\n", 3, "no benchmark header"),
            ("garbage\n", 3, "no benchmark header"),
            (
                ZSTD_OUTPUT.replace("2889011 bytes", "2889012 bytes"),
                3,
                "zstd read 2,889,012 bytes, but the input is 2,889,011",
            ),
            (
                ZSTD_OUTPUT.replace("321.16 MB/s", "0.00 MB/s"),
                3,
                "empty result",
            ),
            (
                ZSTD_OUTPUT.replace("321.16 MB/s", "3.2.1 MB/s"),
                3,
                "cannot read zstd's result line",
            ),
        ]
        for text, level, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    bench.parse_zstd_output(text, 2889011, level)
                self.assertIn(message, str(caught.exception))

    def test_short_error(self):
        self.assertEqual(
            bench._short_error("zli exited with status 1", ft.BAD_PROFILE),
            "zli exited with status 1: Invalid argument(s): / Profile not found: "
            "'nope'. See `zli list-profiles` for a list of supported profiles.",
        )
        self.assertEqual(
            bench._short_error("zli exited with status 1", ft.STRICT_FAILURE),
            "zli exited with status 1: OpenZL error string: Stream parameter invalid",
        )
        self.assertEqual(bench._short_error("no output", "\n\r \n"), "no output")
        long = bench._short_error("x", "y" * 2000)
        self.assertEqual(len(long), bench.MAX_ERROR)
        self.assertTrue(long.endswith("…"))


class FakeToolCase(unittest.TestCase):
    """Fake zli/zstd/taskset first on PATH, a fake /proc tree, a work dir."""

    fake_taskset = True
    cores = 4

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-test-")
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.bin = os.path.join(self.dir, "bin")
        self.work = os.path.join(self.dir, "work")
        os.mkdir(self.bin)
        os.mkdir(self.work)
        self.tools = ft.write_tools(self.bin, taskset=self.fake_taskset)
        self.input = os.path.join(self.dir, "sensors.parquet")
        with open(self.input, "wb") as f:
            f.write(bytes(range(256)) * (INPUT_BYTES // 256))
        self.log = os.path.join(self.dir, "calls.jsonl")
        patcher = mock.patch.dict(os.environ, ft.tool_env(self.bin, log=self.log))
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in bench.ENV_RECORDED + bench.ENV_REMOVED + FAKE_VARS[:-1]:
            if name != "FAKE_LOG":
                os.environ.pop(name, None)
        self.proc = ft.FakeProc(os.path.join(self.dir, "root"), cores=self.cores)
        self.probe = self.make_probe()

    def make_probe(self, cls=bench.CpuProbe, proc=None, **kwargs):
        proc = proc or self.proc
        allowed = kwargs.pop("allowed", range(proc.cores))
        probe = cls(root=proc.root, sleep=proc.sleep, allowed=allowed, **kwargs)
        probe.clk_tck = ft.FakeProc.HZ
        return probe

    def config(self, **overrides):
        values = dict(
            input=self.input,
            zli=self.tools["zli"],
            zstd=self.tools["zstd"],
            profiles=["parquet"],
            levels=[1, 6],
            rounds=1,
            min_time=0.0,
            core=2,
            timeout=20.0,
        )
        values.update(overrides)
        # One list for both tools, as zli and zstd would get it from the options.
        levels = values.pop("levels")
        values.setdefault("zli_levels", [level for level in levels if level > 0])
        values.setdefault("zstd_levels", levels)
        return bench.BenchConfig(**values)

    def run_bench(self, config, probe=None, **kwargs):
        return bench.run_benchmark(config, probe or self.probe, self.work, **kwargs)

    def calls(self, tool=None):
        return ft.read_log(self.log, tool)

    def series_of(self, call):
        argv = call["argv"]
        if call["tool"] == "zli":
            return "zli:" + argv[argv.index("-p") + 1]
        longs = [a for a in argv if a.startswith("--long=")]
        return "zstd:long" + longs[0][len("--long=") :] if longs else "zstd"

    def points(self, doc):
        return {p["id"]: p for p in doc["points"]}

    def size(self, tool, level, long=False):
        return ft.expected_size(tool, level, INPUT_BYTES, long)


class SeriesTest(FakeToolCase):
    def test_series_for(self):
        config = self.config(
            profiles=["csv", "parquet", "serial"],
            profile_arg=";",
            chunk_size_mb=4,
            zstd_long=27,
        )
        self.assertEqual(
            bench.series_for(config),
            [
                {
                    "id": "zli:csv",
                    "tool": "zli",
                    "label": "OpenZL -p csv",
                    "args": ["-p", "csv", "--profile-arg", ";", "--chunk-size-mb", "4"],
                    "slot": 0,
                },
                {
                    "id": "zli:parquet",
                    "tool": "zli",
                    "label": "OpenZL -p parquet",
                    "args": [
                        "-p",
                        "parquet",
                        "--profile-arg",
                        ";",
                        "--chunk-size-mb",
                        "4",
                    ],
                    "slot": 1,
                },
                {
                    "id": "zli:serial",
                    "tool": "zli",
                    "label": "OpenZL -p serial",
                    "args": [
                        "-p",
                        "serial",
                        "--profile-arg",
                        ";",
                        "--chunk-size-mb",
                        "4",
                    ],
                    "slot": 2,
                },
                {
                    "id": "zstd",
                    "tool": "zstd",
                    "label": "zstd",
                    "args": ["--single-thread"],
                    "slot": 0,
                },
                {
                    "id": "zstd:long27",
                    "tool": "zstd",
                    "label": "zstd --long=27",
                    "args": ["--single-thread", "--long=27"],
                    "slot": 1,
                },
            ],
        )
        plain = bench.series_for(self.config())
        self.assertEqual([s["id"] for s in plain], ["zli:parquet", "zstd"])
        self.assertEqual(plain[0]["args"], ["-p", "parquet"])


class ToolLookupTest(FakeToolCase):
    def test_find_tool_order(self):
        other = os.path.join(self.dir, "other")
        os.mkdir(other)
        other_zli = ft.write_tools(other, taskset=False)["zli"]
        fallback = os.path.join(self.dir, "fallback-zli")
        shutil.copy(self.tools["zli"], fallback)
        missing = os.path.join(self.dir, "missing-zli")

        os.environ["ZLI"] = other_zli
        self.assertEqual(
            bench.find_tool(self.tools["zli"], "ZLI", [fallback]), self.tools["zli"]
        )
        self.assertEqual(bench.find_tool(None, "ZLI", [fallback]), other_zli)
        del os.environ["ZLI"]
        self.assertEqual(bench.find_tool(None, "ZLI", [missing, fallback]), fallback)
        self.assertEqual(bench.find_tool(None, "ZLI", [missing]), self.tools["zli"])
        self.assertEqual(bench.find_tool("zstd", "ZSTD", []), self.tools["zstd"])
        os.environ["ZLI"] = ""
        self.assertEqual(bench.find_tool(None, "ZLI", []), self.tools["zli"])

    def test_find_tool_relative_path_is_made_absolute(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.bin)
        self.assertEqual(bench.find_tool("./zli", "ZLI", []), self.tools["zli"])
        self.assertEqual(bench.find_tool("zli", "ZLI", []), self.tools["zli"])

    def test_find_tool_errors(self):
        plain = os.path.join(self.dir, "plain")
        with open(plain, "w") as f:
            f.write("not a program\n")
        cases = [
            ((plain, "ZLI", []), f"--zli {plain}: not an executable file"),
            ((self.dir, "ZLI", []), f"--zli {self.dir}: not an executable file"),
            (("no-such-zli", "ZLI", []), "--zli no-such-zli: not an executable file"),
            ((None, "ZLI", [plain]), f"{plain} is not executable; give --zli PATH"),
        ]
        for args, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(bench.BenchError) as caught:
                    bench.find_tool(*args)
                self.assertEqual(str(caught.exception), message)
        os.environ["ZLI"] = plain
        with self.assertRaises(bench.BenchError) as caught:
            bench.find_tool(None, "ZLI", [])
        self.assertEqual(str(caught.exception), f"$ZLI {plain}: not an executable file")
        del os.environ["ZLI"]
        os.environ["PATH"] = self.work
        with self.assertRaises(bench.BenchError) as caught:
            bench.find_tool(None, "ZLI", [os.path.join(self.dir, "nothing")])
        self.assertEqual(str(caught.exception), "zli not found; give --zli PATH")
        with self.assertRaises(bench.BenchError) as caught:
            bench.find_tool(None, "CODEC_REVIEWER_ZSTD", [])
        self.assertEqual(str(caught.exception), "zstd not found; give --zstd PATH")

    def test_zli_profiles(self):
        self.assertEqual(bench.zli_profiles(self.tools["zli"]), list(ft.PROFILES))
        for mode, message in (
            ("fail", "zli exited with status 1: Error: unknown command"),
            ("garbage", "zli listed no profiles: no profiles here"),
        ):
            with self.subTest(mode=mode):
                os.environ["FAKE_ZLI_MODE"] = mode
                with self.assertRaises(bench.BenchError) as caught:
                    bench.zli_profiles(self.tools["zli"])
                self.assertIn(message, str(caught.exception))
        with self.assertRaises(bench.BenchError):
            bench.zli_profiles(os.path.join(self.dir, "no-zli"))

    def test_profile_listing_format(self):
        listing = (
            "Available profiles:\n"
            "  -| csv\t= CSV. Pass optional non-comma separator with "
            "--profile-arg <char>.\n"
            "  -| parquet\t= Parquet in the canonical format (no compression, "
            "plain encoding)\n"
            "  -| numeric-ml-selector-64\t= 64 bit numeric data (Placeholder)\n"
            "  -| serial\t= Serial data (aka raw bytes)\n\n"
        )
        self.assertEqual(
            bench._PROFILE_LINE.findall(listing),
            ["csv", "parquet", "numeric-ml-selector-64", "serial"],
        )

    def test_versions(self):
        self.assertEqual(bench.zstd_version(self.tools["zstd"]), "1.5.7")
        self.assertEqual(bench.zli_version(self.tools["zli"]), ft.ZLI_VERSION)
        os.environ["FAKE_MODE"] = "badversion"
        with self.assertRaises(bench.BenchError) as caught:
            bench.zstd_version(self.tools["zstd"])
        self.assertIn(
            "does not look like the zstd command line: gzip 1.12", str(caught.exception)
        )
        self.assertEqual(bench.zli_version(self.tools["zli"]), "zli 9.9")
        missing = os.path.join(self.dir, "none")
        self.assertEqual(bench.zli_version(missing), "")
        with self.assertRaises(bench.BenchError):
            bench.zstd_version(missing)

    def test_tool_info(self):
        with open(bench.__file__, "rb") as f:
            source = f.read()
        info = bench.tool_info(bench.__file__, "v1", note="-O3")
        self.assertEqual(
            info,
            {
                "path": "tools/codec_reviewer/bench.py",
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
                "version": "v1",
                "build_dir": None,
                "note": "-O3",
            },
        )

        build = os.path.join(self.dir, "cachedObjs", "a847d4ccf764f93f", "zli")
        os.makedirs(os.path.dirname(build))
        shutil.copy(self.tools["zli"], build)
        os.environ["HOME"] = self.dir
        info = bench.tool_info(build, ft.ZLI_VERSION, note="")
        self.assertEqual(info["path"], "~/cachedObjs/a847d4ccf764f93f/zli")
        self.assertEqual(info["build_dir"], "a847d4ccf764f93f")
        self.assertEqual(info["note"], "")
        self.assertEqual(bench._short_display(info["path"]), "~/cachedObjs/a847…/zli")

        zstd = bench.tool_info(self.tools["zstd"], "1.5.7")
        self.assertEqual(set(zstd), {"path", "sha256", "bytes", "version"})
        self.assertEqual(zstd["path"], "~/bin/zstd")
        os.environ["HOME"] = os.path.join(self.dir, "elsewhere")
        self.assertEqual(bench.display_path(self.tools["zstd"]), self.tools["zstd"])
        os.environ["HOME"] = self.dir + os.sep
        self.assertEqual(bench.display_path(self.tools["zstd"]), "~/bin/zstd")


class CpuProbeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="bench-probe-")
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def probe(self, proc, allowed=range(4)):
        return bench.CpuProbe(root=proc.root, sleep=proc.sleep, allowed=allowed)

    def test_ticks(self):
        proc = ft.FakeProc(self.root)
        ticks = self.probe(proc).ticks()
        self.assertEqual(sorted(ticks), [0, 1, 2, 3])
        # busy = total - (idle + iowait); the guest fields are not added.
        self.assertEqual(ticks[0], (1000, 50000))
        self.assertEqual(ticks[3], (4000, 50000))

    def test_busy_percent(self):
        proc = ft.FakeProc(self.root, loads=[{2: 50, 3: 100}])
        probe = self.probe(proc)
        self.assertEqual(probe.busy_percent(2), 50.0)
        self.assertEqual(probe.busy_percent(3, 1.0), 100.0)
        self.assertEqual(probe.busy_percent(1), 0.0)
        self.assertEqual(proc.sleeps, [0.3, 1.0, 0.3])
        with self.assertRaises(bench.BenchError):
            probe.busy_percent(9)
        # A clock that did not move reads as idle rather than dividing by zero.
        still = bench.CpuProbe(root=proc.root, sleep=lambda s: None, allowed=[0])
        self.assertEqual(still.busy_percent(0), 0.0)

    def test_quietest(self):
        cases = [
            ({0: 0, 1: 0, 2: 10, 3: 0}, range(4), (1, 0.0)),
            ({0: 0, 1: 5, 2: 5, 3: 5}, range(4), (0, 0.0)),
            ({0: 50, 1: 50, 2: 50, 3: 50}, range(4), (1, 50.0)),
            ({0: 0, 1: 0, 2: 20, 3: 30}, {2, 3}, (2, 20.0)),
            ({0: 0, 1: 0, 2: 0, 3: 0}, {0, 3}, (3, 0.0)),
            ({}, {0}, (0, 0.0)),
            ({1: 90}, {1, 7}, (1, 90.0)),
        ]
        for load, allowed, expected in cases:
            with self.subTest(load=load, allowed=allowed):
                proc = ft.FakeProc(tempfile.mkdtemp(dir=self.root), loads=[load])
                self.assertEqual(self.probe(proc, allowed).quietest(), expected)
                self.assertEqual(proc.sleeps, [0.4])
        proc = ft.FakeProc(tempfile.mkdtemp(dir=self.root))
        with self.assertRaises(bench.BenchError):
            self.probe(proc, {8, 9}).quietest()

    def test_missing_stat(self):
        probe = bench.CpuProbe(root=self.root, sleep=lambda s: None)
        with self.assertRaises(bench.BenchError):
            probe.ticks()
        self.assertIsNone(probe.cpu_model())
        self.assertIsNone(probe.governor(0))

    def test_model_and_governor(self):
        probe = self.probe(ft.FakeProc(self.root))
        self.assertEqual(probe.cpu_model(), "Fake CPU 9000 @ 3.00GHz")
        self.assertEqual(probe.governor(1), "performance")
        self.assertIsNone(probe.governor(9))
        bare = ft.FakeProc(tempfile.mkdtemp(dir=self.root), model=None, governor=None)
        self.assertIsNone(self.probe(bare).cpu_model())
        self.assertIsNone(self.probe(bare).governor(0))

    def test_defaults(self):
        probe = bench.CpuProbe()
        self.assertEqual(probe.allowed, set(os.sched_getaffinity(0)))
        self.assertEqual(probe.clk_tck, os.sysconf("SC_CLK_TCK"))
        self.assertEqual(probe.root, "/")
        self.assertTrue(set(probe.ticks()) >= probe.allowed)


class CheckSetupTest(FakeToolCase):
    def test_good_config(self):
        bench.check_setup(self.config(), self.probe)
        bench.check_setup(
            self.config(
                profiles=["csv", "parquet", "le-u64"],
                core="auto",
                zstd_long=27,
                chunk_size_mb=1,
            ),
            self.probe,
        )
        bench.check_setup(self.config(core="none", min_time=0), self.probe)

    def test_errors(self):
        plain = os.path.join(self.dir, "plain")
        with open(plain, "w") as f:
            f.write("text\n")
        self.empty = os.path.join(self.dir, "empty")
        open(self.empty, "wb").close()
        cases = [
            (dict(input=self.work), "is a directory; --bench needs a file"),
            (dict(input=self.empty), "is empty; there is nothing to measure"),
            (dict(input=os.path.join(self.dir, "nope")), "nope: no such file"),
            (dict(input="/dev/null"), "/dev/null is not a regular file"),
            (
                dict(zli=plain),
                f"zli {plain} is not an executable file; give --zli PATH",
            ),
            (dict(zstd=plain), "give --zstd PATH"),
            (
                dict(profiles=["nope"]),
                "zli has no profile 'nope'; it has: csv, json, le-u64, parquet, serial",
            ),
            (dict(profiles=[]), "give 1 to 3 zli profiles with -p"),
            (dict(profiles=["csv", "json", "serial", "parquet"]), "give 1 to 3"),
            (dict(profiles=["csv", "csv"]), "profile csv is given twice"),
            (dict(profiles=["Parquet"]), "'Parquet' is not a zli profile name"),
            (dict(profiles=["a b"]), "not a zli profile name"),
            (dict(compressor="x.zlc"), "trained compressors"),
            (dict(zstd_levels=[]), "no zstd levels to measure; give --zstd-levels"),
            (dict(zstd_levels=[0]), "zstd level 0 is out of range"),
            (dict(zstd_levels=[23]), "zstd level 23 is out of range"),
            (dict(zstd_levels=[-51]), "zstd level -51 is out of range"),
            (dict(zstd_levels=[True]), "out of range"),
            (dict(zli_levels=[-1]), "zli level -1 is out of range"),
            (dict(zli_levels=[0]), "zli level 0 is out of range"),
            (dict(zli_levels=[23]), "zli level 23 is out of range"),
            (dict(rounds=0), "--rounds must be between 1 and 20"),
            (dict(rounds=21), "--rounds must be between 1 and 20"),
            (dict(rounds=2.0), "--rounds"),
            (dict(min_time=-1), "--min-time must be between 0 and 60 seconds"),
            (dict(min_time=61), "--min-time"),
            (dict(min_time=math.nan), "--min-time"),
            (dict(timeout=0), "--timeout must be a positive number"),
            (dict(timeout=math.inf), "--timeout"),
            (dict(chunk_size_mb=0), "--chunk-size-mb must be a positive"),
            (dict(zstd_long=9), "--zstd-long takes a window log from 10 to 31"),
            (dict(zstd_long=32), "--zstd-long"),
            (
                dict(core=7),
                "core 7 is not available to this process; use one of 0-3, auto or none",
            ),
            (dict(core=-1), "core -1 is not available"),
            (dict(core="fast"), "--core takes a core number, auto or none"),
            (dict(core=True), "--core takes"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(bench.BenchError) as caught:
                    bench.check_setup(self.config(**overrides), self.probe)
                self.assertIn(message, str(caught.exception))

    def test_tool_errors(self):
        os.environ["FAKE_ZSTD_MODE"] = "badversion"
        with self.assertRaises(bench.BenchError) as caught:
            bench.check_setup(self.config(), self.probe)
        self.assertIn("does not look like the zstd command line", str(caught.exception))
        os.environ["FAKE_ZLI_MODE"] = "fail"
        with self.assertRaises(bench.BenchError) as caught:
            bench.check_setup(self.config(), self.probe)
        self.assertIn("cannot list zli's profiles", str(caught.exception))

    @unittest.skipIf(os.geteuid() == 0, "root can read any file")
    def test_unreadable_input(self):
        os.chmod(self.input, 0)
        self.addCleanup(os.chmod, self.input, stat.S_IRUSR | stat.S_IWUSR)
        with self.assertRaises(bench.BenchError) as caught:
            bench.check_setup(self.config(), self.probe)
        self.assertIn("cannot read", str(caught.exception))

    def test_nothing_measured(self):
        bench.check_setup(self.config(), self.probe)
        self.assertEqual({c["argv"][0] for c in self.calls()}, {"list-profiles"})
        self.assertEqual(self.proc.sleeps, [])


class ScriptedProbe(bench.CpuProbe):
    """Pre-run checks answer from ``checks`` in turn (then 0); polls ``poll``."""

    def __init__(self, *args, checks=(), poll=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks = list(checks)
        self.poll = poll
        self.busy_calls = []

    def busy_percent(self, core, interval=0.3):
        self.busy_calls.append((core, interval))
        if interval == bench.BUSY_INTERVAL:
            return self.checks.pop(0) if self.checks else 0.0
        return self.poll


class ForeignProbe(bench.CpuProbe):
    """Every tick reading finds core 2 busier by 0.5 s of someone else's work."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = 0

    def busy_percent(self, core, interval=0.3):
        return 0.0

    def ticks(self):
        self.extra += 50
        ticks = super().ticks()
        busy, total = ticks[2]
        ticks[2] = (busy + self.extra, total + self.extra)
        return ticks


class SpawnTest(unittest.TestCase):
    def test_output_is_bounded(self):
        code, out, err, timed_out = bench._spawn(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x' * (3 << 20) + 'end')",
            ],
            30,
        )
        self.assertEqual((code, timed_out), (0, False))
        self.assertLess(len(out), 2 * bench.MAX_CAPTURE + 100)
        self.assertTrue(out.startswith("x" * 1000))
        self.assertTrue(out.endswith("end"))
        self.assertIn("bytes left out]", out)
        self.assertEqual(err, "")

    def test_names_that_are_not_text(self):
        self.assertEqual(
            bench.display_name("/x/caf\udce9.parquet"), "caf\ufffd.parquet"
        )
        self.assertEqual(bench.display_name("dir/plain.bin"), "plain.bin")


class RunBenchmarkTest(FakeToolCase):
    def test_document(self):
        messages = []
        config = self.config(
            profiles=["parquet", "serial"],
            levels=[-1, 1, 6],
            chunk_size_mb=1,
            zstd_long=27,
            rounds=2,
            build_note="built with -O3",
        )
        doc = self.run_bench(config, progress=messages.append)
        self.assertEqual(
            list(doc),
            [
                "format",
                "version",
                "complete",
                "stopped",
                "created",
                "elapsed_s",
                "input",
                "machine",
                "tools",
                "settings",
                "series",
                "points",
                "trace_link",
                "notes",
            ],
        )
        self.assertEqual(doc["format"], "codec_reviewer.benchmark")
        self.assertEqual(doc["version"], 1)
        self.assertIs(doc["complete"], True)
        self.assertIsNone(doc["stopped"])
        self.assertRegex(doc["created"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIsInstance(doc["elapsed_s"], float)
        self.assertGreaterEqual(doc["elapsed_s"], 0)
        with open(self.input, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            doc["input"],
            {"name": "sensors.parquet", "bytes": INPUT_BYTES, "sha256": sha},
        )
        self.assertEqual(
            doc["machine"],
            {
                "cpu": "Fake CPU 9000 @ 3.00GHz",
                "logical_cpus": os.cpu_count(),
                "kernel": platform.release(),
                "python": platform.python_version(),
                "governor": "performance",
                "core": 2,
                "core_choice": "fixed",
                "pin": "taskset",
            },
        )
        zli_path = bench.display_path(self.tools["zli"])
        zstd_path = bench.display_path(self.tools["zstd"])
        self.assertEqual(doc["tools"]["zli"]["version"], ft.ZLI_VERSION)
        self.assertEqual(doc["tools"]["zli"]["note"], "built with -O3")
        self.assertIsNone(doc["tools"]["zli"]["build_dir"])
        self.assertEqual(doc["tools"]["zli"]["path"], zli_path)
        self.assertEqual(doc["tools"]["zstd"]["version"], "1.5.7")
        self.assertEqual(doc["tools"]["zstd"]["path"], zstd_path)
        self.assertEqual(
            doc["settings"],
            {
                "levels": "zli 1,6; zstd fast=1,1,6",
                "rounds": 2,
                "min_time_s": 0.0,
                "quick": False,
                "timeout_s": 20.0,
                "env": {},
            },
        )
        self.assertEqual(doc["series"], bench.series_for(config))
        self.assertEqual(
            [p["id"] for p in doc["points"]],
            [
                "zstd/-1",
                "zstd:long27/-1",
                "zli:parquet/1",
                "zli:serial/1",
                "zstd/1",
                "zstd:long27/1",
                "zli:parquet/6",
                "zli:serial/6",
                "zstd/6",
                "zstd:long27/6",
            ],
        )
        labels = {"zli": "-l {}", "zstd": "-{}"}
        for point in doc["points"]:
            with self.subTest(point=point["id"]):
                series, level = point["id"].split("/")
                tool = series.split(":")[0]
                self.assertEqual(point["series"], series)
                self.assertEqual(point["level"], int(level))
                if point["level"] < 0:
                    self.assertEqual(point["level_label"], "--fast=1")
                else:
                    self.assertEqual(point["level_label"], labels[tool].format(level))
                self.assertEqual(point["status"], "ok")
                self.assertIsNone(point["error"])
                self.assertEqual(
                    point["bytes"],
                    self.size(tool, point["level"], series.endswith("long27")),
                )
                self.assertEqual(len(point["c_samples"]), 2)
                self.assertEqual(len(point["d_samples"]), 2)
                self.assertEqual(point["c_speed"], max(point["c_samples"]))
                self.assertEqual(point["d_speed"], max(point["d_samples"]))
                # With --min-time 0, zli runs one iteration of a few fake ms.
                short = ["short"] if tool == "zli" else []
                self.assertEqual(point["flags"], {"c": short, "d": short, "both": []})
        points = self.points(doc)
        self.assertEqual(points["zli:parquet/6"]["c_speed"], 0.1024 / 0.004)
        self.assertEqual(points["zli:parquet/6"]["d_speed"], 0.1024 / 0.0002)
        self.assertEqual(points["zstd/1"]["c_speed"], 68.27)
        self.assertEqual(
            points["zli:serial/6"]["command"],
            ["taskset", "-c", "2", zli_path, "benchmark", "sensors.parquet"]
            + ["-p", "serial", "--chunk-size-mb", "1", "-l", "6", "-n", "1"]
            + ["-v", "1", "--output-csv", "<tmp>/zli.csv"],
        )
        self.assertEqual(
            points["zstd:long27/-1"]["command"],
            ["taskset", "-c", "2", zstd_path, "--single-thread", "-q", "--long=27"]
            + ["--fast=1", "-b", "-i1", "--", "sensors.parquet"],
        )
        self.assertEqual(
            doc["trace_link"],
            {
                "state": "none",
                "series": None,
                "point": None,
                "frame_bytes": None,
                "stream_bytes": None,
                "given_input_bytes": None,
                "given_stream_bytes": None,
                "verified": None,
            },
        )
        self.assertEqual(doc["notes"], [])
        text = json.dumps(doc, allow_nan=False)
        self.assertNotIn(self.input, text)
        self.assertNotIn(self.work, text)
        self.assertEqual(os.listdir(self.work), [])

        self.assertRegex(
            messages[0],
            r"^Benchmark: 10 points x 2 rounds on core 2 \(fixed\); zli \S+/zli, "
            r"zstd 1\.5\.7$",
        )
        self.assertEqual(len(messages), 21)
        self.assertRegex(
            messages[3],
            r"^\[ 3/20\] OpenZL -p parquet -l 1 +[\d,]+ +\d+\.\d\dx +[\d.]+ MB/s"
            r" +[\d.]+ MB/s$",
        )
        self.assertIn(f"{self.size('zli', 1):,}", messages[3])
        self.assertRegex(messages[20], r"^\[20/20\] zstd --long=27 +-6 ")

    def test_schedule_order(self):
        config = self.config(
            profiles=["parquet", "serial"], levels=[-1, 1, 6], zstd_long=27, rounds=2
        )
        self.run_bench(config)
        visits = [(self.series_of(c), c["level"]) for c in self.calls() if c["key"]]
        level_order = [
            (-1, ["zstd", "zstd:long27"]),
            (1, ["zli:parquet", "zli:serial", "zstd", "zstd:long27"]),
            (6, ["zli:parquet", "zli:serial", "zstd", "zstd:long27"]),
        ]
        round_one, round_two = [], []
        for level, series in level_order:
            for s in series:
                # zli's first visit is a calibration run followed by the sample.
                round_one += [(s, level)] * (2 if s.startswith("zli") else 1)
                round_two.append((s, level))
        self.assertEqual(visits, round_one + round_two)

    def test_iterations_from_calibration(self):
        os.environ["FAKE_TIME"] = json.dumps([4, 2])
        self.run_bench(self.config(levels=[6], rounds=2, min_time=0.05))
        runs = [c["argv"] for c in self.calls("zli") if c["key"]]
        counts = [a[a.index("-n") + 1] for a in runs]
        # calibration: 4 ms / 2 ms -> max(13, min(25, 50)) iterations
        self.assertEqual(counts, ["1", "25", "25"])
        zstd = [c["argv"] for c in self.calls("zstd")]
        self.assertEqual([a[-3] for a in zstd], ["-i1", "-i1"])

    def test_quick(self):
        doc = self.run_bench(self.config(levels=[-2, 6], rounds=3, quick=True))
        runs = self.calls()
        self.assertEqual(len(runs), 3)
        zli = [c["argv"] for c in runs if c["tool"] == "zli"]
        self.assertEqual(len(zli), 1)
        self.assertEqual(zli[0][zli[0].index("-n") + 1], "1")
        self.assertEqual(
            [c["argv"][-3] for c in runs if c["tool"] == "zstd"], ["-i0", "-i0"]
        )
        self.assertEqual(doc["settings"]["rounds"], 1)
        self.assertIs(doc["settings"]["quick"], True)
        for point in doc["points"]:
            self.assertEqual(point["status"], "ok")
            self.assertEqual(point["flags"]["both"], ["quick"])
            self.assertEqual(len(point["c_samples"]), 1)
        # -i0 is one pass (102,400 bytes at 512 MB/s is well under 0.05 s), but
        # "quick" already says so; "short" is not added on top.
        self.assertEqual(self.points(doc)["zstd/6"]["flags"]["d"], [])

    def test_zstd_commands(self):
        doc = self.run_bench(
            self.config(levels=[-5, 3, 20], zstd_long=27, min_time=1.5)
        )
        argvs = {(self.series_of(c), c["level"]): c["argv"] for c in self.calls("zstd")}
        tail = ["--", self.input]
        self.assertEqual(
            argvs[("zstd", -5)],
            ["--single-thread", "-q", "--fast=5", "-b", "-i2"] + tail,
        )
        self.assertEqual(
            argvs[("zstd", 3)], ["--single-thread", "-q", "-b3", "-e3", "-i2"] + tail
        )
        self.assertEqual(
            argvs[("zstd", 20)],
            ["--single-thread", "-q", "--ultra", "-b20", "-e20", "-i2"] + tail,
        )
        self.assertEqual(
            argvs[("zstd:long27", 3)],
            ["--single-thread", "-q", "--long=27", "-b3", "-e3", "-i2"] + tail,
        )
        self.assertEqual(
            argvs[("zstd:long27", 20)],
            ["--single-thread", "-q", "--ultra", "--long=27", "-b20", "-e20", "-i2"]
            + tail,
        )
        points = self.points(doc)
        self.assertEqual(points["zstd/-5"]["level_label"], "--fast=5")
        self.assertEqual(points["zstd/20"]["level_label"], "-20")
        self.assertEqual(points["zstd/20"]["bytes"], self.size("zstd", 20))
        self.assertEqual(points["zstd:long27/20"]["bytes"], self.size("zstd", 20, True))
        self.assertTrue(all(p["status"] == "ok" for p in doc["points"]))
        self.assertEqual(points["zstd/3"]["flags"]["c"], [])

    def test_best_of_rounds_and_axis_flags(self):
        os.environ["FAKE_TIME"] = json.dumps([[10, 2], [10, 2], [5, 3], [8, 1]])
        # Pre-run checks: zli calibration, zli r1, zstd r1, zli r2 (busy), ...
        probe = self.make_probe(ScriptedProbe, checks=[0, 0, 0, 50], poll=50.0)
        doc = self.run_bench(
            self.config(levels=[6], rounds=3, min_time=0.05), probe=probe
        )
        point = self.points(doc)["zli:parquet/6"]
        self.assertEqual(point["status"], "ok")
        self.assertEqual(point["c_samples"], [10.24, 20.48, 12.8])
        self.assertEqual(point["d_samples"], [51.2, 34.1333, 102.4])
        self.assertEqual(point["c_speed"], 20.48)
        self.assertEqual(point["d_speed"], 102.4)
        # The fastest compression ran while the core was busy; the fastest
        # decompression timed 20 x 1 ms, under 0.05 s.
        self.assertEqual(
            point["flags"], {"c": ["contended"], "d": ["short"], "both": []}
        )
        runs = [c["argv"] for c in self.calls("zli")]
        self.assertEqual([a[a.index("-n") + 1] for a in runs], ["1", "20", "20", "20"])
        polls = [call for call in probe.busy_calls if call[1] != bench.BUSY_INTERVAL]
        self.assertEqual(polls, [(2, 1.0)] * bench.WAIT_POLLS)
        self.assertEqual(self.points(doc)["zstd/6"]["flags"]["c"], [])

    def test_sizes_differ(self):
        os.environ["FAKE_ZLI_MODE"] = "sizes_differ"
        doc = self.run_bench(self.config(levels=[6], rounds=3))
        point = self.points(doc)["zli:parquet/6"]
        self.assertEqual(point["status"], "failed")
        self.assertEqual(
            point["error"],
            f"sizes differ between runs ({self.size('zli', 6):,} and "
            f"{self.size('zli', 6) + 1:,} bytes)",
        )
        self.assertIsNone(point["bytes"])
        self.assertIsNone(point["c_speed"])
        self.assertEqual(point["c_samples"], [])
        # Not retried after failing: calibration and one sample only.
        self.assertEqual(len(self.calls("zli")), 2)
        self.assertEqual(self.points(doc)["zstd/6"]["status"], "ok")
        self.assertIs(doc["complete"], True)

    def test_sizes_differ_between_rounds_of_zstd(self):
        os.environ["FAKE_ZSTD_MODE"] = "sizes_differ@6#1"
        doc = self.run_bench(self.config(levels=[6], rounds=3))
        point = self.points(doc)["zstd/6"]
        self.assertEqual(point["status"], "failed")
        self.assertIn("sizes differ between runs", point["error"])
        self.assertEqual(len(self.calls("zstd")), 2)

    def test_failed_point_is_not_retried(self):
        os.environ["FAKE_ZLI_MODE"] = "fail@6"
        doc = self.run_bench(self.config(levels=[1, 6, 9], rounds=3))
        points = self.points(doc)
        self.assertEqual(points["zli:parquet/6"]["status"], "failed")
        self.assertEqual(
            points["zli:parquet/6"]["error"],
            "zli exited with status 1: Invalid argument(s): / Profile not found: "
            "'nope'. See `zli list-profiles` for a list of supported profiles.",
        )
        self.assertEqual(points["zli:parquet/6"]["flags"]["c"], [])
        self.assertTrue(points["zli:parquet/6"]["command"])
        self.assertEqual(points["zli:parquet/9"]["status"], "ok")
        self.assertEqual(len(points["zli:parquet/9"]["c_samples"]), 3)
        levels = [c["level"] for c in self.calls("zli")]
        self.assertEqual(levels.count(6), 1)
        self.assertEqual(levels.count(9), 4)
        self.assertIs(doc["complete"], True)

    def test_timeout_kills_group_and_skips_higher_levels(self):
        os.environ["FAKE_ZLI_MODE"] = "hang@6"
        start = time.monotonic()
        doc = self.run_bench(self.config(levels=[1, 6, 9], rounds=2, timeout=0.5))
        self.assertLess(time.monotonic() - start, 10)
        points = self.points(doc)
        self.assertEqual(points["zli:parquet/1"]["status"], "ok")
        self.assertEqual(points["zli:parquet/6"]["status"], "timeout")
        self.assertEqual(points["zli:parquet/6"]["error"], "timed out after 0.5 s")
        self.assertEqual(points["zli:parquet/9"]["status"], "skipped")
        self.assertEqual(
            points["zli:parquet/9"]["error"], "skipped after a lower level timed out"
        )
        self.assertEqual(points["zli:parquet/9"]["command"], [])
        for level in (1, 6, 9):
            self.assertEqual(points[f"zstd/{level}"]["status"], "ok")
            self.assertEqual(len(points[f"zstd/{level}"]["c_samples"]), 2)
        self.assertIs(doc["complete"], False)
        self.assertIsNone(doc["stopped"])
        calls = self.calls()
        self.assertEqual(
            [c["level"] for c in calls if c["tool"] == "zli"], [1, 1, 6, 1]
        )
        hung = [c for c in calls if c["tool"] == "zli" and c["level"] == 6][0]
        (helper,) = [c for c in calls if c["tool"] == "helper"]
        # The helper shares the hung zli's process group and died with it.
        self.assertTrue(gone(hung["pid"]))
        self.assertTrue(gone(helper["pid"]))

    def test_timeout_in_a_later_round_keeps_measured_levels(self):
        os.environ["FAKE_ZLI_MODE"] = "hang@6#2"
        doc = self.run_bench(self.config(levels=[1, 6, 9], rounds=3, timeout=0.5))
        points = self.points(doc)
        self.assertEqual(points["zli:parquet/6"]["status"], "timeout")
        self.assertEqual(points["zli:parquet/6"]["c_samples"], [])
        nine = points["zli:parquet/9"]
        self.assertEqual(nine["status"], "ok")
        self.assertEqual(len(nine["c_samples"]), 1)
        self.assertEqual(len(points["zli:parquet/1"]["c_samples"]), 3)
        self.assertEqual(
            [c["level"] for c in self.calls("zli")].count(9), 2
        )  # calibration + round 1
        self.assertIs(doc["complete"], True)

    def test_interrupt_before_every_point_ran(self):
        doc = self.run_bench(
            self.config(levels=[1, 6], rounds=2), progress=self.interrupt_on(3)
        )
        points = self.points(doc)
        self.assertIs(doc["complete"], False)
        self.assertEqual(doc["stopped"], "interrupted")
        self.assertEqual(points["zli:parquet/1"]["status"], "ok")
        self.assertEqual(len(points["zli:parquet/1"]["c_samples"]), 1)
        self.assertEqual(points["zstd/1"]["status"], "ok")
        for point_id in ("zli:parquet/6", "zstd/6"):
            self.assertEqual(points[point_id]["status"], "skipped")
            self.assertEqual(points[point_id]["error"], "interrupted")
            self.assertIsNone(points[point_id]["bytes"])
        self.assertEqual(os.listdir(self.work), [])
        json.dumps(doc, allow_nan=False)

    def test_interrupt_in_a_later_round(self):
        doc = self.run_bench(
            self.config(levels=[1, 6], rounds=2), progress=self.interrupt_on(7)
        )
        self.assertIs(doc["complete"], False)
        self.assertEqual(doc["stopped"], "interrupted")
        samples = {p["id"]: len(p["c_samples"]) for p in doc["points"]}
        self.assertEqual(
            samples,
            {"zli:parquet/1": 2, "zstd/1": 2, "zli:parquet/6": 1, "zstd/6": 1},
        )
        self.assertTrue(all(p["status"] == "ok" for p in doc["points"]))

    def interrupt_on(self, n):
        seen = []

        def progress(message):
            seen.append(message)
            if len(seen) == n:
                raise KeyboardInterrupt

        return progress

    def test_interrupt_kills_running_child(self):
        os.environ["FAKE_ZLI_MODE"] = "hang@6"
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        stop = threading.Event()

        def interrupt_when_hung():
            while not stop.wait(0.02):
                if any(c["tool"] == "helper" for c in self.calls()):
                    time.sleep(0.1)
                    os.kill(os.getpid(), signal.SIGINT)
                    return

        watcher = threading.Thread(target=interrupt_when_hung, daemon=True)
        watcher.start()
        self.addCleanup(watcher.join)
        self.addCleanup(stop.set)
        start = time.monotonic()
        doc = self.run_bench(self.config(levels=[1, 6], quick=True, timeout=60))
        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual(doc["stopped"], "interrupted")
        points = self.points(doc)
        self.assertEqual(points["zli:parquet/1"]["status"], "ok")
        self.assertEqual(points["zstd/1"]["status"], "ok")
        self.assertEqual(points["zli:parquet/6"]["status"], "skipped")
        self.assertEqual(points["zli:parquet/6"]["error"], "interrupted")
        self.assertEqual(points["zstd/6"]["status"], "skipped")
        calls = self.calls()
        hung = [c for c in calls if c["tool"] == "zli" and c["level"] == 6][0]
        (helper,) = [c for c in calls if c["tool"] == "helper"]
        self.assertTrue(gone(hung["pid"]))
        self.assertTrue(gone(helper["pid"]))

    def test_zli_failure_modes(self):
        cases = [
            ("fail", "zli exited with status 1: Invalid argument(s): / Profile not"),
            ("strict", "zli exited with status 1: OpenZL error string: Stream"),
            # Python 3.10's csv module refuses the NUL byte; 3.13 reads on.
            ("garbage", ("zli's CSV has no srcSize", "zli's CSV is garbled")),
            (
                "nocsv",
                # zli's "Chunking ..." notices are left out of the message.
                "zli wrote no CSV file: 1 files: 102400 -> ",
            ),
            ("n0", "zli ran 0 iterations"),
            (
                "wrongsize",
                "zli printed 26,257 compressed bytes but its CSV says 26,256",
            ),
            ("wrongsrc", "srcSize 102,401, but the input is 102,400 bytes"),
            ("wrongiters", "iters 2, but -n 1 was given"),
        ]
        for mode, message in cases:
            with self.subTest(mode=mode):
                os.environ["FAKE_ZLI_MODE"] = mode
                doc = self.run_bench(
                    self.config(levels=[6], quick=True, chunk_size_mb=1)
                )
                point = self.points(doc)["zli:parquet/6"]
                self.assertEqual(point["status"], "failed")
                messages = (message,) if isinstance(message, str) else message
                self.assertTrue(
                    any(m in point["error"] for m in messages), point["error"]
                )
                self.assertLessEqual(len(point["error"]), bench.MAX_ERROR)
                self.assertEqual(self.points(doc)["zstd/6"]["status"], "ok")
                self.assertIs(doc["complete"], True)

    def test_zli_accepted_variants(self):
        os.environ["FAKE_ZLI_MODE"] = "csv7"
        doc = self.run_bench(self.config(levels=[6], quick=True))
        self.assertEqual(self.points(doc)["zli:parquet/6"]["status"], "ok")
        os.environ["FAKE_ZLI_MODE"] = "warnings"
        doc = self.run_bench(self.config(levels=[6], quick=True))
        point = self.points(doc)["zli:parquet/6"]
        self.assertEqual(point["status"], "ok")
        self.assertEqual(point["flags"]["both"], ["quick", "fallback"])
        self.assertEqual(self.points(doc)["zstd/6"]["flags"]["both"], ["quick"])

    def test_measured_runs_do_not_print_fallback_warnings(self):
        # zli's warnings pile up over the timed loop ((N+1)(N+2)/2 blocks), so
        # only the one-iteration calibration may print them.
        os.environ["FAKE_ZLI_MODE"] = "warnings"
        os.environ["FAKE_TIME"] = json.dumps([0.001, 0.001])
        doc = self.run_bench(self.config(levels=[6], rounds=2, min_time=0.5))
        runs = [c["argv"] for c in self.calls("zli")]
        self.assertEqual(len(runs), 3)
        self.assertNotIn("-v", runs[0])
        self.assertEqual(runs[0][runs[0].index("-n") + 1], "1")
        for argv in runs[1:]:
            self.assertEqual(argv[argv.index("-v") + 1], "1")
            self.assertEqual(argv[argv.index("-n") + 1], str(bench.MAX_ITERATIONS))
        point = self.points(doc)["zli:parquet/6"]
        self.assertEqual(point["status"], "ok")
        self.assertEqual(point["flags"]["both"], ["fallback"])

    def test_zstd_failure_modes(self):
        cases = [
            ("fail", "zstd exited with status 15: Error loading files"),
            ("badlabel", "zstd printed no result for level -6 (only for --6)"),
            ("wrongsize", "zstd read 102,401 bytes, but the input is 102,400 bytes"),
            ("noheader", "zstd printed no benchmark header"),
            ("garbage", "zstd printed no benchmark header"),
        ]
        for mode, message in cases:
            with self.subTest(mode=mode):
                os.environ["FAKE_ZSTD_MODE"] = mode
                doc = self.run_bench(self.config(levels=[6], quick=True))
                point = self.points(doc)["zstd/6"]
                self.assertEqual(point["status"], "failed")
                self.assertEqual(point["error"], message)
                self.assertEqual(self.points(doc)["zli:parquet/6"]["status"], "ok")
        os.environ["FAKE_ZSTD_MODE"] = "badlabel"
        doc = self.run_bench(self.config(levels=[-3], quick=True))
        self.assertEqual(
            self.points(doc)["zstd/-3"]["error"],
            "zstd printed no result for level --3 (only for -3)",
        )

    def test_waits_for_a_busy_fixed_core(self):
        messages = []
        proc = ft.FakeProc(os.path.join(self.dir, "busy"), loads=[{2: 50}, {2: 50}, {}])
        probe = self.make_probe(proc=proc)
        doc = self.run_bench(
            self.config(levels=[6], quick=True), probe=probe, progress=messages.append
        )
        self.assertEqual(proc.sleeps, [0.3, 1.0, 1.0, 0.3])
        self.assertIn("core 2 is busy (50%); waiting up to 30 s", messages)
        for point in doc["points"]:
            self.assertEqual(point["flags"]["c"], [])
            self.assertEqual(point["command"][:3], ["taskset", "-c", "2"])
        self.assertEqual(doc["machine"]["core"], 2)

    def test_gives_up_waiting_and_marks_results(self):
        messages = []
        proc = ft.FakeProc(os.path.join(self.dir, "busy"), loads=[{2: 60}])
        probe = self.make_probe(proc=proc)
        doc = self.run_bench(
            self.config(levels=[6], quick=True), probe=probe, progress=messages.append
        )
        self.assertEqual(proc.sleeps, ([0.3] + [1.0] * 30) * 2)
        self.assertIn("core 2 is still busy; measuring anyway (marked)", messages)
        self.assertTrue(any(m.endswith("  core busy") for m in messages))
        for point in doc["points"]:
            self.assertEqual(point["status"], "ok")
            self.assertEqual(point["flags"]["c"], ["contended"])
            self.assertEqual(point["flags"]["d"][:1], ["contended"])
        self.assertEqual({c["core"] for c in self.calls()}, {"2"})

    def test_auto_mode_switches_to_a_quiet_core(self):
        messages = []
        proc = ft.FakeProc(os.path.join(self.dir, "busy"), loads=[{}, {1: 80}])
        probe = self.make_probe(proc=proc)
        doc = self.run_bench(
            self.config(levels=[6], quick=True, core="auto"),
            probe=probe,
            progress=messages.append,
        )
        # start: all idle, core 1 wins the tie; then core 1 is busy -> core 2.
        self.assertEqual(proc.sleeps, [0.4, 0.3, 1.0, 0.3])
        self.assertEqual([c["core"] for c in self.calls()], ["2", "2"])
        self.assertEqual(doc["machine"]["core"], 2)
        self.assertEqual(doc["machine"]["core_choice"], "auto")
        self.assertEqual(
            doc["notes"], ["Switched from core 1 to core 2 because core 1 was busy."]
        )
        self.assertIn("switched to core 2 (0% busy)", messages)
        self.assertIn("on core 1 (auto)", messages[0])
        for point in doc["points"]:
            self.assertEqual(point["flags"]["c"], [])
            self.assertEqual(point["command"][:3], ["taskset", "-c", "2"])

    def test_auto_mode_waits_when_every_core_is_busy(self):
        proc = ft.FakeProc(
            os.path.join(self.dir, "busy"),
            loads=[{}, {c: 40 for c in range(4)}, {c: 40 for c in range(4)}, {3: 90}],
        )
        probe = self.make_probe(proc=proc)
        doc = self.run_bench(
            self.config(levels=[6], quick=True, core="auto"), probe=probe
        )
        self.assertEqual(proc.sleeps[:4], [0.4, 0.3, 1.0, 1.0])
        self.assertEqual(doc["machine"]["core"], 1)
        self.assertEqual(doc["notes"], [])

    def test_foreign_load_during_a_run(self):
        probe = self.make_probe(ForeignProbe)
        doc = self.run_bench(self.config(levels=[6], quick=True), probe=probe)
        for point in doc["points"]:
            self.assertEqual(point["flags"]["c"], ["contended"])

    def test_unpinned(self):
        doc = self.run_bench(self.config(levels=[6], rounds=2, core="none"))
        self.assertEqual(self.proc.sleeps, [])
        self.assertEqual(doc["machine"]["core"], None)
        self.assertEqual(doc["machine"]["core_choice"], "none")
        self.assertEqual(doc["machine"]["pin"], "none")
        self.assertIsNone(doc["machine"]["governor"])
        for point in doc["points"]:
            self.assertEqual(point["flags"]["both"], ["unpinned"])
            self.assertNotEqual(point["command"][0], "taskset")
        self.assertEqual({c["core"] for c in self.calls()}, {None})

    def test_affinity_when_taskset_is_missing(self):
        which = shutil.which
        core = max(os.sched_getaffinity(0))
        proc = ft.FakeProc(os.path.join(self.dir, "real"), cores=core + 1)
        probe = self.make_probe(proc=proc, allowed=os.sched_getaffinity(0))

        def no_taskset(name, *args, **kwargs):
            return None if name == "taskset" else which(name, *args, **kwargs)

        with mock.patch.object(bench.shutil, "which", no_taskset):
            with mock.patch.object(bench.threading, "active_count", return_value=1):
                doc = self.run_bench(
                    self.config(levels=[6], quick=True, core=core), probe=probe
                )
            self.assertEqual(doc["machine"]["pin"], "affinity")
            self.assertEqual(doc["machine"]["core"], core)
            for call in self.calls():
                self.assertEqual(call["affinity"], [core])
                self.assertIsNone(call["core"])
            for point in doc["points"]:
                self.assertEqual(point["flags"]["both"], ["quick"])
                self.assertNotIn("taskset", point["command"])

            with mock.patch.object(bench.threading, "active_count", return_value=2):
                doc = self.run_bench(
                    self.config(levels=[6], quick=True, core=core), probe=probe
                )
        self.assertEqual(doc["machine"]["pin"], "none")
        self.assertIsNone(doc["machine"]["core"])
        self.assertEqual(doc["machine"]["core_choice"], "fixed")
        self.assertIn("not pinned", doc["notes"][0])
        for point in doc["points"]:
            self.assertEqual(point["flags"]["both"], ["quick", "unpinned"])

    def test_environment(self):
        os.environ.update(
            ZSTD_CLEVEL="19",
            ZSTD_NBTHREADS="4",
            GLIBC_TUNABLES="glibc.malloc.hugetlb=1",
            MALLOC_ARENA_MAX="2",
            LC_ALL="de_DE.UTF-8",
        )
        doc = self.run_bench(self.config(levels=[-1, 6], quick=True))
        self.assertEqual(
            doc["settings"]["env"],
            {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1", "MALLOC_ARENA_MAX": "2"},
        )
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(
                call["env"],
                {
                    "ZSTD_CLEVEL": None,
                    "ZSTD_NBTHREADS": None,
                    "LC_ALL": "C",
                    "GLIBC_TUNABLES": "glibc.malloc.hugetlb=1",
                },
            )
            self.assertEqual(os.getpgid(0) != call["pgid"], True)
            self.assertEqual(call["pgid"], call["pid"])
        self.assertEqual(os.environ["ZSTD_CLEVEL"], "19")

    def test_input_changed(self):
        def grow(message):
            if message.startswith("["):
                with open(self.input, "ab") as f:
                    f.write(b"more")

        with self.assertRaises(bench.BenchError) as caught:
            self.run_bench(self.config(levels=[6], quick=True), progress=grow)
        self.assertEqual(str(caught.exception), "INPUT changed during the benchmark")

    def test_input_removed(self):
        def remove(message):
            if message.startswith("[") and os.path.exists(self.input):
                os.remove(self.input)

        with self.assertRaises(bench.BenchError):
            self.run_bench(self.config(levels=[6], quick=True), progress=remove)


@unittest.skipUnless(shutil.which("taskset"), "needs taskset")
class RealTasksetTest(FakeToolCase):
    fake_taskset = False

    def test_children_run_on_the_core(self):
        allowed = os.sched_getaffinity(0)
        core = max(allowed)
        proc = ft.FakeProc(os.path.join(self.dir, "real"), cores=core + 1)
        probe = self.make_probe(proc=proc, allowed=allowed)
        doc = self.run_bench(
            self.config(levels=[6], quick=True, core=core), probe=probe
        )
        self.assertEqual(doc["machine"]["pin"], "taskset")
        for call in self.calls():
            self.assertEqual(call["affinity"], [core])
        for point in doc["points"]:
            self.assertEqual(point["command"][:3], ["taskset", "-c", str(core)])
        self.assertEqual(os.sched_getaffinity(0), allowed)


class RecordTraceTest(FakeToolCase):
    def record(self, **overrides):
        config = self.config(**dict(dict(chunk_size_mb=1), **overrides))
        return bench.record_trace(config, self.work, self.probe)

    def test_records_and_verifies(self):
        recording = self.record()
        trace = os.path.join(self.work, "trace.cbor")
        frame = os.path.join(self.work, "trace.zl")
        self.assertEqual(
            recording,
            bench.Recording(
                trace_path=trace,
                frame_bytes=self.size("zli", 6),
                verified=True,
                fallback=False,
                error=None,
            ),
        )
        self.assertTrue(os.path.isfile(trace))
        self.assertEqual(os.path.getsize(frame), self.size("zli", 6))
        self.assertEqual(sorted(os.listdir(self.work)), ["trace.cbor", "trace.zl"])
        compress, decompress = self.calls("zli")
        self.assertEqual(
            compress["argv"],
            ["compress", self.input, "-p", "parquet", "--chunk-size-mb", "1"]
            + ["-o", frame, "--trace", trace, "-f"],
        )
        self.assertIs(compress["stdout_null"], True)
        self.assertEqual(
            decompress["argv"],
            ["decompress", frame, "-o", os.path.join(self.work, "trace.out"), "-f"],
        )
        self.assertEqual([compress["core"], decompress["core"]], ["2", "2"])
        self.assertEqual(self.proc.sleeps, [])

    def test_profile_and_options(self):
        self.record(profile_arg=",", core="none")
        bench.record_trace(
            self.config(profiles=["csv", "serial"]), self.work, self.probe, "serial"
        )
        first, _, second, _ = self.calls("zli")
        self.assertEqual(
            first["argv"][2:8],
            ["-p", "parquet", "--profile-arg", ",", "--chunk-size-mb", "1"],
        )
        self.assertIsNone(first["core"])
        self.assertEqual(second["argv"][2:5], ["-p", "serial", "-o"])

    def test_auto_core(self):
        self.proc.loads = [{0: 0, 1: 90, 2: 0, 3: 50}]
        self.record(core="auto")
        self.assertEqual({c["core"] for c in self.calls("zli")}, {"2"})
        self.assertEqual(self.proc.sleeps, [0.4])

    def test_corrupt_frame(self):
        os.environ["FAKE_ZLI_MODE"] = "corrupt"
        recording = self.record()
        self.assertIs(recording.verified, False)
        self.assertIsNone(recording.error)
        self.assertEqual(sorted(os.listdir(self.work)), ["trace.cbor", "trace.zl"])

    def test_short_of_disk_space(self):
        need = 2 * INPUT_BYTES + 64 * 1024 * 1024
        with mock.patch.object(bench, "free_bytes", return_value=need - 1) as free:
            recording = self.record()
        free.assert_called_once_with(self.work)
        self.assertIsNone(recording.verified)
        self.assertEqual(recording.frame_bytes, self.size("zli", 6))
        self.assertEqual([c["argv"][0] for c in self.calls("zli")], ["compress"])
        with mock.patch.object(bench, "free_bytes", return_value=need):
            self.assertIs(self.record().verified, True)
        self.assertGreater(bench.free_bytes(self.work), 0)
        self.assertEqual(bench.free_bytes(os.path.join(self.dir, "nope")), 0)

    def test_compress_failure(self):
        # zli still writes the trace of a failed compression, but no frame.
        os.environ["FAKE_ZLI_MODE"] = "strict"
        recording = self.record()
        self.assertIsNone(recording.frame_bytes)
        self.assertIsNone(recording.verified)
        self.assertTrue(recording.trace_path.endswith("trace.cbor"))
        self.assertEqual(
            recording.error,
            "zli compress exited with status 1: OpenZL error string: Stream "
            "parameter invalid",
        )
        self.assertEqual(len(self.calls("zli")), 1)

        os.environ["FAKE_ZLI_MODE"] = "crash"
        recording = self.record()
        self.assertIsNone(recording.trace_path)
        self.assertEqual(
            recording.error, "zli compress exited with status 139: Segmentation fault"
        )

    def test_timeout(self):
        os.environ["FAKE_ZLI_MODE"] = "hang"
        recording = self.record(timeout=0.3)
        self.assertEqual(recording.error, "zli compress timed out after 0.3 s")
        self.assertIsNone(recording.frame_bytes)
        (helper,) = self.calls("helper")
        self.assertTrue(gone(helper["pid"]))

    def test_fallback(self):
        os.environ["FAKE_ZLI_MODE"] = "warnings"
        recording = self.record()
        self.assertIs(recording.fallback, True)
        self.assertIs(recording.verified, True)
        self.assertIsNone(recording.error)

    def test_missing_trace(self):
        os.environ["FAKE_ZLI_MODE"] = "notrace"
        recording = self.record()
        self.assertIsNone(recording.trace_path)
        self.assertEqual(recording.frame_bytes, self.size("zli", 6))
        self.assertTrue(recording.error.startswith("zli compress wrote no trace"))

    def test_frame_size_of_the_profile(self):
        os.environ["FAKE_FRAME_BYTES"] = "31000"
        self.assertEqual(self.record().frame_bytes, 31000)


class LinkTraceTest(unittest.TestCase):
    def doc(self):
        def point(series, level, status, size):
            return {
                "id": f"{series}/{level}",
                "series": series,
                "level": level,
                "status": status,
                "bytes": size,
            }

        return {
            "series": [
                {"id": "zli:parquet", "label": "OpenZL -p parquet"},
                {"id": "zli:serial", "label": "OpenZL -p serial"},
                {"id": "zstd", "label": "zstd"},
            ],
            "points": [
                point("zli:parquet", 1, "ok", 6000),
                point("zli:parquet", 6, "ok", 5000),
                point("zli:serial", 6, "failed", None),
                point("zstd", 6, "ok", 5000),
            ],
            "trace_link": {"state": "none"},
            "notes": ["earlier note"],
        }

    def recording(self, frame=5000, verified=True, error=None):
        return bench.Recording("/w/trace.cbor", frame, verified, False, error)

    def test_linked(self):
        doc = self.doc()
        bench.link_trace(doc, "zli:parquet", self.recording(), (102400, 4900))
        self.assertEqual(
            doc["trace_link"],
            {
                "state": "linked",
                "series": "zli:parquet",
                "point": "zli:parquet/6",
                "frame_bytes": 5000,
                "stream_bytes": 4900,
                "given_input_bytes": None,
                "given_stream_bytes": None,
                "verified": True,
            },
        )
        self.assertEqual(doc["notes"], ["earlier note"])

        doc = self.doc()
        bench.link_trace(
            doc, "zli:parquet", self.recording(), (102400, 4900), (102400, 4900)
        )
        self.assertEqual(doc["trace_link"]["state"], "linked")
        self.assertEqual(doc["trace_link"]["given_input_bytes"], 102400)
        self.assertEqual(doc["trace_link"]["given_stream_bytes"], 4900)
        self.assertEqual(doc["notes"], ["earlier note"])

    def test_mismatch(self):
        doc = self.doc()
        bench.link_trace(
            doc, "zli:parquet", self.recording(), (102400, 4900), (102400, 4800)
        )
        link = doc["trace_link"]
        self.assertEqual(link["state"], "mismatch")
        self.assertEqual(link["point"], "zli:parquet/6")
        self.assertEqual(
            (link["given_input_bytes"], link["given_stream_bytes"]), (102400, 4800)
        )
        self.assertEqual(len(doc["notes"]), 2)
        self.assertIn("4,800 B in streams", doc["notes"][1])
        self.assertIn("4,900 B in streams", doc["notes"][1])
        self.assertIn("OpenZL -p parquet", doc["notes"][1])

        doc = self.doc()
        bench.link_trace(doc, "zli:parquet", self.recording(), None, (1, 2))
        self.assertEqual(doc["trace_link"]["state"], "mismatch")
        self.assertIn("no readable trace", doc["notes"][1])

    def test_frame_mismatch(self):
        doc = self.doc()
        # The frame check comes before the given-trace check.
        bench.link_trace(
            doc, "zli:parquet", self.recording(5100), (102400, 5000), (1, 2)
        )
        self.assertEqual(doc["trace_link"]["state"], "frame_mismatch")
        self.assertEqual(doc["trace_link"]["frame_bytes"], 5100)
        self.assertEqual(len(doc["notes"]), 2)
        self.assertIn("5,100 B", doc["notes"][1])
        self.assertIn("5,000 B", doc["notes"][1])

        doc = self.doc()
        failed = self.recording(None, None, "zli compress exited with status 1")
        bench.link_trace(doc, "zli:parquet", failed, None)
        self.assertEqual(doc["trace_link"]["state"], "frame_mismatch")
        self.assertIn("zli compress exited with status 1", doc["notes"][1])
        self.assertIn("5,000 B", doc["notes"][1])

    def test_no_point(self):
        doc = self.doc()
        bench.link_trace(doc, "zli:serial", self.recording(), (102400, 4900))
        link = doc["trace_link"]
        self.assertEqual(link["state"], "no_point")
        self.assertEqual(link["series"], "zli:serial")
        self.assertIsNone(link["point"])
        self.assertEqual(link["frame_bytes"], 5000)
        self.assertIn("OpenZL -p serial has no measured -l 6 point", doc["notes"][1])
        self.assertIn("5,000 B", doc["notes"][1])

        doc = self.doc()
        bench.link_trace(doc, "zli:other", self.recording(), (102400, 4900))
        self.assertEqual(doc["trace_link"]["state"], "no_point")
        self.assertIn("zli:other", doc["notes"][1])

    def test_none(self):
        for args in (
            (None, self.recording(), (102400, 4900)),
            ("zli:parquet", None, None),
        ):
            with self.subTest(args=args):
                doc = self.doc()
                bench.link_trace(doc, *args)
                self.assertEqual(doc["trace_link"], bench._empty_link())
                self.assertEqual(doc["notes"], ["earlier note"])

    def test_unverified_frame_is_noted(self):
        doc = self.doc()
        bench.link_trace(doc, "zli:parquet", self.recording(verified=False), (1, 2))
        self.assertEqual(doc["trace_link"]["state"], "linked")
        self.assertIs(doc["trace_link"]["verified"], False)
        self.assertEqual(
            doc["notes"][1],
            "The frame recorded with OpenZL -p parquet did not decompress back to "
            "the input.",
        )

    def test_notes_are_created(self):
        doc = self.doc()
        del doc["notes"]
        bench.link_trace(doc, "zli:serial", self.recording(), None)
        self.assertEqual(len(doc["notes"]), 1)


if __name__ == "__main__":
    unittest.main()
