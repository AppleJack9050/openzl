# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import itertools
import json
import math
import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench
import pareto
import text_report

BENCH = os.path.join(HERE, "data", "bench_sensors.json")
ALL = "zli:parquet+zli:serial+zstd+zstd:long27"


def raw_doc():
    with open(BENCH, encoding="utf-8") as f:
        return json.load(f)


def pt(pid, size, c, d=None, status="ok", series=None):
    return {
        "id": pid,
        "series": series or pid.split("/")[0],
        "status": status,
        "bytes": size,
        "c_speed": c,
        "d_speed": c if d is None else d,
    }


def brute_frontier(points, axis):
    cands = pareto.candidates(points, axis)
    kept = [q for q in cands if not any(pareto.dominates(p, q, axis) for p in cands)]
    keys = [k + "_speed" for k in axis]
    kept.sort(key=lambda p: (p["bytes"], *[-p[k] for k in keys], p["id"]))
    return [p["id"] for p in kept]


def brute_beaten_by(points, axis):
    cands = pareto.candidates(points, axis)
    front_ids = set(brute_frontier(points, axis))
    front = [p for p in cands if p["id"] in front_ids]
    keys = [k + "_speed" for k in axis]
    beaten = {}
    for q in cands:
        if q["id"] in front_ids:
            continue
        doms = [f for f in front if pareto.dominates(f, q, axis)]
        best = min(doms, key=lambda f: (-f["bytes"], *[-f[k] for k in keys], f["id"]))
        beaten[q["id"]] = best["id"]
    return beaten


def random_three_way(rng, ids):
    """Points with many equal sizes, speeds and whole triples, and some bad ones."""
    points = []
    for i in range(rng.randint(0, 25)):
        s = rng.choice(ids)
        size = (
            rng.choice([100, 150, 200]) if rng.random() < 0.7 else rng.randint(50, 300)
        )
        c, d = (
            rng.choice([1.0, 2.0, 2.5, 3.0])
            if rng.random() < 0.7
            else rng.uniform(0.5, 4)
            for _ in range(2)
        )
        if points and rng.random() < 0.1:
            size, c, d = pareto._triple(rng.choice(points))
        status = "failed" if rng.random() < 0.1 else "ok"
        # Ids whose string order differs from their number order.
        p = pt(f"{s}/{rng.randint(1, 9)}{i}", size, c, d=d, status=status)
        if rng.random() < 0.05:
            key = rng.choice(["c_speed", "d_speed"])
            p[key] = rng.choice([0, None, math.nan, -1.0, True])
        points.append(p)
    return points


def every_subset(ids):
    for size in range(1, len(ids) + 1):
        yield from itertools.combinations(ids, size)


class DominanceTest(unittest.TestCase):
    def test_strictness_and_ties(self):
        a = pt("s/1", 100, 10.0)
        self.assertFalse(pareto.dominates(a, pt("s/2", 100, 10.0), "c"))
        self.assertFalse(pareto.dominates(a, a, "c"))
        self.assertTrue(pareto.dominates(a, pt("s/2", 101, 10.0), "c"))
        self.assertTrue(pareto.dominates(a, pt("s/2", 100, 9.99), "c"))
        self.assertTrue(pareto.dominates(a, pt("s/2", 200, 1.0), "c"))
        self.assertFalse(pareto.dominates(a, pt("s/2", 99, 1.0), "c"))
        self.assertFalse(pareto.dominates(a, pt("s/2", 101, 10.5), "c"))

    def test_axis_picks_the_speed(self):
        a = pt("s/1", 100, 10.0, d=1.0)
        b = pt("s/2", 100, 5.0, d=2.0)
        self.assertTrue(pareto.dominates(a, b, "c"))
        self.assertFalse(pareto.dominates(a, b, "d"))
        self.assertTrue(pareto.dominates(b, a, "d"))

    def test_both_speeds(self):
        a = pt("s/1", 100, 10.0, d=1.0)
        self.assertFalse(pareto.dominates(a, pt("s/2", 100, 10.0, d=1.0), "cd"))
        b = pt("s/2", 100, 5.0, d=2.0)
        self.assertFalse(pareto.dominates(a, b, "cd"))
        self.assertFalse(pareto.dominates(b, a, "cd"))
        self.assertTrue(pareto.dominates(pt("s/3", 100, 10.0, d=1.5), a, "cd"))
        self.assertFalse(pareto.dominates(a, pt("s/3", 100, 10.0, d=1.5), "cd"))
        self.assertTrue(pareto.dominates(pt("s/4", 99, 10.0, d=1.0), a, "cd"))
        big = 2**60 + 1
        small, large = pt("s/5", big, 1.0), pt("s/6", big + 1, 1.0)
        self.assertTrue(pareto.dominates(small, large, "cd"))
        self.assertFalse(pareto.dominates(large, small, "cd"))

    def test_sizes_compare_as_integers(self):
        # As floats these two sizes are equal and neither point would win.
        big = 2**60 + 1
        a, b = pt("s/1", big, 1.0), pt("s/2", big + 1, 1.0)
        self.assertEqual(float(a["bytes"]), float(b["bytes"]))
        self.assertTrue(pareto.dominates(a, b, "c"))
        self.assertEqual(pareto.frontier([b, a], "c"), ["s/1"])

    def test_candidates(self):
        points = [
            pt("s/1", 100, 10.0),
            pt("s/2", 100, 10.0, status="failed"),
            pt("s/3", None, 10.0),
            pt("s/4", 0, 10.0),
            pt("s/5", True, 10.0),
            pt("s/6", 100.0, 10.0),
            pt("s/7", 100, 0.0),
            pt("s/8", 100, math.nan),
            pt("s/9", 100, math.inf),
            pt("s/10", 100, True),
            pt("s/11", 100, 10**400),
            pt("s/12", 100, "10"),
            pt("t/1", 100, 10.0),
            pt("u/1", 100, 10.0, d=0.0),
            pt("u/2", 100, 10.0, d=math.nan),
            pt("u/3", 100, 10.0, d=True),
            pt("u/4", 100, 10.0),
        ]
        points[-1]["d_speed"] = None
        self.assertEqual(
            [p["id"] for p in pareto.candidates(points, "c")],
            ["s/1", "t/1", "u/1", "u/2", "u/3", "u/4"],
        )
        self.assertEqual(
            [p["id"] for p in pareto.candidates(points, "cd")], ["s/1", "t/1"]
        )
        self.assertEqual(
            [p["id"] for p in pareto.candidates(points, "c", ["t"])], ["t/1"]
        )
        self.assertEqual(pareto.candidates(points, "c", []), [])

    def test_frontier_keeps_exact_duplicates_in_order(self):
        points = [
            pt("s/3", 300, 30.0),
            pt("s/2", 100, 10.0),
            pt("s/1", 100, 10.0),
            pt("s/4", 100, 9.0),
            pt("s/5", 200, 10.0),
            pt("s/6", 300, 30.0),
            pt("s/7", 250, 30.0),
        ]
        self.assertEqual(pareto.frontier(points, "c"), ["s/1", "s/2", "s/7"])
        self.assertEqual(pareto.frontier([], "c"), [])
        self.assertEqual(pareto.frontier(points, "c", ["x"]), [])

    def test_frontier_order_best_ratio_first(self):
        points = [pt("s/1", 300, 30.0), pt("s/2", 100, 1.0), pt("s/3", 200, 20.0)]
        self.assertEqual(pareto.frontier(points, "c"), ["s/2", "s/3", "s/1"])


class OracleTest(unittest.TestCase):
    def random_points(self, rng, n, series=("a",)):
        sizes = [rng.randint(1, 12) for _ in range(4)] + [rng.randint(1, 10**9)]
        speeds = [rng.choice([1.0, 2.0, 2.5, 3.0]), rng.uniform(0.001, 5000.0)]
        points = []
        for i in range(n):
            size = rng.choice(sizes) if rng.random() < 0.7 else rng.randint(1, 40)
            speed = rng.choice(speeds) if rng.random() < 0.5 else rng.uniform(0.5, 4)
            status = "ok" if rng.random() < 0.9 else "failed"
            s = rng.choice(series)
            p = pt(f"{s}/{i}", size, speed, d=rng.choice(speeds), status=status)
            if rng.random() < 0.05:
                p["c_speed"] = rng.choice([0.0, None, math.nan])
            if rng.random() < 0.05:
                p["d_speed"] = rng.choice([0.0, None, math.nan])
            points.append(p)
        return points

    def test_frontier_matches_brute_force(self):
        rng = random.Random(1234)
        for _ in range(400):
            points = self.random_points(rng, rng.randint(0, 30))
            rng.shuffle(points)
            for axis in pareto.FRONTIER_KEYS:
                self.assertEqual(
                    pareto.frontier(points, axis), brute_frontier(points, axis)
                )

    def test_subsets_and_beaten_by_match_brute_force(self):
        rng = random.Random(99)
        for trial in range(60):
            count = 5 if trial < 5 else rng.randint(1, 3)
            ids = [f"s{i}" for i in range(count)]
            n = rng.randint(0, 10 if count == 5 else 18)
            doc = {
                "series": [{"id": s} for s in ids],
                "points": self.random_points(rng, n, ids),
            }
            result = pareto.frontiers(doc)
            self.assertEqual(list(result), list(pareto.FRONTIER_KEYS))
            for axis in pareto.FRONTIER_KEYS:
                self.assertEqual(len(result[axis]["subsets"]), 2**count - 1)
                self.assertEqual(len(result[axis]["beaten_by"]), 2**count - 1)
                for size in range(1, count + 1):
                    for combo in itertools.combinations(ids, size):
                        key = "+".join(combo)
                        chosen = [p for p in doc["points"] if p["series"] in combo]
                        self.assertEqual(
                            result[axis]["subsets"][key], brute_frontier(chosen, axis)
                        )
                        self.assertEqual(
                            result[axis]["beaten_by"][key],
                            brute_beaten_by(chosen, axis),
                        )

    def test_three_way_heavy_ties(self):
        rng = random.Random(7)
        for _ in range(300):
            ids = [f"s{i}" for i in range(rng.randint(1, 5))]
            doc = {
                "series": [{"id": s} for s in ids],
                "points": random_three_way(rng, ids),
            }
            by_id = {p["id"]: p for p in doc["points"]}
            result = pareto.frontiers(doc)["cd"]
            for combo in every_subset(ids):
                key = "+".join(combo)
                chosen = [p for p in doc["points"] if p["series"] in combo]
                self.assertEqual(result["subsets"][key], brute_frontier(chosen, "cd"))
                beaten = result["beaten_by"][key]
                self.assertEqual(beaten, brute_beaten_by(chosen, "cd"))
                self.assertEqual(list(beaten), sorted(beaten))
                for q, f in beaten.items():
                    self.assertIn(f, result["subsets"][key])
                    self.assertTrue(pareto.dominates(by_id[f], by_id[q], "cd"))

    def test_two_way_points_stay_unless_tied(self):
        rng = random.Random(7)
        for _ in range(300):
            ids = [f"s{i}" for i in range(rng.randint(1, 5))]
            doc = {
                "series": [{"id": s} for s in ids],
                "points": random_three_way(rng, ids),
            }
            result = pareto.frontiers(doc)
            for combo in every_subset(ids):
                key = "+".join(combo)
                cands = pareto.candidates(doc["points"], "cd", combo)
                on_both = set(result["cd"]["subsets"][key])
                usable = {p["id"]: p for p in cands}
                for a, other in (("c", "d"), ("d", "c")):
                    for pid in result[a]["subsets"][key]:
                        p = usable.get(pid)
                        if p is None or pid in on_both:
                            continue
                        # Only a point of the same size and speed that is faster
                        # at the other speed can push it off.
                        self.assertTrue(
                            any(
                                q["bytes"] == p["bytes"]
                                and q[a + "_speed"] == p[a + "_speed"]
                                and q[other + "_speed"] > p[other + "_speed"]
                                for q in cands
                            ),
                            (key, a, pid),
                        )

        points = [
            pt("s/1", 100, 5.0, d=1.0),
            pt("s/2", 100, 5.0, d=2.0),
            pt("s/3", 50, 1.0, d=1.0),
        ]
        result = pareto.frontiers({"series": [{"id": "s"}], "points": points})
        self.assertEqual(result["c"]["subsets"]["s"], ["s/3", "s/1", "s/2"])
        self.assertEqual(result["d"]["subsets"]["s"], ["s/3", "s/2"])
        self.assertEqual(result["cd"]["subsets"]["s"], ["s/3", "s/2"])
        self.assertEqual(result["cd"]["beaten_by"]["s"], {"s/1": "s/2"})

    def test_three_way_is_fast(self):
        ids = [f"s{i}" for i in range(pareto.MAX_SERIES)]
        points = []
        for k in range(1000):
            points.append(pt(f"s{k % 5}/{k}", 1000 + k, 1000.0 + k, d=5000.0 - k))
        for j in range(1000):
            i = 1000 + j
            points.append(pt(f"s{i % 5}/{i}", 1001 + j, 999.5 + j, d=4999.5 - j))
        self.assertEqual(len(points), pareto.MAX_POINTS)
        doc = {"series": [{"id": s} for s in ids], "points": points}
        start = time.process_time()
        result = pareto.frontiers(doc)
        # About 0.1 s; the bound leaves room for a busy machine.
        self.assertLess(time.process_time() - start, 5)
        everything = "+".join(ids)
        self.assertEqual(
            result["cd"]["subsets"][everything],
            [f"s{k % 5}/{k}" for k in range(1000)],
        )
        self.assertEqual(
            result["cd"]["beaten_by"][everything],
            {f"s{(1000 + j) % 5}/{1000 + j}": f"s{j % 5}/{j}" for j in range(1000)},
        )
        self.assertEqual(len(result["cd"]["subsets"]), 31)


class SubsetTest(unittest.TestCase):
    def setUp(self):
        self.doc = pareto.normalize(raw_doc())

    def test_every_subset_in_document_order(self):
        ids = [s["id"] for s in self.doc["series"]]
        self.assertEqual(set(self.doc["frontiers"]), {"c", "d", "cd"})
        for axis in pareto.FRONTIER_KEYS:
            part = self.doc["frontiers"][axis]
            self.assertEqual(set(part), {"subsets", "beaten_by"})
            self.assertEqual(len(part["subsets"]), 15)
            self.assertEqual(set(part["subsets"]), set(part["beaten_by"]))
            for key in part["subsets"]:
                members = key.split("+")
                self.assertEqual(members, [i for i in ids if i in members])
            expected = {
                "+".join(c)
                for size in range(1, 5)
                for c in itertools.combinations(ids, size)
            }
            self.assertEqual(set(part["subsets"]), expected)
        self.assertIn(ALL, self.doc["frontiers"]["c"]["subsets"])

    def test_subset_key(self):
        doc = self.doc
        self.assertEqual(
            pareto.subset_key(doc, ["zstd", "zli:parquet"]), "zli:parquet+zstd"
        )
        self.assertEqual(pareto.subset_key(doc, {"zstd:long27"}), "zstd:long27")
        self.assertEqual(pareto.subset_key(doc, ["zstd", "nope"]), "zstd")
        self.assertEqual(
            pareto.subset_key(doc, reversed([s["id"] for s in doc["series"]])), ALL
        )

    def test_subset_only_holds_its_series(self):
        for axis in pareto.FRONTIER_KEYS:
            part = self.doc["frontiers"][axis]
            for key, ids in part["subsets"].items():
                members = set(key.split("+"))
                for pid in ids + list(part["beaten_by"][key]):
                    self.assertIn(pid.split("/")[0], members)


class BeatenByTest(unittest.TestCase):
    def beaten(self, points, axis="c"):
        doc = {"series": [{"id": "s"}], "points": points}
        return pareto.frontiers(doc)[axis]["beaten_by"]["s"]

    def test_closest_ratio_wins(self):
        points = [
            pt("s/1", 100, 10.0),
            pt("s/2", 200, 20.0),
            pt("s/3", 300, 30.0),
            pt("s/4", 250, 15.0),
            pt("s/5", 350, 5.0),
            pt("s/6", 200, 19.0),
            pt("s/7", 150, 10.0),
        ]
        self.assertEqual(
            self.beaten(points),
            {"s/4": "s/2", "s/5": "s/3", "s/6": "s/2", "s/7": "s/1"},
        )

    def test_ties_go_to_the_lowest_id(self):
        points = [
            pt("s/9", 100, 10.0),
            pt("s/10", 100, 10.0),
            pt("s/2", 100, 10.0),
            pt("s/3", 120, 5.0),
        ]
        # String order: "s/10" < "s/2" < "s/9".
        self.assertEqual(self.beaten(points), {"s/3": "s/10"})

    def test_only_candidates_are_listed(self):
        points = [
            pt("s/1", 100, 10.0),
            pt("s/2", 200, 5.0, status="failed"),
            pt("s/3", 200, 0.0),
        ]
        self.assertEqual(self.beaten(points), {})

    def test_frontier_points_are_not_beaten(self):
        points = [pt("s/1", 100, 10.0), pt("s/2", 100, 10.0), pt("s/3", 90, 1.0)]
        self.assertEqual(self.beaten(points), {})


class BeatenByBothTest(unittest.TestCase):
    def check(self, points, front, beaten):
        result = pareto.frontiers({"series": [{"id": "s"}], "points": points})["cd"]
        self.assertEqual(result["subsets"]["s"], front)
        self.assertEqual(result["beaten_by"]["s"], beaten)
        self.assertEqual(pareto.frontier(points, "cd"), front)

    def test_equal_sizes_go_to_the_faster_compressor(self):
        points = [
            pt("s/1", 100, 10.0, d=1.0),
            pt("s/2", 100, 1.0, d=10.0),
            pt("s/3", 200, 1.0, d=1.0),
            pt("s/4", 150, 5.0, d=0.5),
        ]
        self.check(points, ["s/1", "s/2"], {"s/3": "s/1", "s/4": "s/1"})

    def test_closest_ratio_wins(self):
        points = [
            pt("s/1", 100, 5.0, d=5.0),
            pt("s/2", 150, 6.0, d=6.0),
            pt("s/3", 200, 5.0, d=5.0),
            pt("s/4", 160, 5.5, d=5.5),
        ]
        self.check(points, ["s/1", "s/2"], {"s/3": "s/2", "s/4": "s/2"})

    def test_ties_go_to_the_lowest_id(self):
        points = [
            pt("s/9", 100, 10.0, d=10.0),
            pt("s/10", 100, 10.0, d=10.0),
            pt("s/2", 100, 10.0, d=10.0),
            pt("s/3", 200, 1.0, d=1.0),
        ]
        self.check(points, ["s/10", "s/2", "s/9"], {"s/3": "s/10"})

    def test_equal_sizes_and_compression_go_to_the_faster_decompressor(self):
        points = [
            pt("s/1", 100, 9.0, d=5.0),
            pt("s/2", 100, 5.0, d=9.0),
            pt("s/3", 120, 4.0, d=4.0),
            pt("s/4", 100, 9.0, d=4.0),
        ]
        self.check(points, ["s/1", "s/2"], {"s/3": "s/1", "s/4": "s/1"})

    def test_equal_points_do_not_beat_each_other(self):
        self.check([pt("s/1", 100, 1.0), pt("s/2", 100, 1.0)], ["s/1", "s/2"], {})

    def test_only_a_point_that_beats_all_three(self):
        # s/2 is the smallest but slower to compress than s/1, so s/1 goes to s/3.
        points = [
            pt("s/1", 100, 5.0, d=5.0),
            pt("s/2", 90, 1.0, d=9.0),
            pt("s/3", 95, 6.0, d=6.0),
            pt("s/4", 120, 1.0, d=1.0),
        ]
        self.check(points, ["s/2", "s/3"], {"s/1": "s/3", "s/4": "s/3"})

    def test_only_candidates_are_listed(self):
        points = [
            pt("s/1", 100, 10.0),
            pt("s/2", 200, 5.0, status="failed"),
            pt("s/3", 200, 5.0, d=0.0),
            pt("s/4", 200, 5.0),
            pt("s/5", 50, 50.0, d=math.nan),
        ]
        points[3]["d_speed"] = None
        self.check(points, ["s/1"], {})


class DemoTest(unittest.TestCase):
    """The sensors.parquet measurement shipped as tests/data/bench_sensors.json."""

    def setUp(self):
        with open(BENCH, "rb") as f:
            self.doc = pareto.loads(f.read())
        self.by_id = {p["id"]: p for p in self.doc["points"]}

    def test_compression_frontier(self):
        c = self.doc["frontiers"]["c"]
        front = c["subsets"][ALL]
        self.assertEqual(front, brute_frontier(self.doc["points"], "c"))
        self.assertEqual(
            front,
            ["zli:parquet/1", "zstd/4", "zstd/3", "zstd/2", "zstd/1", "zstd/-5"],
        )
        best = self.by_id["zli:parquet/1"]
        self.assertEqual((best["bytes"], best["c_speed"]), (537684, 282.2))
        labels = [self.by_id[i]["level_label"] for i in front]
        self.assertEqual(labels, ["-l 1", "-4", "-3", "-2", "-1", "--fast=5"])

    def test_parquet_level_1_beats_the_higher_levels(self):
        c = self.doc["frontiers"]["c"]
        self.assertEqual(c["subsets"]["zli:parquet"], ["zli:parquet/1"])
        for level in (2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 19, 22):
            pid = f"zli:parquet/{level}"
            self.assertEqual(c["beaten_by"]["zli:parquet"][pid], "zli:parquet/1")
            self.assertEqual(c["beaten_by"][ALL][pid], "zli:parquet/1")
            self.assertTrue(
                pareto.dominates(self.by_id["zli:parquet/1"], self.by_id[pid], "c")
            )

    def test_three_way_frontier(self):
        fr = self.doc["frontiers"]
        cd = fr["cd"]
        self.assertEqual(
            cd["subsets"][ALL],
            [
                "zli:parquet/1",
                "zstd/4",
                "zstd/3",
                "zstd/2",
                "zstd:long27/1",
                "zstd/1",
                "zstd/-5",
            ],
        )
        # zstd --long=27 -1 gives up a little compression speed to zstd -1 and a
        # little size, but decompresses faster: it wins only with both speeds.
        self.assertNotIn("zstd:long27/1", fr["c"]["subsets"][ALL])
        self.assertNotIn("zstd:long27/1", fr["d"]["subsets"][ALL])
        beaten = cd["beaten_by"][ALL]
        self.assertEqual(len(beaten), 42)
        others = {
            "zli:serial/1": "zstd/1",
            "zli:serial/2": "zstd/2",
            "zli:serial/3": "zstd/3",
            "zli:serial/4": "zstd/4",
            "zstd/-1": "zstd/1",
        }
        self.assertEqual({k: v for k, v in beaten.items() if k in others}, others)
        rest = {v for k, v in beaten.items() if k not in others}
        self.assertEqual(rest, {"zli:parquet/1"})
        self.assertEqual(
            cd["subsets"]["zstd"],
            [
                "zstd/19",
                "zstd/15",
                "zstd/12",
                "zstd/9",
                "zstd/8",
                "zstd/7",
                "zstd/6",
                "zstd/5",
                "zstd/4",
                "zstd/3",
                "zstd/2",
                "zstd/1",
                "zstd/-5",
            ],
        )
        self.assertEqual(
            cd["beaten_by"]["zstd"], {"zstd/-1": "zstd/1", "zstd/22": "zstd/19"}
        )
        self.assertEqual(cd["subsets"]["zli:parquet"], ["zli:parquet/1"])
        self.assertEqual(
            cd["subsets"]["zli:parquet+zli:serial+zstd:long27"],
            ["zli:parquet/1", "zli:serial/2", "zstd:long27/1"],
        )
        for key, ids in cd["subsets"].items():
            for axis in pareto.AXES:
                self.assertLessEqual(set(fr[axis]["subsets"][key]), set(ids))

    def test_other_frontiers_match_the_oracle(self):
        points = self.doc["points"]
        d = self.doc["frontiers"]["d"]
        self.assertEqual(d["subsets"][ALL], ["zli:parquet/1", "zstd/-5"])
        for axis in pareto.FRONTIER_KEYS:
            part = self.doc["frontiers"][axis]
            for key, ids in part["subsets"].items():
                chosen = [p for p in points if p["series"] in key.split("+")]
                self.assertEqual(ids, brute_frontier(chosen, axis))
                self.assertEqual(part["beaten_by"][key], brute_beaten_by(chosen, axis))
        # zstd -1 (level 1) is smaller and faster than zstd --fast=1.
        self.assertEqual(
            self.doc["frontiers"]["c"]["beaten_by"][ALL]["zstd/-1"], "zstd/1"
        )

    def test_normalized_document_round_trips(self):
        doc = self.doc
        self.assertEqual(pareto.loads(pareto.dumps(doc)), doc)
        self.assertEqual(pareto.normalize(doc), doc)
        self.assertEqual(doc["trace_link"]["state"], "linked")
        self.assertEqual(doc["trace_link"]["point"], "zli:parquet/6")
        self.assertEqual(len(doc["points"]), 49)
        self.assertEqual(doc["points"][0]["c_samples"], [589.03])


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        self.raw = raw_doc()

    def rejects(self, message, change=None):
        doc = copy.deepcopy(self.raw)
        if change is not None:
            change(doc)
        with self.assertRaisesRegex(pareto.BenchmarkFormatError, message):
            pareto.normalize(doc)

    def point(self, doc, pid="zli:parquet/1"):
        return next(p for p in doc["points"] if p["id"] == pid)

    def test_error_is_a_value_error(self):
        self.assertTrue(issubclass(pareto.BenchmarkFormatError, ValueError))

    def test_not_a_dict(self):
        for value in ([], "x", None, 3):
            with self.assertRaisesRegex(
                pareto.BenchmarkFormatError, "not a JSON object"
            ):
                pareto.normalize(value)

    def test_format_and_version(self):
        self.rejects("not a codec_reviewer benchmark", lambda d: d.update(format="x"))
        self.rejects("not a codec_reviewer benchmark", lambda d: d.pop("format"))
        self.rejects("version", lambda d: d.update(version=2))
        self.rejects("version", lambda d: d.update(version=True))
        self.rejects("version", lambda d: d.update(version="1"))
        self.rejects("version", lambda d: d.update(version=1.0))

    def test_series_count(self):
        self.rejects("1 to 5 series", lambda d: d.update(series=[]))
        self.rejects("1 to 5 series", lambda d: d.update(series={}))
        self.rejects("1 to 5 series", lambda d: d.pop("series"))

        def six(d):
            d["series"] += [
                {"id": f"zstd:x{i}", "tool": "zstd", "slot": 1} for i in range(2)
            ]

        self.rejects("1 to 5 series", six)

    def test_bad_series_ids(self):
        bad = ["", "Zstd", "-zstd", "a" * 65, "zstd\n", "zst d", "zstd/1", "é", 5, None]
        for sid in bad:
            with self.subTest(sid=sid):
                self.rejects("bad id", lambda d, s=sid: d["series"][2].update(id=s))
        self.rejects("appears twice", lambda d: d["series"][1].update(id="zli:parquet"))
        self.rejects("not an object", lambda d: d["series"].append("zstd"))
        doc = copy.deepcopy(self.raw)
        doc["series"] = [
            {"id": "a" * 64, "tool": "zstd", "slot": 0},
            {"id": "0:._-z", "tool": "zli", "slot": 0},
        ]
        doc["points"] = []
        doc["trace_link"] = None
        self.assertEqual(len(pareto.normalize(doc)["series"]), 2)

    def test_tool_and_slot(self):
        self.rejects("unknown tool", lambda d: d["series"][0].update(tool="gzip"))
        self.rejects("unknown tool", lambda d: d["series"][0].pop("tool"))
        for slot in (5, -1, "0", True, 1.0, None):
            with self.subTest(slot=slot):
                self.rejects("slot", lambda d, s=slot: d["series"][0].update(slot=s))

    def test_too_many_points(self):
        self.rejects(
            "more than 2,000 points",
            lambda d: d.update(points=[{}] * (pareto.MAX_POINTS + 1)),
        )
        self.rejects("points must be a list", lambda d: d.update(points={}))

    def test_point_ids(self):
        self.rejects(
            "should have the id zli:parquet/1",
            lambda d: self.point(d).update(id="zli:parquet/2"),
        )
        self.rejects(
            "should have the id", lambda d: self.point(d).update(id="zli:parquet/01")
        )
        self.rejects("should have the id", lambda d: self.point(d).pop("id"))
        self.rejects(
            "appears twice",
            lambda d: d["points"].append(copy.deepcopy(self.point(d))),
        )
        self.rejects("not an object", lambda d: d["points"].append([]))

    def test_unknown_series(self):
        self.rejects("unknown series", lambda d: self.point(d).update(series="zli:csv"))
        self.rejects("unknown series", lambda d: self.point(d).update(series=["x"]))
        self.rejects("unknown series", lambda d: self.point(d).pop("series"))

    def test_levels(self):
        def level(value, pid="zstd/1"):
            def change(d):
                p = self.point(d, pid)
                p["level"] = value
                p["id"] = f"{p['series']}/{value}"

            return change

        for value in (0, 23, -51, 1.0, True, "1", None):
            with self.subTest(level=value):
                self.rejects("bad level", level(value))
        self.rejects("zli levels start at 1", level(-1, "zli:parquet/1"))
        self.rejects("zli levels start at 1", level(-50, "zli:serial/1"))
        doc = copy.deepcopy(self.raw)
        level(-50)(doc)
        p = next(p for p in pareto.normalize(doc)["points"] if p["level"] == -50)
        self.assertEqual(p["id"], "zstd/-50")

    def test_status(self):
        for status in ("OK", "", None, "error"):
            with self.subTest(status=status):
                self.rejects(
                    "unknown status",
                    lambda d, s=status: self.point(d).update(status=s),
                )

    def test_ok_point_needs_size_and_speeds(self):
        for field, value in (
            ("bytes", None),
            ("bytes", 0),
            ("c_speed", None),
            ("d_speed", 0),
            ("d_speed", 0.0),
        ):
            with self.subTest(field=field, value=value):
                self.rejects(
                    "Point zli:parquet/1",
                    lambda d, f=field, v=value: self.point(d).update({f: v}),
                )
        self.rejects("Point zli:parquet/1", lambda d: self.point(d).pop("bytes"))
        self.rejects(
            "bytes must be an integer", lambda d: self.point(d).update(bytes=5.0)
        )
        self.rejects("bytes must be", lambda d: self.point(d).update(bytes=-3))
        self.rejects("bytes must be", lambda d: self.point(d).update(bytes=2**53))

    def test_bools_are_not_numbers(self):
        cases = [
            lambda d: self.point(d).update(bytes=True),
            lambda d: self.point(d).update(c_speed=True),
            lambda d: self.point(d).update(d_speed=True),
            lambda d: self.point(d).update(c_samples=[True]),
            lambda d: d["input"].update(bytes=True),
            lambda d: d["input"].update(bytes=False),
            lambda d: d.update(elapsed_s=True),
            lambda d: d["settings"].update(rounds=True),
            lambda d: d["machine"].update(core=False),
            lambda d: d["trace_link"].update(frame_bytes=True),
        ]
        for i, change in enumerate(cases):
            with self.subTest(case=i):
                self.rejects("must be", change)

    def test_non_finite_numbers(self):
        cases = [
            lambda d: self.point(d).update(c_speed=math.nan),
            lambda d: self.point(d).update(d_speed=math.inf),
            lambda d: self.point(d).update(c_samples=[1.0, -math.inf]),
            lambda d: d.update(elapsed_s=math.nan),
            lambda d: d["settings"].update(min_time_s=math.inf),
            # Anywhere, even in a key that would be dropped.
            lambda d: d.update(extra={"x": [math.nan]}),
            lambda d: self.point(d, "zstd/22").update(
                status="failed", c_speed=math.nan
            ),
        ]
        for i, change in enumerate(cases):
            with self.subTest(case=i):
                self.rejects("finite", change)
        self.rejects("finite", lambda d: self.point(d).update(c_speed=10**400))

    def test_samples(self):
        self.rejects("c_samples", lambda d: self.point(d).update(c_samples="1"))
        self.rejects("c_samples", lambda d: self.point(d).update(c_samples={}))
        self.rejects("d_samples", lambda d: self.point(d).update(d_samples=[1, -2]))
        self.rejects("d_samples", lambda d: self.point(d).update(d_samples=[0]))
        self.rejects("d_samples", lambda d: self.point(d).update(d_samples=["1"]))
        self.rejects(
            "at most 50",
            lambda d: self.point(d).update(c_samples=[1.0] * (pareto.MAX_SAMPLES + 1)),
        )
        doc = copy.deepcopy(self.raw)
        self.point(doc).update(c_samples=[1] * pareto.MAX_SAMPLES, d_samples=None)
        clean = self.point(pareto.normalize(doc))
        self.assertEqual(clean["c_samples"], [1.0] * pareto.MAX_SAMPLES)
        self.assertIsInstance(clean["c_samples"][0], float)
        self.assertEqual(clean["d_samples"], [])

    def test_input_complete_and_trace_link(self):
        self.rejects("input.bytes", lambda d: d["input"].update(bytes=-1))
        self.rejects("input.bytes", lambda d: d["input"].update(bytes="5"))
        self.rejects("input.bytes", lambda d: d["input"].pop("bytes"))
        self.rejects("input.bytes", lambda d: d.pop("input"))
        for value in ("yes", None, 1, 0):
            with self.subTest(complete=value):
                self.rejects(
                    "complete must be", lambda d, v=value: d.update(complete=v)
                )
        for value in ("bogus", None, "LINKED", ["linked"]):
            with self.subTest(state=value):
                self.rejects(
                    "trace_link.state",
                    lambda d, v=value: d["trace_link"].update(state=v),
                )
        self.rejects("trace_link.state", lambda d: d.update(trace_link="linked"))
        self.rejects("verified", lambda d: d["trace_link"].update(verified="yes"))

    def test_missing_trace_link_is_none(self):
        doc = copy.deepcopy(self.raw)
        del doc["trace_link"]
        link = pareto.normalize(doc)["trace_link"]
        self.assertEqual(link["state"], "none")
        self.assertEqual(
            set(link),
            {
                "state",
                "series",
                "point",
                "frame_bytes",
                "stream_bytes",
                "given_input_bytes",
                "given_stream_bytes",
                "verified",
            },
        )
        self.assertTrue(all(v is None for k, v in link.items() if k != "state"))

    def test_trace_link_refs_must_exist(self):
        doc = copy.deepcopy(self.raw)
        doc["trace_link"].update(series="zli:csv", point=["x"])
        link = pareto.normalize(doc)["trace_link"]
        self.assertEqual((link["series"], link["point"]), (None, None))
        doc = copy.deepcopy(self.raw)
        self.point(doc, "zli:parquet/6").update(status="failed")
        self.assertIsNone(pareto.normalize(doc)["trace_link"]["point"])

    def test_tampered_frontiers_are_recomputed(self):
        clean = pareto.normalize(copy.deepcopy(self.raw))
        doc = copy.deepcopy(self.raw)
        doc["frontiers"] = {
            "c": {"subsets": {ALL: ["zstd/22"]}, "beaten_by": {ALL: {"x": "y"}}},
            "cd": {"subsets": {ALL: []}, "beaten_by": {}},
            "evil": 1,
        }
        out = pareto.normalize(doc)
        self.assertEqual(out["frontiers"], clean["frontiers"])
        self.assertEqual(out["frontiers"], pareto.frontiers(out))
        doc["frontiers"] = "garbage"
        self.assertEqual(pareto.normalize(doc)["frontiers"], clean["frontiers"])

    def test_unknown_keys_dropped_everywhere(self):
        doc = copy.deepcopy(self.raw)
        doc["evil"] = 1
        for part in ("input", "machine", "settings", "trace_link"):
            doc[part]["evil"] = 1
        doc["tools"]["zli"]["evil"] = 1
        doc["tools"]["zstd"]["build_dir"] = "x"
        doc["tools"]["gzip"] = {"path": "/bin/gzip"}
        doc["series"][0]["evil"] = 1
        p = self.point(doc)
        p["evil"] = 1
        p["flags"]["evil"] = ["x"]
        p["flags"]["c"] = ["contended", "loud", "short", "contended", 3]
        p["flags"]["both"] = ["short", "fallback", "quick"]
        p["flags"]["d"] = "contended"
        out = pareto.normalize(doc)
        text = json.dumps(out)
        self.assertNotIn("evil", text)
        self.assertNotIn("gzip", text)
        self.assertNotIn("build_dir", out["tools"]["zstd"])
        self.assertEqual(
            list(out),
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
                "frontiers",
            ],
        )
        flags = self.point(out)["flags"]
        self.assertEqual(
            flags, {"c": ["contended", "short"], "d": [], "both": ["quick", "fallback"]}
        )

    def test_missing_optional_fields(self):
        doc = {
            "format": pareto.FORMAT,
            "version": 1,
            "complete": False,
            "input": {"bytes": 10},
            "series": [{"id": "zstd", "tool": "zstd", "slot": 0}],
            "points": [
                {"id": "zstd/-3", "series": "zstd", "level": -3, "status": "skipped"},
                {
                    "id": "zstd/19",
                    "series": "zstd",
                    "level": 19,
                    "status": "ok",
                    "bytes": 4,
                    "c_speed": 2,
                    "d_speed": 3.5,
                },
            ],
        }
        out = pareto.normalize(doc)
        self.assertEqual(out["input"], {"name": "", "bytes": 10, "sha256": ""})
        self.assertEqual(out["series"][0]["label"], "zstd")
        self.assertEqual(out["series"][0]["args"], [])
        self.assertIsNone(out["stopped"])
        self.assertEqual(out["created"], "")
        self.assertIsNone(out["elapsed_s"])
        self.assertEqual(out["machine"]["pin"], "none")
        self.assertIsNone(out["machine"]["core"])
        self.assertEqual(out["tools"]["zli"]["note"], "")
        self.assertIsNone(out["tools"]["zli"]["build_dir"])
        self.assertEqual(out["settings"]["env"], {})
        self.assertFalse(out["settings"]["quick"])
        self.assertEqual(out["notes"], [])
        skipped, ok = out["points"]
        self.assertEqual(skipped["level_label"], "--fast=3")
        self.assertIsNone(skipped["error"])
        self.assertEqual(ok["level_label"], "-19")
        self.assertEqual((ok["c_speed"], ok["d_speed"]), (2.0, 3.5))
        self.assertIsInstance(ok["c_speed"], float)
        self.assertEqual(ok["flags"], {"c": [], "d": [], "both": []})
        self.assertEqual(out["frontiers"]["c"]["subsets"], {"zstd": ["zstd/19"]})

    def test_non_ok_points_lose_their_numbers(self):
        doc = copy.deepcopy(self.raw)
        self.point(doc).update(status="timeout", error="took too long")
        p = self.point(pareto.normalize(doc))
        self.assertEqual(p["error"], "took too long")
        for field in ("bytes", "c_speed", "d_speed"):
            self.assertIsNone(p[field])
        self.assertEqual((p["c_samples"], p["d_samples"]), ([], []))
        self.point(doc).update(error="x" * 900)
        self.assertEqual(len(self.point(pareto.normalize(doc))["error"]), 500)
        self.point(doc).update(status="ok", error="stale")
        self.assertIsNone(self.point(pareto.normalize(doc))["error"])

    def test_long_strings_cut(self):
        doc = copy.deepcopy(self.raw)
        long = "y" * (pareto.MAX_TEXT + 500)
        doc["series"][0]["label"] = long
        doc["tools"]["zli"]["note"] = long
        doc["machine"]["cpu"] = long
        doc["notes"] = [long]
        self.point(doc)["command"] = ["zli", long]
        out = pareto.normalize(doc)
        self.assertEqual(out["series"][0]["label"], "y" * pareto.MAX_TEXT)
        self.assertEqual(len(out["tools"]["zli"]["note"]), pareto.MAX_TEXT)
        self.assertEqual(len(out["machine"]["cpu"]), pareto.MAX_TEXT)
        self.assertEqual(len(out["notes"][0]), pareto.MAX_TEXT)
        self.assertEqual(len(self.point(out)["command"][1]), pareto.MAX_TEXT)

    def test_lenient_lists_and_text(self):
        doc = copy.deepcopy(self.raw)
        doc["input"]["name"] = "/home/someone/data/sensors.parquet"
        doc["input"]["sha256"] = "not hex"
        doc["notes"] = ["a", 3, None] + [f"n{i}" for i in range(30)]
        doc["settings"]["env"] = {f"V{i}": str(i) for i in range(30)}
        doc["settings"]["env"]["V0"] = 5
        doc["series"][0]["args"] = ["-p", 3]
        doc["series"][1]["label"] = 7
        doc["machine"]["pin"] = "chroot"
        doc["machine"]["core_choice"] = 3
        doc["stopped"] = "crashed"
        self.point(doc)["command"] = ["x"] * 65
        self.point(doc, "zstd/1")["command"] = "zstd -1"
        self.point(doc, "zstd/3")["command"] = ["zstd", None]
        out = pareto.normalize(doc)
        self.assertEqual(out["input"]["name"], "sensors.parquet")
        self.assertEqual(out["input"]["sha256"], "")
        self.assertEqual(out["notes"][:3], ["a", "n0", "n1"])
        self.assertEqual(len(out["notes"]), 20)
        self.assertEqual(len(out["settings"]["env"]), 20)
        self.assertNotIn("V0", out["settings"]["env"])
        self.assertEqual(out["series"][0]["args"], [])
        self.assertEqual(out["series"][1]["label"], "zli:serial")
        self.assertEqual(out["machine"]["pin"], "none")
        self.assertEqual(out["machine"]["core_choice"], "none")
        self.assertIsNone(out["stopped"])
        by_id = {p["id"]: p for p in out["points"]}
        for pid in ("zli:parquet/1", "zstd/1", "zstd/3"):
            self.assertEqual(by_id[pid]["command"], [])
        self.assertEqual(len(by_id["zstd/4"]["command"]), 11)
        doc["input"]["name"] = "C:\\data\\x.parquet"
        self.assertEqual(pareto.normalize(doc)["input"]["name"], "x.parquet")

    def test_input_is_not_modified(self):
        doc = copy.deepcopy(self.raw)
        doc["evil"] = 1
        before = copy.deepcopy(doc)
        out = pareto.normalize(doc)
        self.assertEqual(doc, before)
        out["series"][0]["label"] = "changed"
        out["points"][0]["c_samples"].append(1.0)
        self.assertEqual(doc, before)


class LoadsDumpsTest(unittest.TestCase):
    def setUp(self):
        with open(BENCH, "rb") as f:
            self.data = f.read()

    def assert_rejected(self, text, message):
        with self.assertRaisesRegex(pareto.BenchmarkFormatError, message):
            pareto.loads(text)

    def test_str_bytes_and_bom(self):
        doc = pareto.loads(self.data)
        self.assertEqual(pareto.loads(self.data.decode("utf-8")), doc)
        self.assertEqual(pareto.loads(bytearray(self.data)), doc)
        self.assertEqual(pareto.loads(b"\xef\xbb\xbf" + self.data), doc)

    def test_size_limit(self):
        padded = self.data + b" " * (pareto.MAX_DOCUMENT - len(self.data))
        self.assertEqual(len(pareto.loads(padded)["points"]), 49)
        self.assert_rejected(padded + b" ", "larger than 8,388,608 bytes")
        self.assert_rejected(padded.decode("utf-8") + " ", "larger than")

    def test_largest_results_file_loads(self):
        # The biggest file --bench-json writes: bench.py's five series with the
        # longest ids, every level, the most rounds, all text at its limit and
        # escaped by JSON to six bytes a character, and frontiers as large as they
        # get (in every subset, one point beats all the others).
        text = "\u00e9" * pareto.MAX_TEXT
        name = "\ufffd" * 255
        config = bench.BenchConfig(
            input=name,
            zli=text,
            zstd=text,
            profiles=[ch * 60 for ch in "abc"],
            profile_arg=text,
            chunk_size_mb=1024,
            zstd_long=27,
        )
        series = bench.series_for(config)
        points = []
        for rank, x in enumerate(series):
            zli = x["tool"] == "zli"
            for level in range(-bench.MAX_FAST, bench.MAX_LEVEL + 1):
                if level == 0 or (zli and level < 0):
                    continue
                top = level == bench.MAX_LEVEL
                speed = 1000.0 - rank if top else 1.23457e-05
                if zli:
                    run = [text, "benchmark", name, *x["args"], "-l", str(level)]
                    run += ["-n", "10000", "-v", "1", "--output-csv", "<tmp>/zli.csv"]
                else:
                    run = [text, *x["args"], "-q", "--ultra", f"-b{level}"]
                    run += [f"-e{level}", "-i1", "--", name]
                points.append(
                    {
                        "id": f"{x['id']}/{level}",
                        "series": x["id"],
                        "level": level,
                        "status": "ok",
                        "bytes": 1000 + rank if top else 10**9 - level,
                        "c_speed": speed,
                        "d_speed": speed,
                        "c_samples": [speed] * bench.MAX_ROUNDS,
                        "d_samples": [speed] * bench.MAX_ROUNDS,
                        "flags": {
                            "c": list(pareto.FLAGS_AXIS),
                            "d": list(pareto.FLAGS_AXIS),
                            "both": list(pareto.FLAGS_BOTH),
                        },
                        "command": ["taskset", "-c", "1023", *run],
                    }
                )
        tool = {"path": text, "sha256": "0" * 64, "bytes": 1, "version": text}
        raw = {
            "format": pareto.FORMAT,
            "version": pareto.VERSION,
            "complete": True,
            "created": text,
            "input": {"name": name, "bytes": pareto.MAX_INT, "sha256": "0" * 64},
            "machine": {"cpu": text, "kernel": text, "python": text},
            "tools": {"zli": dict(tool, build_dir=text, note=text), "zstd": tool},
            "settings": {
                "levels": text,
                "rounds": bench.MAX_ROUNDS,
                "env": dict.fromkeys(bench.ENV_RECORDED, text),
            },
            "series": series,
            "points": points,
            "notes": [text] * bench.MAX_NOTES,
        }
        doc = pareto.normalize(raw)
        self.assertEqual(len(doc["points"]), 3 * 22 + 2 * 72)
        beaten = doc["frontiers"]["cd"]["beaten_by"]
        self.assertEqual(sum(map(len, beaten.values())), 210 * 16 - 31)
        data = (pareto.dumps(doc) + "\n").encode("utf-8")
        # About 5.3 MB, 0.3 MB of it the three-way frontier.
        self.assertLess(len(data), pareto.MAX_DOCUMENT)
        self.assertEqual(pareto.loads(data), doc)

    def test_bad_utf8(self):
        self.assert_rejected(self.data.replace(b"sample CPU", b"sample \xff"), "UTF-8")
        self.assert_rejected(b'{"format": "\xc3"}', "not valid UTF-8")

    def test_not_json(self):
        self.assert_rejected(b"", "not valid JSON")
        self.assert_rejected(self.data[:-5], "not valid JSON")
        self.assert_rejected("{'format': 1}", "not valid JSON")
        self.assert_rejected("[]", "not a JSON object")
        self.assert_rejected("null", "not a JSON object")
        self.assert_rejected("[" * 100000 + "]" * 100000, "nested too deeply|JSON")
        self.assert_rejected("1" * 5000, "JSON|object")

    def test_non_finite_constants(self):
        text = self.data.decode("utf-8")
        for bad in ("NaN", "Infinity", "-Infinity", "1e999", "-1e400"):
            with self.subTest(bad=bad):
                self.assert_rejected(
                    text.replace('"elapsed_s": 41.2', f'"elapsed_s": {bad}'), "finite"
                )
        self.assert_rejected(
            text.replace('"c_speed": 282.2', '"c_speed": NaN'), "finite"
        )
        # Rejected even inside a key that normalize() would drop.
        self.assert_rejected(text.replace("{", '{"x": [Infinity], ', 1), "finite")

    def test_other_errors_become_format_errors(self):
        text = self.data.decode("utf-8")
        self.assert_rejected(text.replace('"version": 1', '"version": 2'), "version")

    def test_dumps(self):
        doc = pareto.loads(self.data)
        text = pareto.dumps(doc)
        self.assertNotIn(", ", text[:200])
        self.assertTrue(text.startswith('{"format":"codec_reviewer.benchmark"'))
        self.assertEqual(json.loads(text), doc)
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                pareto.dumps({"x": [bad]})


class SummaryTest(unittest.TestCase):
    def test_summary(self):
        doc = pareto.normalize(raw_doc())
        self.assertEqual(
            pareto.summary(doc),
            {
                "input": "sensors.parquet",
                "bytes": 2889011,
                "series": [
                    "OpenZL -p parquet",
                    "OpenZL -p serial",
                    "zstd",
                    "zstd --long=27",
                ],
                "points": 49,
                "complete": True,
            },
        )
        raw = raw_doc()
        raw["complete"] = False
        raw["points"][0]["status"] = "failed"
        raw["points"][1]["status"] = "skipped"
        summary = pareto.summary(pareto.normalize(raw))
        self.assertEqual((summary["points"], summary["complete"]), (47, False))
        self.assertEqual(json.loads(json.dumps(summary)), summary)


class RenderTableTest(unittest.TestCase):
    def render(self, change=None, glyphs=text_report.UNICODE):
        raw = raw_doc()
        if change is not None:
            change(raw)
        return pareto.render_table(pareto.normalize(raw), glyphs)

    @staticmethod
    def point(doc, pid):
        return next(p for p in doc["points"] if p["id"] == pid)

    @staticmethod
    def row(text, label, level):
        for line in text.splitlines():
            rest = line[len(label) + 2 :].lstrip()
            if line.startswith(f"  {label}  ") and rest.startswith(level + " "):
                return line
        raise AssertionError(f"no row for {label} {level}")

    def test_unicode_layout(self):
        text = self.render()
        lines = text.splitlines()
        self.assertEqual(
            lines[0],
            "RATIO VS SPEED  sensors.parquet · 2,889,011 B · one core (core 4) "
            "· best of 1 round",
        )
        self.assertEqual(
            lines[1],
            "zli: cachedObjs/a847…/zli (sha256 00000000) · built without -march: "
            "OpenZL SIMD kernels off   zstd 1.5.7",
        )
        header = lines[2].split()
        self.assertEqual(
            header,
            [
                "Series",
                "Level",
                "Bytes",
                "Ratio",
                "Comp",
                "MB/s",
                "Decomp",
                "MB/s",
                "Frontier",
                "Notes",
            ],
        )
        row = self.row(text, "OpenZL -p parquet", "-l 1")
        self.assertEqual(
            row.split(),
            [
                "OpenZL",
                "-p",
                "parquet",
                "-l",
                "1",
                "537,684",
                "5.37×",
                "282.2",
                "1,690.8",
                "C",
                "D",
                "3D",
            ],
        )
        self.assertIn("C D 3D", row)
        zstd1 = self.row(text, "zstd", "-1")
        self.assertEqual(zstd1.split()[-2:], ["C", "3D"])
        self.assertTrue(zstd1.endswith("C   3D"))
        long1 = self.row(text, "zstd --long=27", "-1")
        self.assertEqual(long1.split()[-1], "3D")
        self.assertNotIn("C", long1.split())
        self.assertTrue(long1.endswith("    3D"))
        self.assertEqual(lines[52], pareto._FRONTIER_KEY_LINE)
        self.assertTrue(self.row(text, "OpenZL -p parquet", "-l 6").endswith("traced"))
        self.assertIn("5.70", self.row(text, "zstd", "-22"))
        self.assertIn("The review above is the -l 6 run of OpenZL -p parquet.", lines)
        self.assertNotIn("⚠", text)
        self.assertTrue(text.endswith("\n"))
        # Columns line up: every decompression speed ends where its header does,
        # followed by the one-character "~" column.
        body = lines[3:52]
        self.assertEqual(len(body), 49)
        end = lines[2].index("Frontier") - 4
        self.assertEqual(lines[2][end - 10 : end + 1], "Decomp MB/s")
        for line in body:
            self.assertTrue(line[end].isdigit(), line)
            self.assertIn(line[end + 1 : end + 4], ("", "   "), line)

    def test_rows_grouped_by_series_levels_ascending(self):
        text = self.render(lambda d: d["points"].reverse())
        rows = text.splitlines()[3:52]
        labels = [r.split("  ")[1] for r in rows]
        self.assertEqual(
            list(dict.fromkeys(labels)),
            ["OpenZL -p parquet", "OpenZL -p serial", "zstd", "zstd --long=27"],
        )
        zstd = [r.split()[1] for r in rows if r.split("  ")[1] == "zstd"]
        self.assertEqual(zstd[:4], ["--fast=5", "--fast=1", "-1", "-2"])
        self.assertEqual(zstd[-1], "-22")
        parquet = [r.split()[4] for r in rows[:13]]
        self.assertEqual(parquet[:3], ["1", "2", "3"])
        self.assertEqual(parquet[-1], "22")

    def test_ascii(self):
        def change(d):
            d["series"][0]["label"] = "OpenZL → parquet é"
            d["notes"] = ["note — with dash"]
            self.point(d, "zstd/22").update(status="failed", error="boom")

        text = self.render(change, text_report.ASCII)
        text.encode("ascii")
        lines = text.splitlines()
        self.assertTrue(
            lines[0].startswith("RATIO VS SPEED  sensors.parquet | 2,889,011 B")
        )
        self.assertIn("cachedObjs/a847.../zli (sha256 00000000)", lines[1])
        self.assertIn("5.37x", self.row(text, "OpenZL -> parquet ?", "-l 1"))
        row = self.row(text, "zstd", "-22")
        self.assertEqual(row.split()[2:6], ["-", "-", "-", "-"])
        self.assertIn("note - with dash", lines)
        self.assertIn("! 1 point failed (see Notes)", lines)

    def test_failed_timeout_and_skipped_rows(self):
        def change(d):
            d["complete"] = False
            self.point(d, "zstd/22").update(
                status="failed", error="Error loading files\n\x1b[31mred"
            )
            self.point(d, "zstd/19").update(
                status="timeout", error="timed out after 900 s"
            )
            self.point(d, "zli:serial/22").update(
                status="skipped", error="skipped after a lower level timed out"
            )

        text = self.render(change)
        failed = self.row(text, "zstd", "-22")
        self.assertEqual(failed.split()[2:6], ["—", "—", "—", "—"])
        self.assertTrue(failed.endswith("failed: Error loading files ?[31mred"))
        self.assertNotIn("\x1b", text)
        self.assertIn("timeout: timed out after 900 s", self.row(text, "zstd", "-19"))
        self.assertIn(
            "skipped: skipped after a lower level timed out",
            self.row(text, "OpenZL -p serial", "-l 22"),
        )
        lines = text.splitlines()
        self.assertIn("⚠ incomplete: 1 of 49 points not measured", lines)
        self.assertIn("⚠ 1 point failed and 1 point timed out (see Notes)", lines)
        # The failed zstd -22 was not on the frontier, so the frontier is unchanged.
        self.assertEqual(
            self.row(text, "zstd", "--fast=5").split()[-3:], ["C", "D", "3D"]
        )

    def test_contended_and_short_marks(self):
        def change(d):
            self.point(d, "zli:parquet/1")["flags"] = {
                "c": ["contended"],
                "d": [],
                "both": [],
            }
            self.point(d, "zstd/3")["flags"] = {"c": [], "d": ["short"], "both": []}
            self.point(d, "zstd/4")["flags"] = {
                "c": ["contended"],
                "d": ["contended", "short"],
                "both": [],
            }

        text = self.render(change)
        lines = text.splitlines()
        parquet = self.row(text, "OpenZL -p parquet", "-l 1")
        self.assertIn("282.2~", parquet)
        self.assertIn("1,690.8 ", parquet)
        self.assertNotIn("1,690.8~", parquet)
        self.assertIn("core busy (C)", parquet)
        zstd3 = self.row(text, "zstd", "-3")
        self.assertIn("319.3 ", zstd3)
        self.assertIn("1,274.6~", zstd3)
        self.assertIn("short run (D)", zstd3)
        zstd4 = self.row(text, "zstd", "-4")
        self.assertIn("core busy (C D); short run (D)", zstd4)
        self.assertEqual(self.row(text, "zstd", "-5").count("~"), 0)
        self.assertIn("⚠ 2 points ran while the core was busy (marked ~)", lines)
        self.assertIn("⚠ 2 points timed less than 0.05 s per run (marked ~)", lines)
        # The number columns still line up with their "~" marks.
        self.assertEqual(parquet.index("282.2~"), zstd3.index("319.3 "))

    def test_interrupted_banner(self):
        def change(d):
            d["complete"] = False
            d["stopped"] = "interrupted"
            for p in d["points"][-12:]:
                p.update(status="skipped", error="interrupted")

        lines = self.render(change).splitlines()
        self.assertIn("⚠ interrupted: 12 of 49 points not measured", lines)
        self.assertNotIn("incomplete", "\n".join(lines))

    def test_interrupted_after_the_first_round(self):
        def change(d):
            d["complete"] = False
            d["stopped"] = "interrupted"
            d["settings"]["rounds"] = 3
            for p in d["points"]:
                p["c_samples"] = p["c_samples"] * 3
                p["d_samples"] = p["d_samples"] * 3
            for p in d["points"][:20]:
                p["c_samples"] = p["c_samples"][:1]
                p["d_samples"] = p["d_samples"][:1]

        lines = self.render(change).splitlines()
        self.assertIn("best of up to 3 rounds", lines[0])
        self.assertIn("⚠ interrupted: 20 points with fewer than 3 runs", lines)

    def test_nothing_measured_unpinned_fallback_quick(self):
        def change(d):
            d["machine"].update(pin="none", core=None, core_choice="none")
            d["settings"]["quick"] = True
            for p in d["points"]:
                p["flags"]["both"] = ["quick", "unpinned"]
            self.point(d, "zli:serial/1")["flags"]["both"].append("fallback")
            self.point(d, "zli:serial/2")["flags"]["both"].append("fallback")

        text = self.render(change)
        lines = text.splitlines()
        self.assertEqual(
            lines[0],
            "RATIO VS SPEED  sensors.parquet · 2,889,011 B · unpinned · quick: one pass",
        )
        self.assertIn(
            "⚠ the runs were not pinned to one core, so speeds are less repeatable",
            lines,
        )
        self.assertIn(
            "⚠ OpenZL fell back to generic compression for OpenZL -p serial -l 1, -l 2",
            lines,
        )
        self.assertIn(
            "quick; unpinned; fell back to generic",
            self.row(text, "OpenZL -p serial", "-l 1"),
        )

        def none_ok(d):
            for p in d["points"]:
                p.update(status="failed", error="x")
            d["trace_link"]["state"] = "no_point"

        lines = self.render(none_ok).splitlines()
        self.assertIn("⚠ no point was measured successfully", lines)
        self.assertIn("⚠ 49 points failed (see Notes)", lines)
        self.assertNotIn(pareto._FRONTIER_KEY_LINE, lines)

    def test_trace_link_sentences(self):
        def link(**fields):
            return lambda d: d["trace_link"].update(fields)

        text = self.render(link(verified=False))
        self.assertIn("The review above is the -l 6 run of OpenZL -p parquet.", text)
        self.assertIn("⚠ The traced frame did not decompress back to the input.", text)

        text = self.render(
            link(state="mismatch", given_input_bytes=100, given_stream_bytes=40)
        )
        self.assertNotIn("The review above is the", text)
        self.assertIn(
            "⚠ The review above is not one of these points: the given trace does not "
            "match what OpenZL -p parquet produces with these settings (100 B in, "
            "40 B in streams; the benchmark's trace has 2,889,011 B in, 560,780 B in "
            "streams).",
            text,
        )
        self.assertNotIn("traced", self.row(text, "OpenZL -p parquet", "-l 6"))

        text = self.render(link(state="frame_mismatch", frame_bytes=600000))
        self.assertIn(
            "⚠ The review above is not one of these points: OpenZL -p parquet wrote a "
            "600,000 B frame when traced, which does not match its -l 6 result "
            "(the profile picks its own level).",
            text,
        )
        text = self.render(link(state="no_point", series="zli:serial"))
        self.assertIn(
            "⚠ The review above cannot be matched to a point: OpenZL -p serial has "
            "no successful -l 6 result.",
            text,
        )
        text = self.render(link(state="none", series=None, point=None))
        self.assertNotIn("review above", text)
        self.assertNotIn("traced", text)

    def test_notes_are_printed(self):
        text = self.render(lambda d: d.update(notes=["first note", "second\nnote"]))
        lines = text.splitlines()
        self.assertEqual(lines[-2:], ["first note", "second note"])

    def test_zli_only_and_missing_frontiers(self):
        doc = pareto.normalize(raw_doc())
        doc["series"] = doc["series"][:1]
        doc["points"] = [p for p in doc["points"] if p["series"] == "zli:parquet"]
        without_cd = copy.deepcopy(doc)
        del without_cd["frontiers"]["cd"]
        del doc["frontiers"]
        for shown in (doc, without_cd):
            text = pareto.render_table(shown, text_report.UNICODE)
            self.assertNotIn("zstd", text)
            self.assertEqual(
                self.row(text, "OpenZL -p parquet", "-l 1").split()[-3:],
                ["C", "D", "3D"],
            )
            # Two fact lines and the header, the rows, the key, the trace link.
            self.assertEqual(len(text.splitlines()), 3 + 13 + 1 + 1)


if __name__ == "__main__":
    unittest.main()
