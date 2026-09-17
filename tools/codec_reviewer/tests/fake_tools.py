# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Fake `zli`, `zstd` and `taskset` programs, and a fake /proc tree, for tests.

The fake tools print what the real ones print (the formats bench.py parses)
without compressing anything, so a benchmark over them takes milliseconds.
Environment variables steer them:

``FAKE_MODE``
    ``ok`` (default) or a failure mode, for every call; ``MODE@L,MODE@L`` sets
    the mode of single levels (zli ``-l L``; zstd level L, or ``-K`` for
    ``--fast=K``) and ``MODE@L#I`` the mode of the I-th call (from 0) with the
    same arguments at that level; an item without ``@`` is the mode of the
    other calls.
    zli: fail, strict, crash, hang, garbage, nocsv, csv7, sizes_differ,
    warnings, n0, wrongsize, wrongsrc, wrongiters, notrace, corrupt, badversion.
    zstd: fail, hang, garbage, sizes_differ, badlabel, wrongsize, noheader,
    badversion.
``FAKE_ZLI_MODE`` / ``FAKE_ZSTD_MODE``
    override ``FAKE_MODE`` for one tool.
``FAKE_LOG``
    a file that gets one JSON line per call (see :func:`read_log`).
``FAKE_SIZES``
    a JSON object: level -> compressed bytes (both tools).
``FAKE_TIME``
    per-iteration milliseconds: ``C`` (decompression takes C/5), ``[C, D]``,
    or a list of those used in turn by the calls with the same arguments
    (``[[10, 2], [5, 3]]``: the first call of a point takes 10/2, the next 5/3).
``FAKE_FRAME_BYTES``
    the size of the frame `zli compress` writes (default: the level-6 size).
``FAKE_TRACE``
    a trace file `zli compress --trace` copies (default: a tiny CBOR map that is
    not a zli trace).
"""

from __future__ import annotations

import json
import os
import stat
import sys

PROFILES = ("csv", "json", "le-u64", "parquet", "serial")
CHUNKED_PROFILES = ("csv", "parquet")
ZLI_VERSION = "zstrong-cli version 0.1"
ZSTD_VERSION = "1.5.7"
FALLBACK_BLOCK = (
    "Encountered warnings during operation!:\n"
    "Code: Stream parameter invalid\n"
    "Message: Check `nbInputs == 1' failed\n"
    "Stack Trace:\n"
    "\t#0 parquetSegmenter (custom_parsers/parquet/parquet_graph.c:186)\n"
)
BAD_PROFILE = (
    "Invalid argument(s):\n"
    "\tProfile not found: 'nope'. See `zli list-profiles` for a list of "
    "supported profiles.\n"
)
STRICT_FAILURE = (
    "OpenZL Library Exception:\n"
    "\tOpenZL error code: 76\n"
    "OpenZL error string: Stream parameter invalid\n"
    "OpenZL error context: Code: Stream parameter invalid\n"
    "Stack Trace:\n"
    "\t#0 CCTX_startCompression (src/openzl/compress/cctx.c:1284): Forwarding error: \n"
)

_COMMON = r'''
import json, os, subprocess, sys, time

ARGS = sys.argv[1:]


def mode_for(level, index=None):
    """The mode of this call: "MODE@L" and "MODE@L#I" (the I-th call with these
    arguments, from 0) items win over a plain "MODE"."""
    spec = os.environ.get("FAKE_%s_MODE" % TOOL.upper()) or os.environ.get("FAKE_MODE")
    plain = "ok"
    for item in (spec or "ok").split(","):
        name, _, at = item.strip().partition("@")
        at_level, _, at_index = at.partition("#")
        if not at:
            plain = name
        elif level is not None and at_level == str(level):
            if not at_index or (index is not None and at_index == str(index)):
                return name
    return plain


def call_key():
    """The call's arguments without the repetition counts (-n N, -iT) and the
    log level (-v L)."""
    key, skip = [], False
    for arg in ARGS:
        if skip:
            skip = False
        elif arg in ("-n", "-v"):
            skip = True
        elif not (TOOL == "zstd" and arg.startswith("-i")):
            key.append(arg)
    return key


def call_index():
    """How many earlier calls had the same arguments (by FAKE_LOG)."""
    path = os.environ.get("FAKE_LOG")
    if not path or not os.path.exists(path):
        return 0
    key = call_key()
    index = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("tool") == TOOL and entry.get("key") == key:
                index += 1
    return index


def log(level, mode, index=0, **extra):
    """Append this call to FAKE_LOG."""
    path = os.environ.get("FAKE_LOG")
    if not path:
        return
    try:
        null = os.stat("/dev/null")
        out = os.fstat(1)
        stdout_null = (out.st_rdev, out.st_ino) == (null.st_rdev, null.st_ino)
    except OSError:
        stdout_null = False
    entry = {
        "tool": TOOL,
        "argv": ARGS,
        "key": call_key(),
        "level": level,
        "mode": mode,
        "index": index,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "affinity": sorted(os.sched_getaffinity(0)),
        "core": os.environ.get("FAKE_CORE"),
        "stdout_null": stdout_null,
        "env": {k: os.environ.get(k) for k in (
            "ZSTD_CLEVEL", "ZSTD_NBTHREADS", "LC_ALL", "GLIBC_TUNABLES")},
    }
    entry.update(extra)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def logged_call(level):
    """Log this call; (mode, index)."""
    index = call_index()
    mode = mode_for(level, index)
    log(level, mode, index)
    return mode, index


def hang():
    helper = subprocess.Popen([sys.executable, "-S", "-c", "import time; time.sleep(60)"])
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"tool": "helper", "pid": helper.pid, "key": None}) + "\n")
    time.sleep(60)
    sys.exit(0)


def size_for(level, input_bytes, index=0, mode="ok"):
    sizes = json.loads(os.environ.get("FAKE_SIZES") or "{}")
    if str(level) in sizes:
        size = int(sizes[str(level)])
    else:
        if TOOL == "zli":
            ratio = 3.0 + 0.15 * level
        elif level > 0:
            ratio = 2.0 + 0.1 * level
        else:
            ratio = 2.0 / (1 + 0.05 * -level)
        if any(a.startswith("--long=") for a in ARGS):
            ratio *= 1.02
        size = max(1, int(input_bytes / ratio))
    if mode == "sizes_differ":
        size += index
    return size


def times_for(level, index):
    """(compression, decompression) milliseconds per iteration."""
    spec = os.environ.get("FAKE_TIME")
    if spec:
        value = json.loads(spec)
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[index % len(value)]
        if isinstance(value, list):
            return float(value[0]), float(value[1])
        return float(value), float(value) / 5
    c = 1.0 + 0.5 * level if level > 0 else 1.0 / (1 + 0.1 * -level)
    return c, 0.2


def option(name, default=None):
    if name in ARGS:
        i = ARGS.index(name)
        if i + 1 < len(ARGS):
            return ARGS[i + 1]
    return default


def fail(text, code=1):
    sys.stderr.write(text)
    sys.exit(code)
'''

_ZLI = r"""

def benchmark():
    source = ARGS[1]
    level = int(option("-l", "0"))
    mode, index = logged_call(level)
    iterations = int(option("-n", "1"))
    profile = option("-p", "")
    if option("--chunk-size-mb") is not None:
        sys.stderr.write(CHUNK_NOTICE)
    if mode == "hang":
        hang()
    if mode == "fail":
        fail(BAD_PROFILE)
    if mode == "strict":
        fail(STRICT_FAILURE)
    if not os.path.isfile(source):
        fail("Error: cannot open %s\n" % source)
    input_bytes = os.path.getsize(source)
    size = size_for(level, input_bytes, index, mode)
    c_ms, d_ms = times_for(level, index)
    src, iters = input_bytes, iterations
    ctime, dtime = c_ms * iterations, d_ms * iterations
    if mode == "n0":
        iters, ctime, dtime = 0, 0.00011, 3e-05
    if mode == "wrongsrc":
        src += 1
    if mode == "wrongiters":
        iters += 1
    printed = size + 1 if mode == "wrongsize" else size
    csv_path = option("--output-csv")
    if csv_path and mode != "nocsv":
        with open(csv_path, "w") as f:
            if mode == "garbage":
                f.write("\x00\x01garbage\nnot,a,benchmark\n")
            else:
                f.write("srcSize,compressedSize,compressionRatio,ctimeMs,dtimeMs,iters,path\n")
                row = [str(src), str(size), "%g" % (input_bytes / size),
                       "%g" % ctime, "%g" % dtime, str(iters)]
                if mode == "csv7":
                    row.append(source)
                f.write(",".join(row) + "\n")
    # zli logs warnings (and the result line) only up to -v; its warnings pile
    # up in the context, so iteration k of the timed loop prints k + 1 blocks.
    quiet = option("-v", "3") in ("0", "1")
    if mode == "warnings" and not quiet:
        sys.stderr.write(FALLBACK_BLOCK * ((iterations + 1) * (iterations + 2) // 2))
    if quiet:
        return
    if iters:
        c_speed = input_bytes / 1e6 / (ctime / iters / 1000)
        d_speed = input_bytes / 1e6 / (dtime / iters / 1000)
    else:
        c_speed = d_speed = 0.0
    sys.stderr.write("\r1 files: %d -> %d (%.2f),  %.2f MB/s  %.2f MB/s\x1b[K\n" % (
        input_bytes, printed, input_bytes / printed, c_speed, d_speed))


def write_trace():
    trace = os.environ.get("FAKE_TRACE")
    if trace:
        with open(trace, "rb") as f:
            content = f.read()
    else:
        content = b"\xa1\x64fake\x64cbor"
    with open(option("--trace"), "wb") as f:
        f.write(content)


def compress():
    source = ARGS[1]
    mode, _ = logged_call(None)
    sys.stdout.write("digraph {\n  start -> store;\n}\n" * 2000)
    if option("--chunk-size-mb") is not None:
        sys.stderr.write(CHUNK_NOTICE)
    if mode == "hang":
        hang()
    if mode == "crash":
        fail("Segmentation fault\n", 139)
    if mode in ("fail", "strict"):
        # zli writes the trace of the failed compression, but no frame.
        write_trace()
        fail(STRICT_FAILURE)
    if mode == "warnings":
        sys.stderr.write(FALLBACK_BLOCK)
    input_bytes = os.path.getsize(source)
    frame_bytes = int(os.environ.get("FAKE_FRAME_BYTES") or size_for(6, input_bytes))
    header = ("FAKEZL\n%s\n" % os.path.abspath(source)).encode()
    with open(option("-o"), "wb") as f:
        f.write(header + b"\0" * max(0, frame_bytes - len(header)))
    if mode != "notrace":
        write_trace()
    sys.stderr.write("Compressed %d -> %d (%.2fx) in 1.000 ms, 1.00 MB/s\n" % (
        input_bytes, frame_bytes, input_bytes / frame_bytes))


def decompress():
    mode, _ = logged_call(None)
    with open(ARGS[1], "rb") as f:
        lines = f.read().split(b"\n")
    if lines[0] != b"FAKEZL":
        fail("OpenZL Library Exception:\n\tOpenZL error code: 1\n")
    with open(lines[1].decode(), "rb") as f:
        data = f.read()
    if mode == "corrupt":
        data = data[:-1] + bytes([data[-1] ^ 1]) if data else b"x"
    with open(option("-o"), "wb") as f:
        f.write(data)


def err(text=""):
    sys.stderr.write(text + "\n")


def main():
    command = ARGS[0] if ARGS else ""
    # The real zli prints its version and profile list on stderr.
    if command == "--version":
        err("zli 9.9" if mode_for(None) == "badversion" else ZLI_VERSION)
    elif command == "list-profiles":
        mode, _ = logged_call(None)
        if mode == "fail":
            fail("Error: unknown command\n")
        if mode == "garbage":
            err("no profiles here")
            return
        err("Available profiles:")
        for name in PROFILES:
            err("  -| %s\t= The %s profile" % (name, name))
        err()
    elif command == "benchmark":
        benchmark()
    elif command == "compress":
        compress()
    elif command == "decompress":
        decompress()
    else:
        fail("Error parsing arguments:\n\t unknown command %r\n" % command)


main()
"""

_ZSTD = r"""

def main():
    if ARGS == ["-V"]:
        if mode_for(None) == "badversion":
            print("gzip 1.12")
        else:
            print("*** Zstandard CLI (64-bit) v%s, by Yann Collet ***" % ZSTD_VERSION)
        return
    first = last = None
    fast = None
    seconds = 3
    files = []
    rest = False
    for arg in ARGS:
        if rest:
            files.append(arg)
        elif arg == "--":
            rest = True
        elif arg.startswith("--fast="):
            fast = int(arg[len("--fast="):])
        elif arg.startswith("-b") and arg != "-b":
            first = int(arg[2:])
        elif arg.startswith("-e"):
            last = int(arg[2:])
        elif arg.startswith("-i"):
            seconds = int(arg[2:])
        elif not arg.startswith("-"):
            files.append(arg)
    if fast is not None:
        levels = [-fast]
    else:
        first = 3 if first is None else first
        levels = list(range(first, (first if last is None else last) + 1))
    mode, index = logged_call(levels[0])
    if mode == "hang":
        hang()
    if mode == "fail" or len(files) != 1 or not os.path.isfile(files[0]):
        fail("Error loading files\n", 15)
    if mode == "garbage":
        print("garbage")
        return
    input_bytes = os.path.getsize(files[0])
    name = os.path.basename(files[0])
    if mode != "noheader":
        shown = input_bytes + 1 if mode == "wrongsize" else input_bytes
        print("bench %s : input %d bytes, %d seconds, 0 KB blocks" % (
            ZSTD_VERSION, shown, seconds))
    for level in levels:
        label = "-%d" % level if level > 0 else "--%d" % -level
        if mode == "badlabel":
            label = "--%d" % level if level > 0 else "-%d" % -level
        size = size_for(level, input_bytes, index, mode)
        c_ms, d_ms = times_for(level, index)
        c_speed = input_bytes / 1e6 / (c_ms / 1000)
        d_speed = input_bytes / 1e6 / (d_ms / 1000)
        print("%-8s %6d (%5.3f) %6.2f MB/s %6.1f MB/s  %s" % (
            label, size, input_bytes / size, c_speed, d_speed, name))


main()
"""

_TASKSET = r"""
import os, sys

args = sys.argv[1:]
if len(args) < 3 or args[0] != "-c" or not args[1].isdigit():
    sys.stderr.write("taskset: bad usage\n")
    sys.exit(1)
os.environ["FAKE_CORE"] = args[1]
os.execv(args[2], args[2:])
"""


def _constants(tool: str) -> str:
    values = {
        "TOOL": tool,
        "PROFILES": PROFILES,
        "ZLI_VERSION": ZLI_VERSION,
        "ZSTD_VERSION": ZSTD_VERSION,
        "FALLBACK_BLOCK": FALLBACK_BLOCK,
        "BAD_PROFILE": BAD_PROFILE,
        "STRICT_FAILURE": STRICT_FAILURE,
        "CHUNK_NOTICE": (
            "Chunking is not currently implemented for all profiles. Ignoring "
            "size parameter if unimplemented.\n"
            "Chunking is implemented for the following profiles: "
            + ", ".join(CHUNKED_PROFILES)
            + "\n"
        ),
    }
    return "".join(f"{k} = {v!r}\n" for k, v in values.items())


def _write(path: str, text: str) -> str:
    # -S skips site-packages: the fakes start in a few milliseconds.
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#!{sys.executable} -S\n{text}")
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_tools(directory: str, taskset: bool = True) -> dict[str, str]:
    """Write fake zli, zstd (and taskset) into directory; their paths by name."""
    tools = {
        "zli": _write(
            os.path.join(directory, "zli"), _constants("zli") + _COMMON + _ZLI
        ),
        "zstd": _write(
            os.path.join(directory, "zstd"), _constants("zstd") + _COMMON + _ZSTD
        ),
    }
    if taskset:
        tools["taskset"] = _write(os.path.join(directory, "taskset"), _TASKSET)
    return tools


def tool_env(
    directory: str,
    log: str | None = None,
    mode: str | None = None,
    sizes: dict[int, int] | None = None,
    time: object = None,
    **extra: str,
) -> dict[str, str]:
    """Environment variables for the fakes: PATH starts with directory."""
    env = {"PATH": directory + os.pathsep + os.environ.get("PATH", "")}
    if log is not None:
        env["FAKE_LOG"] = log
    if mode is not None:
        env["FAKE_MODE"] = mode
    if sizes is not None:
        env["FAKE_SIZES"] = json.dumps({str(k): v for k, v in sizes.items()})
    if time is not None:
        env["FAKE_TIME"] = json.dumps(time)
    env.update(extra)
    return env


def read_log(path: str, tool: str | None = None) -> list[dict]:
    """The logged calls, oldest first; only ``tool``'s when given."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return [e for e in entries if tool is None or e.get("tool") == tool]


def expected_size(tool: str, level: int, input_bytes: int, long: bool = False) -> int:
    """The compressed size the fakes report by default."""
    if tool == "zli":
        ratio = 3.0 + 0.15 * level
    elif level > 0:
        ratio = 2.0 + 0.1 * level
    else:
        ratio = 2.0 / (1 + 0.05 * -level)
    if long:
        ratio *= 1.02
    return max(1, int(input_bytes / ratio))


class FakeProc:
    """A fake /proc and /sys tree whose tick counters advance on sleep().

    ``loads`` lists, per sleep() call, the busy percent of each core
    ({core: percent}); the last entry repeats. Pass ``sleep`` to CpuProbe.
    """

    HZ = 100

    def __init__(
        self,
        root: str,
        cores: int = 4,
        loads: list[dict[int, float]] | None = None,
        model: str | None = "Fake CPU 9000 @ 3.00GHz",
        governor: str | None = "performance",
    ) -> None:
        self.root = root
        self.cores = cores
        self.loads = list(loads or [{}])
        self.sleeps: list[float] = []
        self.busy = {c: 1000 * (c + 1) for c in range(cores)}
        self.total = {c: 50000 for c in range(cores)}
        os.makedirs(os.path.join(root, "proc"), exist_ok=True)
        if model is not None:
            with open(os.path.join(root, "proc", "cpuinfo"), "w") as f:
                f.write(
                    "".join(
                        f"processor\t: {c}\nvendor_id\t: FakeVendor\n"
                        f"model name\t: {model}\ncpu MHz\t\t: 3000.000\n\n"
                        for c in range(cores)
                    )
                )
        if governor is not None:
            for c in range(cores):
                directory = os.path.join(
                    root, "sys", "devices", "system", "cpu", f"cpu{c}", "cpufreq"
                )
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, "scaling_governor"), "w") as f:
                    f.write(governor + "\n")
        self.write()

    def write(self) -> None:
        lines = []
        for c in range(self.cores):
            busy, total = self.busy[c], self.total[c]
            user, system = busy - busy // 4, busy // 4
            iowait = (total - busy) // 10
            idle = total - busy - iowait
            # guest (user // 2) is part of user already: a parser that adds the
            # guest fields gets the busy share wrong.
            lines.append(
                f"cpu{c} {user} 0 {system} {idle} {iowait} 0 0 0 {user // 2} 0"
            )
        agg = [sum(int(line.split()[i]) for line in lines) for i in range(1, 11)]
        text = "cpu  " + " ".join(map(str, agg)) + "\n" + "\n".join(lines) + "\n"
        text += "intr 12345 0 0\nctxt 999\nbtime 1700000000\n"
        with open(os.path.join(self.root, "proc", "stat"), "w") as f:
            f.write(text)

    def load_now(self) -> dict[int, float]:
        return self.loads[min(len(self.sleeps), len(self.loads) - 1)]

    def advance(self, seconds: float, load: dict[int, float]) -> None:
        ticks = round(seconds * self.HZ)
        for c in range(self.cores):
            self.busy[c] += round(ticks * load.get(c, 0) / 100)
            self.total[c] += ticks
        self.write()

    def sleep(self, seconds: float) -> None:
        load = self.load_now()
        self.sleeps.append(seconds)
        self.advance(seconds, load)
