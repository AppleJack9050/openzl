# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Measure compression ratio and speed with `zli benchmark` and `zstd -b`.

Everything that starts processes or reads /proc and /sys lives here.
:func:`run_benchmark` returns the benchmark document that ``pareto.py``
describes, without its ``frontiers`` (``pareto.normalize`` computes those).

Each measured process runs on one CPU core (``taskset -c``). Before a run the
core is checked for other load, and around the run the core's busy time is
compared with the child's own CPU time, so results disturbed by other tenants
are flagged instead of silently kept.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import hashlib
import io
import math
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence

# OpenZL sizes do not always shrink as the level rises, so zli gets a wide sweep.
DEFAULT_ZLI_LEVELS = "1-9,12,15,19,22"
DEFAULT_ZSTD_LEVELS = "3,6,9,12"
ZLI_DEFAULT_LEVEL = 6
BUSY_LIMIT = 15.0
ENV_RECORDED = (
    "GLIBC_TUNABLES",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "MALLOC_ARENA_MAX",
    "MALLOC_TOP_PAD_",
    "MALLOC_TRIM_THRESHOLD_",
    "MALLOC_MMAP_THRESHOLD_",
)
ENV_REMOVED = ("ZSTD_CLEVEL", "ZSTD_NBTHREADS")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_LEVEL = 22
MAX_FAST = 50
MAX_PROFILES = 3
MAX_ROUNDS = 20
MAX_MIN_TIME = 60.0
MAX_TIMEOUT = 7 * 24 * 3600.0
MAX_ITERATIONS = 10000
MAX_ERROR = 500
MAX_NOTES = 20
SHORT_SECONDS = 0.05
BUSY_INTERVAL = 0.3
WAIT_POLLS = 30
TOOL_QUERY_TIMEOUT = 30.0
MAX_CAPTURE = 256 * 1024
"""Bytes kept from each end of a child's stdout and stderr."""
FALLBACK_TEXT = "Encountered warnings during operation"
SIZES_DIFFER = "sizes differ between runs"
SKIPPED_AFTER_TIMEOUT = "skipped after a lower level timed out"
INTERRUPTED = "interrupted"
_MIB = 1024 * 1024
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,59}")
_PROFILE_LINE = re.compile(r"^\s*-\|\s*([^\s=]+)\s*=", re.MULTILINE)
_ZSTD_VERSION = re.compile(r"Zstandard CLI\b.*?\bv(\d+(?:\.\d+)+)")
_BUILD_DIR = re.compile(r"(?:^|/)cachedObjs/([0-9A-Fa-f]+)/")
_ZLI_TOTAL = re.compile(r"(\d+) files?: (\d+) -> (\d+) \(")
_ZLI_CSV_FIELDS = ("srcSize", "compressedSize", "ctimeMs", "dtimeMs", "iters")
_ZSTD_HEADER = re.compile(r"^bench \S+ : input (\d+) bytes", re.MULTILINE)
_ZSTD_RESULT = re.compile(
    r"^(-{1,2}\d+)\s+(\d+)\s+\(\s*[\d.]+\)\s+([\d.]+) MB/s\s+([\d.]+) MB/s",
    re.MULTILINE,
)


class BenchError(Exception):
    """A setup problem: the benchmark cannot run as asked."""


# ---------------------------------------------------------------------------
# Levels


def parse_levels(spec: str, fast: bool = True) -> list[int]:
    """Levels from "fast=5,fast=1,1-9,19": zstd fast levels as negative numbers.

    With ``fast=False`` (zli's levels) fast levels are refused."""
    allow_fast = fast
    if not spec or not spec.strip():
        raise ValueError("no levels given; for example 1-9,19 or fast=5,1-19")
    levels = set()
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            raise ValueError(f"empty item in the level list {spec!r}")
        if item == "all":
            levels.update(range(1, MAX_LEVEL + 1))
            continue
        fast = item.startswith("fast=")
        if fast and not allow_fast:
            raise ValueError(
                f"{raw.strip()!r}: zli has no fast levels; fast=K is for zstd"
            )
        body = item[len("fast=") :] if fast else item
        if re.fullmatch(r"-\d+", body):
            raise ValueError(
                f"level {raw.strip()!r} is negative; use fast=K for zstd's fast "
                f"levels (fast={body[1:]} is zstd --fast={body[1:]})"
            )
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", body)
        if not match:
            raise ValueError(
                f"cannot read level {raw.strip()!r}; use N, A-B, fast=K, "
                "fast=A-B or all"
            )
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        if low > high:
            prefix = "fast=" if fast else ""
            raise ValueError(
                f"range {raw.strip()!r} goes backwards; write {prefix}{high}-{low}"
            )
        if fast:
            if low < 1 or high > MAX_FAST:
                raise ValueError(
                    f"{raw.strip()!r}: zstd's fast levels go from fast=1 to "
                    f"fast={MAX_FAST}"
                )
            levels.update(-k for k in range(low, high + 1))
            continue
        if low == 0:
            raise ValueError(
                f"0 is zli's default level ({ZLI_DEFAULT_LEVEL}) and zstd's "
                "default (3); name the level"
            )
        if high > MAX_LEVEL:
            raise ValueError(
                f"level {high} is too high: levels go up to {MAX_LEVEL} "
                "(zstd's fast levels are written fast=K)"
            )
        levels.update(range(low, high + 1))
    return sorted(levels)


def _runs(values: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for value in values:
        if runs and value == runs[-1][1] + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return [(low, high) for low, high in runs]


def format_levels(levels: Sequence[int]) -> str:
    """The inverse of :func:`parse_levels`: "fast=5,fast=1,1-9,12"."""
    items: list[str] = []
    fast = sorted(-level for level in set(levels) if level < 0)
    for low, high in reversed(_runs(fast)):
        if high - low >= 2:
            items.append(f"fast={low}-{high}")
        else:
            items.extend(f"fast={k}" for k in range(high, low - 1, -1))
    positive = sorted(level for level in set(levels) if level > 0)
    for low, high in _runs(positive):
        if high - low >= 2:
            items.append(f"{low}-{high}")
        else:
            items.extend(str(k) for k in range(low, high + 1))
    return ",".join(items)


def zli_levels(levels: Sequence[int]) -> list[int]:
    """zli has no fast levels; level 6 is always measured for the trace link."""
    return sorted({level for level in levels if level > 0} | {ZLI_DEFAULT_LEVEL})


def zstd_levels(levels: Sequence[int]) -> list[int]:
    return sorted(set(levels))


def zli_level_label(level: int) -> str:
    return f"-l {level}"


def zstd_level_label(level: int) -> str:
    return f"-{level}" if level > 0 else f"--fast={-level}"


def zstd_result_label(level: int) -> str:
    """How `zstd -b` names the level in its result line: "-3" or "--5"."""
    return f"-{level}" if level > 0 else f"--{-level}"


def iterations_for(min_time: float, c_seconds: float, d_seconds: float) -> int:
    """zli -n for a sample: at least ``min_time`` of compression, and enough
    decompression iterations for ``min_time`` unless compression would then
    take more than four times as long. ``c_seconds``/``d_seconds`` are the
    per-iteration times of the calibration run."""
    if min_time <= 0:
        return 1

    def ratio(x: float) -> float:
        # Keeps 0.3 / 0.1 from becoming 3.0000000000000004 -> 4 iterations.
        return round(x, 9)

    n = max(
        math.ceil(ratio(min_time / c_seconds)),
        min(
            math.ceil(ratio(min_time / d_seconds)),
            math.floor(ratio(4 * min_time / c_seconds)),
        ),
    )
    return max(1, min(MAX_ITERATIONS, n))


# ---------------------------------------------------------------------------
# Processes


def child_env() -> dict[str, str]:
    """The environment of every child: zstd's own defaults removed, C locale."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_REMOVED}
    env["LC_ALL"] = "C"
    return env


def recorded_env() -> dict[str, str]:
    return {k: os.environ[k] for k in ENV_RECORDED if k in os.environ}


def _text(data: bytes | None) -> str:
    return data.decode("utf-8", "replace") if data else ""


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _read_capture(f) -> bytes:
    """What a child wrote to f: all of it, or its head and tail when it is long."""
    size = f.seek(0, os.SEEK_END)
    f.seek(0)
    if size <= 2 * MAX_CAPTURE:
        return f.read()
    head = f.read(MAX_CAPTURE)
    f.seek(size - MAX_CAPTURE)
    left_out = size - 2 * MAX_CAPTURE
    return head + f"\n[{left_out:,} bytes left out]\n".encode() + f.read()


def _spawn(
    argv: list[str],
    timeout: float,
    stdout: int = subprocess.PIPE,
    preexec_fn: Callable[[], None] | None = None,
) -> tuple[int | None, str, str, bool]:
    """Run argv in its own process group: (exit status, stdout, stderr, timed out).

    The exit status is None when the program could not be started. On a timeout
    or an interrupt the whole group is killed, so helpers the tool started do
    not outlive it. Output goes to temporary files and only its head and tail
    are kept, so a tool that prints without end cannot fill the memory.
    """
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out_f if stdout == subprocess.PIPE else stdout,
                stderr=err_f,
                start_new_session=True,
                env=child_env(),
                # Only given while this is the process's only thread (_Runner).
                preexec_fn=preexec_fn,  # noqa: PLW1509
            )
        except OSError as e:
            return None, "", f"cannot run {argv[0]}: {e.strerror or e}", False
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            proc.wait()
        except BaseException:
            _kill_group(proc)
            proc.wait()
            raise
        out, err = _read_capture(out_f), _read_capture(err_f)
    return proc.returncode, _text(out), _text(err), timed_out


# Notices zli prints for --chunk-size-mb on every run; they never explain an error.
_ZLI_NOTICES = ("Chunking is not currently implemented", "Chunking is implemented")


def _output_lines(text: str) -> list[str]:
    lines = re.split(r"[\r\n]+", text.replace("\x1b[K", ""))
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(_ZLI_NOTICES)
    ]


def _short_error(reason: str, output: str = "") -> str:
    """``reason``, followed by the most telling lines of the tool's output."""
    lines = _output_lines(output)
    # An OpenZL exception ends in a stack trace; its message line says more.
    detail = [line for line in lines if line.startswith("OpenZL error string:")]
    tail = " / ".join(detail[-1:] or lines[-3:])
    message = f"{reason}: {tail}" if tail else reason
    if len(message) > MAX_ERROR:
        message = message[: MAX_ERROR - 1] + "…"
    return message


def _exit_reason(tool: str, code: int | None) -> str:
    if code is None:
        return f"{tool} could not be started"
    if code < 0:
        return f"{tool} was killed by signal {-code}"
    return f"{tool} exited with status {code}"


def _executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_tool(explicit: str | None, env_var: str, fallbacks: list[str]) -> str:
    """The absolute path of a tool: ``explicit``, else $env_var, else the first
    existing fallback, else the tool's name on PATH."""
    name = env_var.rsplit("_", 1)[-1].lower()
    for source, value in (
        (f"--{name}", explicit),
        (f"${env_var}", os.environ.get(env_var)),
    ):
        if not value:
            continue
        path = value
        if not os.path.exists(path) and os.sep not in path:
            path = shutil.which(path) or path
        if not _executable(path):
            raise BenchError(f"{source} {value}: not an executable file")
        return os.path.abspath(path)
    for path in fallbacks:
        if os.path.isfile(path):
            if not _executable(path):
                raise BenchError(f"{path} is not executable; give --{name} PATH")
            return os.path.abspath(path)
    found = shutil.which(name)
    if found:
        return os.path.abspath(found)
    raise BenchError(f"{name} not found; give --{name} PATH")


def zli_profiles(zli: str) -> list[str]:
    """The profile names `zli list-profiles` prints."""
    code, out, err, timed_out = _spawn([zli, "list-profiles"], TOOL_QUERY_TIMEOUT)
    # zli prints the list on stderr.
    names = _PROFILE_LINE.findall(out + err)
    if timed_out:
        raise BenchError(
            f"`zli list-profiles` did not finish in {TOOL_QUERY_TIMEOUT:g} s"
        )
    if code != 0 or not names:
        reason = _exit_reason("zli", code) if code != 0 else "zli listed no profiles"
        raise BenchError(
            _short_error(f"cannot list zli's profiles with {zli}: {reason}", out + err)
        )
    return names


def zstd_version(zstd: str) -> str:
    """ "1.5.7" from `zstd -V`."""
    code, out, err, _ = _spawn([zstd, "-V"], TOOL_QUERY_TIMEOUT)
    match = _ZSTD_VERSION.search(out + err)
    if code != 0 or not match:
        raise BenchError(
            _short_error(f"{zstd} does not look like the zstd command line", out + err)
        )
    return match.group(1)


def zli_version(zli: str) -> str:
    """The first line of `zli --version`, or "" when it prints nothing useful."""
    code, out, err, _ = _spawn([zli, "--version"], TOOL_QUERY_TIMEOUT)
    # zli prints its version on stderr.
    lines = _output_lines(out + err) if code == 0 else []
    return lines[0][:200] if lines else ""


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(_MIB)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def display_path(path: str) -> str:
    """A path to show: relative inside the repository, else with "~" for home.

    Symlinks are resolved, so the repository's `zli` link shows the build it
    points at (cachedObjs/<flags hash>/zli)."""
    path = os.path.realpath(path)
    if path.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(path, REPO_ROOT)
    home = os.path.expanduser("~").rstrip(os.sep)
    if home and home != "~" and path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def display_name(path: str) -> str:
    """The file name of path as text: undecodable bytes become U+FFFD."""
    name = os.path.basename(os.path.abspath(path))
    return name.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _short_display(path: str) -> str:
    return re.sub(r"(cachedObjs/[0-9A-Fa-f]{4})[0-9A-Fa-f]{5,}", r"\1…", path)


def tool_info(path: str, version: str, note: str | None = None) -> dict[str, object]:
    """A "tools" entry of the document. ``note`` (zli only) adds build details."""
    sha256, size = _sha256_file(path)
    info: dict[str, object] = {
        "path": display_path(path),
        "sha256": sha256,
        "bytes": size,
        "version": version,
    }
    if note is not None:
        match = _BUILD_DIR.search(os.path.realpath(path))
        info["build_dir"] = match.group(1) if match else None
        info["note"] = note
    return info


# ---------------------------------------------------------------------------
# CPU cores


def _percent(before: tuple[int, int], after: tuple[int, int]) -> float:
    total = after[1] - before[1]
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (after[0] - before[0]) / total))


def _cpu_list(cores: Sequence[int]) -> str:
    return ",".join(
        str(low) if low == high else f"{low}-{high}"
        for low, high in _runs(sorted(cores))
    )


class CpuProbe:
    """How busy each CPU core is, from the kernel's tick counters."""

    clk_tck: int = os.sysconf("SC_CLK_TCK")

    def __init__(
        self,
        root: str = "/",
        sleep: Callable[[float], None] = time.sleep,
        allowed: Sequence[int] | None = None,
    ) -> None:
        self.root = root
        self.sleep = sleep
        if allowed is None:
            if hasattr(os, "sched_getaffinity"):
                allowed = sorted(os.sched_getaffinity(0))
            else:
                allowed = range(os.cpu_count() or 1)
        self.allowed = set(allowed)

    def _read(self, *parts: str) -> str | None:
        try:
            with open(os.path.join(self.root, *parts), encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None

    def ticks(self) -> dict[int, tuple[int, int]]:
        """core -> (busy ticks, total ticks) since boot."""
        text = self._read("proc", "stat")
        if text is None:
            raise BenchError("cannot read /proc/stat to watch the CPU cores")
        result = {}
        for line in text.splitlines():
            match = re.match(r"cpu(\d+)\s", line)
            if not match:
                continue
            try:
                # user nice system idle iowait irq softirq steal; the guest
                # fields that may follow are already counted in user and nice.
                values = [int(v) for v in line.split()[1:9]]
            except ValueError:
                continue
            if len(values) < 4:
                continue
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[int(match.group(1))] = (total - idle, total)
        return result

    def _core_ticks(self, core: int) -> tuple[int, int]:
        ticks = self.ticks().get(core)
        if ticks is None:
            raise BenchError(f"core {core} is not listed in /proc/stat")
        return ticks

    def busy_percent(self, core: int, interval: float = 0.3) -> float:
        before = self._core_ticks(core)
        self.sleep(interval)
        return _percent(before, self._core_ticks(core))

    def quietest(self, interval: float = 0.4) -> tuple[int, float]:
        """(core, busy percent) of the least busy allowed core.

        On a tie a core other than 0 wins: interrupts and housekeeping tend to
        land on core 0."""
        before = self.ticks()
        self.sleep(interval)
        after = self.ticks()
        cores = [c for c in self.allowed if c in before and c in after]
        if not cores:
            raise BenchError("none of this process's CPU cores is in /proc/stat")
        best = min(cores, key=lambda c: (_percent(before[c], after[c]), c == 0, c))
        return best, _percent(before[best], after[best])

    def cpu_model(self) -> str | None:
        for line in (self._read("proc", "cpuinfo") or "").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name" and value.strip():
                return value.strip()
        return None

    def governor(self, core: int) -> str | None:
        text = self._read(
            "sys",
            "devices",
            "system",
            "cpu",
            f"cpu{core}",
            "cpufreq",
            "scaling_governor",
        )
        if text is None:
            return None
        return text.strip() or None


# ---------------------------------------------------------------------------
# Configuration


@dataclasses.dataclass
class BenchConfig:
    input: str
    zli: str
    zstd: str
    profiles: list[str]
    profile_arg: str | None = None
    chunk_size_mb: int | None = None
    compressor: str | None = None
    """Reserved: trained compressors are not measured yet."""
    zstd_long: int | None = None
    zli_levels: list[int] = dataclasses.field(
        default_factory=lambda: parse_levels(DEFAULT_ZLI_LEVELS, fast=False)
    )
    """zli levels; level 6 is always added (see zli_levels())."""
    zstd_levels: list[int] = dataclasses.field(
        default_factory=lambda: parse_levels(DEFAULT_ZSTD_LEVELS)
    )
    rounds: int = 3
    min_time: float = 0.5
    quick: bool = False
    core: str | int = "auto"
    timeout: float = 900.0
    build_note: str = ""


def _zli_args(config: BenchConfig, profile: str) -> list[str]:
    args = ["-p", profile]
    if config.profile_arg is not None:
        args += ["--profile-arg", config.profile_arg]
    if config.chunk_size_mb is not None:
        args += ["--chunk-size-mb", str(config.chunk_size_mb)]
    return args


def series_for(config: BenchConfig) -> list[dict[str, object]]:
    series: list[dict[str, object]] = []
    for slot, profile in enumerate(config.profiles):
        series.append(
            {
                "id": f"zli:{profile}",
                "tool": "zli",
                "label": f"OpenZL -p {profile}",
                "args": _zli_args(config, profile),
                "slot": slot,
            }
        )
    series.append(
        {
            "id": "zstd",
            "tool": "zstd",
            "label": "zstd",
            "args": ["--single-thread"],
            "slot": 0,
        }
    )
    if config.zstd_long is not None:
        window = config.zstd_long
        series.append(
            {
                "id": f"zstd:long{window}",
                "tool": "zstd",
                "label": f"zstd --long={window}",
                "args": ["--single-thread", f"--long={window}"],
                "slot": 1,
            }
        )
    return series


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def check_setup(config: BenchConfig, probe: CpuProbe) -> None:
    """Refuse a configuration that cannot give results, before any timed work."""
    path = config.input
    if os.path.isdir(path):
        raise BenchError(f"{path} is a directory; --bench needs a file")
    if not os.path.exists(path):
        raise BenchError(f"{path}: no such file")
    if not os.path.isfile(path):
        raise BenchError(f"{path} is not a regular file; --bench needs a file")
    try:
        with open(path, "rb"):
            pass
    except OSError as e:
        raise BenchError(f"cannot read {path}: {e.strerror or e}") from None
    if os.path.getsize(path) == 0:
        raise BenchError(f"{path} is empty; there is nothing to measure")

    profiles = list(config.profiles)
    if not 1 <= len(profiles) <= MAX_PROFILES:
        raise BenchError(f"give 1 to {MAX_PROFILES} zli profiles with -p")
    for profile in profiles:
        if profiles.count(profile) > 1:
            raise BenchError(f"profile {profile} is given twice")
        if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
            raise BenchError(f"{profile!r} is not a zli profile name")
    if config.compressor is not None:
        raise BenchError("trained compressors cannot be benchmarked yet; use -p")
    if not config.zstd_levels:
        raise BenchError("no zstd levels to measure; give --zstd-levels")
    for level in config.zli_levels:
        if not _is_int(level) or not 1 <= level <= MAX_LEVEL:
            raise BenchError(f"zli level {level!r} is out of range")
    for level in config.zstd_levels:
        if not _is_int(level) or not (-MAX_FAST <= level <= MAX_LEVEL) or level == 0:
            raise BenchError(f"zstd level {level!r} is out of range")
    if not _is_int(config.rounds) or not 1 <= config.rounds <= MAX_ROUNDS:
        raise BenchError(f"--rounds must be between 1 and {MAX_ROUNDS}")
    if not _is_number(config.min_time) or not 0 <= config.min_time <= MAX_MIN_TIME:
        raise BenchError(f"--min-time must be between 0 and {MAX_MIN_TIME:g} seconds")
    if not _is_number(config.timeout) or not 0 < config.timeout <= MAX_TIMEOUT:
        raise BenchError("--timeout must be a positive number of seconds")
    if config.chunk_size_mb is not None and (
        not _is_int(config.chunk_size_mb) or config.chunk_size_mb < 1
    ):
        raise BenchError("--chunk-size-mb must be a positive whole number")
    if config.zstd_long is not None and (
        not _is_int(config.zstd_long) or not 10 <= config.zstd_long <= 31
    ):
        raise BenchError("--zstd-long takes a window log from 10 to 31")

    for name, tool in (("zli", config.zli), ("zstd", config.zstd)):
        if not _executable(tool):
            raise BenchError(
                f"{name} {tool} is not an executable file; give --{name} PATH"
            )
    available = zli_profiles(config.zli)
    for profile in profiles:
        if profile not in available:
            raise BenchError(
                f"zli has no profile {profile!r}; it has: {', '.join(available)}"
            )
    zstd_version(config.zstd)

    core = config.core
    if _is_int(core):
        if core not in probe.allowed:
            raise BenchError(
                f"core {core} is not available to this process; use one of "
                f"{_cpu_list(sorted(probe.allowed))}, auto or none"
            )
    elif core not in ("auto", "none"):
        raise BenchError("--core takes a core number, auto or none")


# ---------------------------------------------------------------------------
# Running one process


@dataclasses.dataclass
class _Run:
    argv: list[str]
    code: int | None
    out: str
    err: str
    timed_out: bool
    contended: bool
    wall: float


class _Runner:
    """Starts children pinned to one core and watches that core for other load."""

    def __init__(
        self,
        config: BenchConfig,
        probe: CpuProbe,
        clock: Callable[[], float] = time.monotonic,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.probe = probe
        self.clock = clock
        self.progress = progress
        self.notes: list[str] = []
        self.core: int | None = None
        self.pin = "none"
        self.taskset: str | None = None
        if config.core == "none":
            self.choice = "none"
            return
        self.choice = "auto" if config.core == "auto" else "fixed"
        self.taskset = shutil.which("taskset")
        if self.taskset:
            self.pin = "taskset"
        # preexec_fn is not safe once other threads exist.
        elif threading.active_count() == 1:
            self.pin = "affinity"
        else:
            self.notes.append(
                "taskset was not found and other threads were running, so the "
                "runs were not pinned to a core."
            )
            return
        if self.choice == "auto":
            self.core = probe.quietest()[0]
        else:
            self.core = int(config.core)

    def say(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def _prefix(self) -> list[str]:
        if self.pin == "taskset" and self.core is not None:
            return [self.taskset, "-c", str(self.core)]
        return []

    def _preexec(self) -> Callable[[], None] | None:
        if self.pin != "affinity" or self.core is None:
            return None
        core = self.core
        return lambda: os.sched_setaffinity(0, {core})

    def _settle(self) -> float:
        """Wait (up to WAIT_POLLS s) for the core to be quiet; the last reading."""
        core = self.core
        busy = self.probe.busy_percent(core, BUSY_INTERVAL)
        if busy <= BUSY_LIMIT:
            return busy
        self.say(f"core {core} is busy ({busy:.0f}%); waiting up to {WAIT_POLLS} s")
        for _ in range(WAIT_POLLS):
            if self.choice == "auto":
                other, quiet = self.probe.quietest(1.0)
                if quiet <= BUSY_LIMIT:
                    if other != self.core:
                        self._note(
                            f"Switched from core {self.core} to core {other} "
                            f"because core {self.core} was busy."
                        )
                        self.say(f"switched to core {other} ({quiet:.0f}% busy)")
                        self.core = other
                    return quiet
            else:
                busy = self.probe.busy_percent(core, 1.0)
                if busy <= BUSY_LIMIT:
                    return busy
        self.say(f"core {self.core} is still busy; measuring anyway (marked)")
        return busy

    def run(
        self, command: list[str], measure: bool = True, stdout: int = subprocess.PIPE
    ) -> _Run:
        watch = measure and self.core is not None
        busy = self._settle() if watch else 0.0
        argv = self._prefix() + command
        ticks = self.probe._core_ticks(self.core) if watch else None
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = self.clock()
        code, out, err, timed_out = _spawn(
            argv, self.config.timeout, stdout, self._preexec()
        )
        wall = max(0.0, self.clock() - start)
        contended = False
        if ticks is not None:
            after = self.probe._core_ticks(self.core)
            used = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_s = (used.ru_utime - usage.ru_utime) + (
                used.ru_stime - usage.ru_stime
            )
            foreign_s = max(0.0, (after[0] - ticks[0]) / self.probe.clk_tck - child_s)
            contended = busy > BUSY_LIMIT or foreign_s > max(0.05, 0.05 * wall)
        return _Run(argv, code, out, err, timed_out, contended, wall)


# ---------------------------------------------------------------------------
# Parsing tool output


def _csv_number(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"zli's CSV has {name}={value!r}, not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"zli's CSV has {name}={value!r}, not a number")
    return number


def _count_text(value: float) -> str:
    return f"{int(value):,}" if value == int(value) else repr(value)


def parse_zli_csv(text: str, input_bytes: int, iterations: int) -> dict[str, float]:
    """bytes, c_speed, d_speed (MB/s) and the timed c_seconds/d_seconds of one
    `zli benchmark --output-csv` file. Raises ValueError when it cannot be used."""
    reader = csv.DictReader(io.StringIO(text))
    try:
        fields = [f for f in (reader.fieldnames or []) if f]
        missing = [f for f in _ZLI_CSV_FIELDS if f not in fields]
        if missing:
            raise ValueError(f"zli's CSV has no {', '.join(missing)} column")
        rows = list(reader)
    except csv.Error as e:
        raise ValueError(f"zli's CSV is garbled: {e}") from None
    if len(rows) != 1:
        raise ValueError(f"zli's CSV has {len(rows)} data rows instead of 1")
    row = rows[0]
    extra = row.pop(None, None)  # type: ignore[call-overload]
    given = sum(1 for v in row.values() if v is not None)
    # zli leaves out the trailing "path" value today.
    if extra or given < len(fields) - 1:
        count = given + len(extra or [])
        raise ValueError(f"zli's CSV row has {count} values for {len(fields)} columns")
    src = _csv_number(row, "srcSize")
    size = _csv_number(row, "compressedSize")
    ctime = _csv_number(row, "ctimeMs")
    dtime = _csv_number(row, "dtimeMs")
    iters = _csv_number(row, "iters")
    if src <= 0 or src != input_bytes:
        raise ValueError(
            f"zli's CSV has srcSize {_count_text(src)}, but the input is "
            f"{input_bytes:,} bytes"
        )
    if iters == 0:
        raise ValueError("zli ran 0 iterations (it then reports 0 MB/s)")
    if iters != iterations:
        raise ValueError(
            f"zli's CSV has iters {_count_text(iters)}, but -n {iterations} was given"
        )
    if size <= 0 or size != int(size):
        raise ValueError(f"zli's CSV has compressedSize {_count_text(size)}")
    if ctime <= 0 or dtime <= 0:
        raise ValueError(
            f"zli's CSV has a time that is not positive (ctimeMs {ctime:g}, "
            f"dtimeMs {dtime:g})"
        )
    c_seconds = ctime / 1000
    d_seconds = dtime / 1000
    return {
        "bytes": int(size),
        "c_speed": src / 1e6 / (c_seconds / iterations),
        "d_speed": src / 1e6 / (d_seconds / iterations),
        "c_seconds": c_seconds,
        "d_seconds": d_seconds,
    }


def parse_zli_output(text: str) -> int | None:
    """The compressed size in zli benchmark's last "1 files: A -> B (" line."""
    last = None
    for line in re.split(r"[\r\n]", text.replace("\x1b[K", "")):
        match = _ZLI_TOTAL.search(line)
        if match:
            last = int(match.group(3))
    return last


def parse_zstd_output(
    text: str, input_bytes: int, level: int
) -> tuple[int, float, float]:
    """(bytes, compression MB/s, decompression MB/s) from `zstd -b -q` stdout."""
    text = text.replace("\r", "\n")
    header = _ZSTD_HEADER.search(text)
    if not header:
        raise ValueError("zstd printed no benchmark header")
    if int(header.group(1)) != input_bytes:
        raise ValueError(
            f"zstd read {int(header.group(1)):,} bytes, but the input is "
            f"{input_bytes:,} bytes"
        )
    label = zstd_result_label(level)
    results = list(_ZSTD_RESULT.finditer(text))
    found = [m for m in results if m.group(1) == label]
    if not found:
        others = ", ".join(m.group(1) for m in results)
        raise ValueError(
            f"zstd printed no result for level {label}"
            + (f" (only for {others})" if others else "")
        )
    match = found[-1]
    try:
        size = int(match.group(2))
        c_speed = float(match.group(3))
        d_speed = float(match.group(4))
    except ValueError:
        raise ValueError(f"cannot read zstd's result line {match.group(0)!r}") from None
    if size <= 0 or c_speed <= 0 or d_speed <= 0:
        raise ValueError(f"zstd reported an empty result: {match.group(0)!r}")
    return size, c_speed, d_speed


# ---------------------------------------------------------------------------
# The sweep


@dataclasses.dataclass
class _Sample:
    bytes: int
    c_speed: float
    d_speed: float
    c_short: bool
    d_short: bool
    contended: bool


@dataclasses.dataclass
class _Point:
    series: dict[str, object]
    level: int
    status: str | None = None
    """None while the point is still being measured."""
    error: str | None = None
    samples: list[_Sample] = dataclasses.field(default_factory=list)
    size: int | None = None
    iterations: int | None = None
    fallback: bool = False
    finished: bool = False
    """No more samples: a lower level of the series timed out."""
    command: list[str] = dataclasses.field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.series['id']}/{self.level}"

    @property
    def tool(self) -> str:
        return str(self.series["tool"])

    @property
    def level_label(self) -> str:
        if self.tool == "zli":
            return zli_level_label(self.level)
        return zstd_level_label(self.level)

    def fail(self, status: str, error: str) -> None:
        self.status = status
        self.error = error[:MAX_ERROR]


def _significant(value: float) -> float:
    return float(f"{value:.6g}")


class _Benchmark:
    def __init__(
        self,
        config: BenchConfig,
        probe: CpuProbe,
        workdir: str,
        progress: Callable[[str], None] | None,
        clock: Callable[[], float],
        input_bytes: int,
        tools: dict[str, dict[str, object]],
    ) -> None:
        self.config = config
        self.workdir = os.path.abspath(workdir)
        self.input = os.path.abspath(config.input)
        self.zli = os.path.abspath(config.zli)
        self.zstd = os.path.abspath(config.zstd)
        self.input_bytes = input_bytes
        self.tools = tools
        self.runner = _Runner(config, probe, clock, progress)
        self.series = series_for(config)
        self.points: list[_Point] = []
        zli = zli_levels(config.zli_levels)
        zstd = zstd_levels(config.zstd_levels)
        for level in sorted(set(zli) | set(zstd)):
            for series in self.series:
                if level in (zli if series["tool"] == "zli" else zstd):
                    self.points.append(_Point(series, level))
        self.rounds = 1 if config.quick else config.rounds
        self.label_width = max(len(str(s["label"])) for s in self.series)

    def display(self, argv: list[str]) -> list[str]:
        names = {
            self.zli: str(self.tools["zli"]["path"]),
            self.zstd: str(self.tools["zstd"]["path"]),
            self.input: display_name(self.input),
        }
        if self.runner.taskset:
            names[self.runner.taskset] = "taskset"
        shown = []
        for arg in argv:
            if arg in names:
                shown.append(names[arg])
            elif arg.startswith(self.workdir + os.sep):
                shown.append("<tmp>" + arg[len(self.workdir) :])
            else:
                shown.append(arg)
        return shown

    def header(self) -> str:
        runner = self.runner
        if runner.core is None:
            where = "no pinned core"
        else:
            where = f"core {runner.core} ({runner.choice})"
        rounds = "1 round (quick)" if self.config.quick else f"{self.rounds} rounds"
        return (
            f"Benchmark: {len(self.points)} points x {rounds} on {where}; "
            f"zli {_short_display(str(self.tools['zli']['path']))}, "
            f"zstd {self.tools['zstd']['version']}"
        )

    # -- one sample --------------------------------------------------------

    def _zli_sample(
        self, point: _Point, iterations: int, quiet: bool
    ) -> _Sample | None:
        csv_path = os.path.join(self.workdir, "zli.csv")
        _remove(csv_path)
        command = [self.zli, "benchmark", self.input]
        command += [str(a) for a in point.series["args"]]  # type: ignore[union-attr]
        command += ["-l", str(point.level), "-n", str(iterations)]
        if quiet:
            # In permissive mode zli prints its fallback warnings inside the timed
            # loop, and they pile up: iteration k prints k + 1 of them. Errors
            # still print at this level.
            command += ["-v", "1"]
        command += ["--output-csv", csv_path]
        run = self.runner.run(command)
        point.command = self.display(run.argv)
        output = run.out + run.err
        if FALLBACK_TEXT in output:
            point.fallback = True
        if run.timed_out:
            point.fail("timeout", f"timed out after {self.config.timeout:g} s")
            return None
        if run.code != 0:
            point.fail("failed", _short_error(_exit_reason("zli", run.code), output))
            return None
        try:
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                csv_text = f.read()
        except OSError:
            point.fail("failed", _short_error("zli wrote no CSV file", output))
            return None
        try:
            parsed = parse_zli_csv(csv_text, self.input_bytes, iterations)
        except ValueError as e:
            point.fail("failed", str(e))
            return None
        printed = parse_zli_output(output)
        if printed is not None and printed != parsed["bytes"]:
            point.fail(
                "failed",
                f"zli printed {printed:,} compressed bytes but its CSV says "
                f"{int(parsed['bytes']):,}",
            )
            return None
        return _Sample(
            bytes=int(parsed["bytes"]),
            c_speed=parsed["c_speed"],
            d_speed=parsed["d_speed"],
            c_short=parsed["c_seconds"] < SHORT_SECONDS,
            d_short=parsed["d_seconds"] < SHORT_SECONDS,
            contended=run.contended,
        )

    def _zstd_sample(self, point: _Point) -> _Sample | None:
        level = point.level
        seconds = 0 if self.config.quick else max(1, math.ceil(self.config.min_time))
        command = [self.zstd, "--single-thread", "-q"]
        if level >= 20:
            command.append("--ultra")
        command += [
            str(a)
            for a in point.series["args"]
            if a != "--single-thread"  # type: ignore[union-attr]
        ]
        if level > 0:
            command += [f"-b{level}", f"-e{level}", f"-i{seconds}"]
        else:
            command += [f"--fast={-level}", "-b", f"-i{seconds}"]
        command += ["--", self.input]
        run = self.runner.run(command)
        point.command = self.display(run.argv)
        if run.timed_out:
            point.fail("timeout", f"timed out after {self.config.timeout:g} s")
            return None
        if run.code != 0:
            point.fail(
                "failed",
                _short_error(_exit_reason("zstd", run.code), run.out + run.err),
            )
            return None
        try:
            size, c_speed, d_speed = parse_zstd_output(run.out, self.input_bytes, level)
        except ValueError as e:
            point.fail("failed", _short_error(str(e), run.err))
            return None

        def timed(speed: float) -> float:
            # zstd -iT times each direction for at least T seconds; -i0 is one pass.
            return seconds if seconds else self.input_bytes / 1e6 / speed

        return _Sample(
            bytes=size,
            c_speed=c_speed,
            d_speed=d_speed,
            c_short=timed(c_speed) < SHORT_SECONDS,
            d_short=timed(d_speed) < SHORT_SECONDS,
            contended=run.contended,
        )

    def _keep(self, point: _Point, sample: _Sample | None) -> bool:
        """Check a sample's size against the point's earlier runs."""
        if sample is None:
            return False
        if point.size is None:
            point.size = sample.bytes
        elif sample.bytes != point.size:
            point.fail(
                "failed",
                f"{SIZES_DIFFER} ({point.size:,} and {sample.bytes:,} bytes)",
            )
            return False
        return True

    def _measure(self, point: _Point) -> None:
        self._take_sample(point)
        if point.status == "timeout":
            for other in self.points:
                if (
                    other.series is point.series
                    and other.level > point.level
                    and other.status is None
                ):
                    if other.samples:
                        other.finished = True
                    else:
                        other.status = "skipped"
                        other.error = SKIPPED_AFTER_TIMEOUT

    def _take_sample(self, point: _Point) -> None:
        if point.tool == "zstd":
            sample = self._zstd_sample(point)
        else:
            # One iteration at the normal log level shows whether zli falls back
            # (point.fallback); the measured runs are quiet.
            quiet = not self.config.quick
            if point.iterations is None:
                if self.config.quick:
                    point.iterations = 1
                else:
                    calibration = self._zli_sample(point, 1, quiet=False)
                    if not self._keep(point, calibration):
                        return
                    point.iterations = iterations_for(
                        self.config.min_time,
                        self.input_bytes / 1e6 / calibration.c_speed,
                        self.input_bytes / 1e6 / calibration.d_speed,
                    )
            sample = self._zli_sample(point, point.iterations, quiet)
        if self._keep(point, sample):
            point.samples.append(sample)

    def _report(self, index: int, total: int, point: _Point) -> None:
        width = len(str(total))
        head = (
            f"[{index:>{width}}/{total}] "
            f"{point.series['label']!s:<{self.label_width}} {point.level_label:<9}"
        )
        if point.status is not None:
            self.runner.say(f"{head} {point.status}: {point.error}")
            return
        sample = point.samples[-1]
        line = (
            f"{head} {sample.bytes:>11,} {self.input_bytes / sample.bytes:6.2f}x "
            f"{sample.c_speed:8.1f} MB/s {sample.d_speed:8.1f} MB/s"
        )
        if sample.contended:
            line += "  core busy"
        self.runner.say(line)

    def sweep(self) -> bool:
        """Measure every point; False when interrupted."""
        total = len(self.points) * self.rounds
        index = 0
        try:
            self.runner.say(self.header())
            for _ in range(self.rounds):
                for point in self.points:
                    index += 1
                    if point.status is not None or point.finished:
                        continue
                    self._measure(point)
                    self._report(index, total, point)
        except KeyboardInterrupt:
            return False
        finally:
            _remove(os.path.join(self.workdir, "zli.csv"))
        return True

    # -- the document ------------------------------------------------------

    def point_doc(self, point: _Point, interrupted: bool) -> dict[str, object]:
        if point.status is None:
            if point.samples:
                point.status = "ok"
            else:
                point.fail("skipped", INTERRUPTED if interrupted else "not measured")
        ok = point.status == "ok"
        flags: dict[str, list[str]] = {"c": [], "d": [], "both": []}
        doc: dict[str, object] = {
            "id": point.id,
            "series": point.series["id"],
            "level": point.level,
            "level_label": point.level_label,
            "status": point.status,
            "error": None if ok else point.error,
            "bytes": point.size if ok else None,
            "c_speed": None,
            "d_speed": None,
            "c_samples": [],
            "d_samples": [],
            "flags": flags,
            "command": point.command,
        }
        if ok:
            for axis in ("c", "d"):
                speeds = [getattr(s, f"{axis}_speed") for s in point.samples]
                best = point.samples[speeds.index(max(speeds))]
                doc[f"{axis}_speed"] = _significant(max(speeds))
                doc[f"{axis}_samples"] = [_significant(v) for v in speeds]
                if best.contended:
                    flags[axis].append("contended")
                # A quick run is short by design; "quick" already says so.
                if getattr(best, f"{axis}_short") and not self.config.quick:
                    flags[axis].append("short")
        if self.config.quick:
            flags["both"].append("quick")
        if self.runner.core is None:
            flags["both"].append("unpinned")
        if point.fallback:
            flags["both"].append("fallback")
        return doc


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_link() -> dict[str, object]:
    return {
        "state": "none",
        "series": None,
        "point": None,
        "frame_bytes": None,
        "stream_bytes": None,
        "given_input_bytes": None,
        "given_stream_bytes": None,
        "verified": None,
    }


def run_benchmark(
    config: BenchConfig,
    probe: CpuProbe,
    workdir: str,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Measure every series at every level; the document without "frontiers"."""
    start = clock()
    created = _now()
    before = os.stat(config.input)
    sha256, input_bytes = _sha256_file(config.input)
    tools = {
        "zli": tool_info(config.zli, zli_version(config.zli), config.build_note or ""),
        "zstd": tool_info(config.zstd, zstd_version(config.zstd)),
    }
    bench = _Benchmark(config, probe, workdir, progress, clock, input_bytes, tools)
    finished = bench.sweep()
    try:
        after = os.stat(config.input)
    except OSError:
        after = None
    if after is None or (after.st_size, after.st_mtime_ns) != (
        before.st_size,
        before.st_mtime_ns,
    ):
        raise BenchError("INPUT changed during the benchmark")

    points = [bench.point_doc(p, not finished) for p in bench.points]
    runner = bench.runner
    return {
        "format": "codec_reviewer.benchmark",
        "version": 1,
        "complete": finished and all(p["status"] != "skipped" for p in points),
        "stopped": None if finished else "interrupted",
        "created": created,
        "elapsed_s": round(max(0.0, clock() - start), 3),
        "input": {
            "name": display_name(config.input),
            "bytes": input_bytes,
            "sha256": sha256,
        },
        "machine": {
            "cpu": probe.cpu_model(),
            "logical_cpus": os.cpu_count(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "governor": probe.governor(runner.core)
            if runner.core is not None
            else None,
            "core": runner.core,
            "core_choice": runner.choice,
            "pin": runner.pin if runner.core is not None else "none",
        },
        "tools": tools,
        "settings": {
            "levels": f"zli {format_levels(zli_levels(config.zli_levels))}; "
            f"zstd {format_levels(config.zstd_levels)}",
            "rounds": bench.rounds,
            "min_time_s": config.min_time,
            "quick": config.quick,
            "timeout_s": config.timeout,
            "env": recorded_env(),
        },
        "series": bench.series,
        "points": points,
        "trace_link": _empty_link(),
        "notes": runner.notes[:MAX_NOTES],
    }


# ---------------------------------------------------------------------------
# The recorded trace


@dataclasses.dataclass
class Recording:
    trace_path: str | None
    frame_bytes: int | None
    verified: bool | None
    fallback: bool
    error: str | None


def free_bytes(path: str) -> int:
    """Free disk space at path (0 when unknown)."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def record_trace(
    config: BenchConfig, workdir: str, probe: CpuProbe, profile: str | None = None
) -> Recording:
    """Compress INPUT once with a trace, pinned like the benchmark but untimed,
    and check that the frame decompresses back to INPUT."""
    profile = profile or config.profiles[0]
    runner = _Runner(config, probe)
    source = os.path.abspath(config.input)
    frame = os.path.join(os.path.abspath(workdir), "trace.zl")
    trace = os.path.join(os.path.abspath(workdir), "trace.cbor")
    restored = os.path.join(os.path.abspath(workdir), "trace.out")
    for path in (frame, trace, restored):
        _remove(path)
    command = [config.zli, "compress", source] + _zli_args(config, profile)
    command += ["-o", frame, "--trace", trace, "-f"]
    # zli prints the DOT graph on stdout; the CBOR file has the same content.
    run = runner.run(command, measure=False, stdout=subprocess.DEVNULL)
    fallback = FALLBACK_TEXT in run.err
    trace_path = trace if os.path.isfile(trace) and os.path.getsize(trace) else None
    if run.timed_out:
        error = f"zli compress timed out after {config.timeout:g} s"
        return Recording(trace_path, None, None, fallback, error)
    if run.code != 0 or not os.path.isfile(frame):
        reason = _exit_reason("zli compress", run.code)
        if run.code == 0:
            reason = "zli compress wrote no frame"
        return Recording(
            trace_path, None, None, fallback, _short_error(reason, run.err)
        )
    frame_bytes = os.path.getsize(frame)
    error = None if trace_path else _short_error("zli compress wrote no trace", run.err)

    verified = None
    input_sha, input_bytes = _sha256_file(source)
    if free_bytes(workdir) >= 2 * input_bytes + 64 * _MIB:
        check = runner.run(
            [config.zli, "decompress", frame, "-o", restored, "-f"], measure=False
        )
        verified = (
            check.code == 0
            and not check.timed_out
            and os.path.isfile(restored)
            and _sha256_file(restored)[0] == input_sha
        )
        _remove(restored)
    return Recording(trace_path, frame_bytes, verified, fallback, error)


def _series_label(doc: dict[str, object], series_id: str) -> str:
    for series in doc.get("series", []):  # type: ignore[union-attr]
        if series.get("id") == series_id:
            return str(series.get("label") or series_id)
    return series_id


def _bytes_text(value: object) -> str:
    return f"{value:,} B" if _is_int(value) else "unknown size"


def link_trace(
    doc: dict[str, object],
    series_id: str | None,
    recording: Recording | None,
    recorded_totals: tuple[int, int] | None,
    given_totals: tuple[int, int] | None = None,
) -> None:
    """Say how the reviewed trace relates to the measured points (doc["trace_link"])."""
    link = _empty_link()
    notes: list[str] = doc.setdefault("notes", [])  # type: ignore[assignment]
    doc["trace_link"] = link
    if series_id is None or (recording is None and given_totals is None):
        return
    label = _series_label(doc, series_id)
    point = next(
        (
            p
            for p in doc.get("points", [])  # type: ignore[union-attr]
            if p.get("series") == series_id
            and p.get("level") == ZLI_DEFAULT_LEVEL
            and p.get("status") == "ok"
        ),
        None,
    )
    frame_bytes = recording.frame_bytes if recording else None
    link.update(
        series=series_id,
        point=point["id"] if point else None,
        frame_bytes=frame_bytes,
        stream_bytes=recorded_totals[1] if recorded_totals else None,
        given_input_bytes=given_totals[0] if given_totals else None,
        given_stream_bytes=given_totals[1] if given_totals else None,
        verified=recording.verified if recording else None,
    )
    level = zli_level_label(ZLI_DEFAULT_LEVEL)
    if point is None:
        link["state"] = "no_point"
        notes.append(
            f"{label} has no measured {level} point, so the review's trace "
            f"(frame {_bytes_text(frame_bytes)}) is not marked on the chart."
        )
    elif frame_bytes != point["bytes"]:
        link["state"] = "frame_mismatch"
        if recording is not None and recording.error and frame_bytes is None:
            notes.append(
                f"Recording the trace failed ({recording.error}), so it is not "
                f"matched with the {label} {level} point ({_bytes_text(point['bytes'])})."
            )
        else:
            notes.append(
                f"The recorded {label} frame is {_bytes_text(frame_bytes)} but the "
                f"{level} point is {_bytes_text(point['bytes'])}: the profile picks "
                f"its own settings, so the review is not exactly a point on the chart."
            )
    elif given_totals is not None and tuple(given_totals) != (
        tuple(recorded_totals) if recorded_totals else None
    ):
        link["state"] = "mismatch"
        given = f"{_bytes_text(given_totals[0])} in, {_bytes_text(given_totals[1])} in streams"
        if recorded_totals:
            recorded = (
                f"{_bytes_text(recorded_totals[0])} in, "
                f"{_bytes_text(recorded_totals[1])} in streams"
            )
        else:
            recorded = "no readable trace"
        notes.append(
            f"The given trace ({given}) is not what these settings produce with "
            f"{label} ({recorded}), so no point is marked as the traced run."
        )
    else:
        link["state"] = "linked"
    if recording is not None and recording.verified is False:
        notes.append(
            f"The frame recorded with {label} did not decompress back to the input."
        )
