# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduler behind run_suite.sh: the gluon MoE pytest suite as balanced units.

    run_suite.py --out DIR --gpus "4 5 6 7" [--plan] [test_file ...]
    run_suite.py --weigh RUN        # rewrites suite_weights.json from RUN

Run from the repo root (run_suite.sh does). Collects the node ids of every file,
keeps light files whole and cuts heavy ones into contiguous chunks of about
UNIT_SECONDS (default 60) of estimated test time. Units run longest first from one
queue on JOBS_PER_GPU (default 8) workers per GPU, each pinned with
HIP_VISIBLE_DEVICES. UNIT_TIMEOUT (default 1800 s) bounds one unit and
COLLECT_TIMEOUT (default 300 s) one collection. A failed collection is retried once
and the tail of its output printed; a file that still fails runs whole, and counts
by its own pytest summary if that run is clean.

Estimates come from suite_weights.json: the seconds of each test, listed per test
function in collection order. A function whose parameters changed costs its old
mean per test; one the table lacks costs its file's mean, or the suite's. To
refresh the table, run with PYTEST_ADDOPTS='--durations=0 --durations-min=0' and
--weigh the result: RUN is its directory or its run_suite.sh label. The table is
replaced only once the new one is complete. On HIP 4-7 the suite takes about 5
minutes cold, bounded by its slowest single test (~280 s), and 1.5 minutes warm.

Each unit writes DIR/<file>[.NN].log, whose last line is the pytest summary. A
unit that crashes, times out or leaves tests unaccounted for gets an ERROR line
per test and an "N error" summary appended, so compare_suite.py fails on it. A
unit that fails with out of GPU memory is rerun once, one worker per GPU, after
the rest; its first log stays as <unit>.log.oom. DIR/units.json records each
unit's tests, GPU and duration. --plan prints the units and the predicted wall
time, and runs nothing. Ending the runner, by any signal including SIGKILL, kills
every pytest process it started.
"""

import argparse
import contextlib
import ctypes
import glob
import heapq
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
# $OUT as in _env.sh: run_suite.sh writes suite_<label>/ there.
OUT = Path(os.environ.get("OUT") or HERE.parents[6] / "bench_out/gluon_moe_regression")
FILES = "op_tests/triton_tests/moe/test_moe_gemm_gluon*.py"
PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
KINDS = ("passed", "failed", "skipped", "error")
JOBS_PER_GPU = int(os.environ.get("JOBS_PER_GPU", "8"))
UNIT_SECONDS = float(os.environ.get("UNIT_SECONDS", "60"))
UNIT_TIMEOUT = float(os.environ.get("UNIT_TIMEOUT", "1800"))
COLLECT_TIMEOUT = float(os.environ.get("COLLECT_TIMEOUT", "300"))  # ~10 s normally
STARTUP = 8  # seconds before a pytest process runs its first test
# pytest -q summary: "4 failed, 40 passed, 39 warnings in 321.40s (0:05:21)".
SUMMARY = re.compile(r"^=*\s*(no tests ran|\d+ [a-z]+(, \d+ [a-z]+)*) in [\d.]+s\b")
# Out of GPU memory: other tenants and this suite's own workers share each GPU.
OOM = re.compile(r"OutOfMemoryError|out of memory|hipErrorOutOfMemory|failed with OOM")
LOCK, STOP, LIVE = threading.Lock(), threading.Event(), set()  # LIVE: pytest Popens
# prctl(PR_SET_PDEATHSIG, SIGKILL), bound here because the child calls it after fork.
PRCTL, PDEATHSIG = ctypes.CDLL(None, use_errno=True).prctl, (1, int(signal.SIGKILL))


@dataclass
class Unit:
    name: str
    file: str
    args: list  # pytest arguments: the file itself, or its node ids
    ids: list | None  # tests the unit must account for; None if collection failed
    est: float
    gpu: str = ""
    start: float = 0.0
    seconds: float = 0.0
    rc: int | None = None
    counts: dict = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    reason: str = ""  # why the unit counts as broken; empty if it ran cleanly
    retried: bool = False  # rerun alone on its GPU after running out of memory


def suite_dir(arg):
    """A run directory: a path if it has a "/" or exists, else a run_suite.sh label."""
    return Path(arg) if "/" in arg or Path(arg).is_dir() else OUT / f"suite_{arg}"


def kill(proc):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


@contextlib.contextmanager
def spawn(cmd, gpu, **popen):
    """A pytest process on one GPU that cannot outlive the runner.

    Its own process group lets kill() take whatever it starts. main() kills every
    live one on its way out; for a runner that gets no way out (SIGKILL), the kernel
    kills it on PR_SET_PDEATHSIG. That fires when the spawning thread ends, and each
    thread here waits for its process first.

    preexec_fn is unsafe beside threads in general: the child could deadlock on a
    lock another thread held at fork. This one takes none (a pre-bound libc call and
    getppid), and UNIT_TIMEOUT or COLLECT_TIMEOUT would still end a stuck child.
    """
    runner = os.getpid()

    def die_with_runner():
        PRCTL(*PDEATHSIG)
        if os.getppid() != runner:  # the runner died before prctl took effect
            os._exit(1)

    proc = subprocess.Popen(
        cmd,
        env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu)},
        start_new_session=True,
        preexec_fn=die_with_runner,  # noqa: PLW1509 -- see the docstring
        **popen,
    )
    with LOCK:
        LIVE.add(proc)
        if STOP.is_set():
            kill(proc)
    try:
        yield proc
    finally:
        with LOCK:
            LIVE.discard(proc)


def collect(path, gpu):
    """Node ids of one file, or None, and what to print about failed attempts.

    A failure is retried once: one of 14 collections at once may run out of GPU
    memory or meet a JIT lock, and pass alone.
    """
    cmd, note = [*PYTEST, "--collect-only", path], ""
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    for attempt in ("failed", "failed again"):
        with spawn(cmd, gpu, **pipes) as proc:
            try:
                out, err = proc.communicate(timeout=COLLECT_TIMEOUT)
            except subprocess.TimeoutExpired:
                kill(proc)
                out, err = proc.communicate()
                err += f"\ntimed out after {COLLECT_TIMEOUT:.0f}s"
        ids = [line for line in out.splitlines() if line.startswith(path + "::")]
        if proc.returncode == 0 and ids:
            return ids, note
        tail = [*out.strip().splitlines()[-6:], *err.strip().splitlines()[-6:]]
        note += f"collection {attempt} on HIP {gpu}, exit {proc.returncode}: {path}\n"
        note += "".join(f"    {line[:200]}\n" for line in tail)
        if STOP.is_set():
            break
    return None, note


def function(nid):
    """Weight-table key of a node id: file name and test function, no parameters."""
    path, test = nid.split("::", 1)
    return Path(path).name + "::" + re.sub(r"\[.*", "", test)


def plan(paths, ids):
    """Units, longest estimate first. Heavy files become contiguous chunks."""
    table = json.loads((HERE / "suite_weights.json").read_text())["tests"]
    mean = sum(map(sum, table.values())) / sum(map(len, table.values()))
    units = []
    for path in paths:
        stem, found = Path(path).stem, ids[path]
        rows = [v for k, v in table.items() if k.split("::")[0] == Path(path).name]
        default = sum(map(sum, rows)) / sum(map(len, rows)) if rows else mean
        if found is None:  # run the whole file so its log shows the error
            units.append(Unit(stem, path, [path], None, default))
            continue
        by_function, cost = defaultdict(list), {}
        for nid in found:
            by_function[function(nid)].append(nid)
        for key, nids in by_function.items():
            known = table.get(key, [])
            if len(known) != len(nids):  # parameters changed: the old mean per test
                known = [sum(known) / len(known) if known else default] * len(nids)
            cost.update(zip(nids, known))
        total = sum(cost.values())
        k = min(len(found), math.ceil(total / UNIT_SECONDS))
        if k <= 1:
            units.append(Unit(stem, path, [path], found, total))
            continue
        # Cut where the running cost crosses a multiple of total / k. A test heavier
        # than one share lands alone; empty chunks are dropped.
        chunks, acc = [[] for _ in range(k)], 0.0
        for nid in found:
            chunks[min(k - 1, int((acc + cost[nid] / 2) * k / total))].append(nid)
            acc += cost[nid]
        for i, chunk in enumerate(c for c in chunks if c):
            est = sum(cost[nid] for nid in chunk)
            units.append(Unit(f"{stem}.{i:02d}", path, chunk, chunk, est))
    return sorted(units, key=lambda u: -u.est)


def weigh(run):
    """suite_weights.json text from a run made with --durations=0 --durations-min=0.

    Seconds per test, listed per test function in collection order, which the run's
    units.json gives: chunks are contiguous and numbered in order.
    """
    if not (run / "units.json").is_file():
        sys.exit(f"--weigh: {run} has no units.json; not a finished run_suite.py run")
    seconds = defaultdict(float)
    for log in run.glob("*.log"):
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"([\d.]+)s (?:setup|call|teardown) +(\S.*)", line)
            if m:
                seconds[m.group(2)] += float(m.group(1))
    if not seconds:
        sys.exit(
            f"--weigh: {run} has no durations; run it with "
            "PYTEST_ADDOPTS='--durations=0 --durations-min=0'"
        )
    units = json.loads((run / "units.json").read_text())["units"]
    table = defaultdict(list)
    for u in sorted(units, key=lambda u: (u["file"], len(u["name"]), u["name"])):
        for nid in u["ids"] or []:
            table[function(nid)].append(round(seconds[nid], 1))
    note = (
        "seconds of each test, per test function in collection order: run_suite.py "
        f"--weigh {run.name}, a run with PYTEST_ADDOPTS='--durations=0 "
        "--durations-min=0'"
    )
    rows = ",\n".join(f"    {json.dumps(k)}: {v}" for k, v in sorted(table.items()))
    return f'{{\n  "_comment": {json.dumps(note)},\n  "tests": {{\n{rows}\n  }}\n}}'


def makespan(units, workers):
    """Predicted wall time of longest-first list scheduling."""
    ends = [0.0] * workers
    for u in units:
        heapq.heapreplace(ends, ends[0] + STARTUP + u.est)
    return max(ends)


def summary(log):
    """Outcome counts from the last pytest summary line, or None without one.

    Output after the summary (exit-time noise on stderr) would hide it from
    compare_suite.py, which reads only the last line, so it is repeated at the end.
    """
    lines = log.read_text(errors="replace").strip().splitlines()
    found = [line for line in lines if SUMMARY.match(line)]
    if not found:
        return None
    if lines[-1] != found[-1]:
        with open(log, "a") as fh:
            fh.write(f"# run_suite.py: pytest summary repeated\n{found[-1]}\n")
    counts = dict.fromkeys(KINDS, 0)
    for n, kind in re.findall(r"(\d+) (passed|failed|skipped|error)", found[-1]):
        counts[kind] += int(n)
    return counts


def run(unit, gpu, out):
    """Run one unit on one GPU; mark it broken unless pytest accounted for it."""
    log = out / f"{unit.name}.log"
    unit.gpu, unit.start = gpu, time.time()
    cmd = [*PYTEST, "-rfE", *unit.args]
    with (
        open(log, "w") as fh,
        spawn(cmd, gpu, stdout=fh, stderr=subprocess.STDOUT) as proc,
    ):
        try:
            unit.rc = proc.wait(timeout=UNIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill(proc)
            unit.rc = proc.wait()
            unit.reason = f"timed out after {UNIT_TIMEOUT:.0f}s"
    unit.seconds = time.time() - unit.start
    counts = summary(log)
    if unit.reason:
        pass
    elif unit.rc not in (0, 1):
        unit.reason = f"pytest exited with {unit.rc}"
    elif counts is None:
        unit.reason = f"left no pytest summary (exit {unit.rc})"
    # A file that failed collection in planning ran whole: pytest's count stands.
    elif unit.ids is not None and sum(counts.values()) != len(unit.ids):
        unit.reason = f"summary counts {sum(counts.values())} of {len(unit.ids)} tests"
    if not unit.reason:
        unit.counts = counts
        return
    bad = unit.ids or [unit.file]
    unit.counts = dict.fromkeys(KINDS, 0) | {"error": len(bad)}
    with open(log, "a") as fh:
        fh.write(f"\n# run_suite.py: unit {unit.name} {unit.reason}\n")
        fh.writelines(f"ERROR {nid} - unit {unit.reason}\n" for nid in bad)
        fh.write(f"{len(bad)} error in {unit.seconds:.2f}s (unit {unit.reason})\n")


def drain(units, gpus, per_gpu, out):
    """Run units in order on per_gpu workers per GPU, each taking the next free."""
    queue, done = list(units), []

    def worker(gpu):
        while True:
            with LOCK:
                if STOP.is_set() or not queue:
                    return
                unit = queue.pop(0)
            run(unit, gpu, out)
            with LOCK:
                done.append(unit)
                status = unit.reason.upper() or fmt(unit.counts)
                print(
                    f"[{len(done):3d}/{len(units)}] {unit.seconds:5.0f}s HIP {gpu} "
                    f"{unit.name}: {status}",
                    flush=True,
                )

    threads = [
        threading.Thread(target=worker, args=(gpus[i % len(gpus)],), daemon=True)
        for i in range(len(gpus) * per_gpu)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def fmt(counts):
    return ", ".join(f"{counts[k]} {k}" for k in KINDS if counts[k]) or "nothing"


def suite(args, paths):
    """Collect, plan and run the suite, then report; True if every test counted."""
    workers = len(args.gpus) * JOBS_PER_GPU
    t0 = time.time()
    # No `with`: on a signal its exit would wait for collections main() has yet to
    # kill. Collections spread over the GPUs, as they may touch device memory.
    pool = ThreadPoolExecutor(len(paths))
    gpus = [args.gpus[i % len(args.gpus)] for i in range(len(paths))]
    ids, notes = {}, {}
    for path, (found, note) in zip(paths, pool.map(collect, paths, gpus)):
        ids[path], notes[path] = found, note
    pool.shutdown()
    units = plan(paths, ids)
    collected = sum(len(v) for v in ids.values() if v is not None)
    print(
        f"{collected} tests in {len(paths)} files -> {len(units)} units, "
        f"{workers} workers (JOBS_PER_GPU={JOBS_PER_GPU} x HIP {' '.join(args.gpus)}), "
        f"UNIT_SECONDS={UNIT_SECONDS:.0f}, predicted {makespan(units, workers):.0f}s",
        flush=True,
    )
    for path in paths:
        print(notes[path], end="")
        if ids[path] is None:
            print(f"COLLECTION FAILED {path}: running it whole", flush=True)
    if args.plan:
        for u in units:
            print(f"{u.est:7.0f}s {len(u.ids or []):4d} tests  {u.name}")
        return True

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in [*args.out.glob("*.log*"), *args.out.glob("units.json")]:
        stale.unlink()  # a rerun of a label must not mix in old chunk logs
    drain(units, args.gpus, JOBS_PER_GPU, args.out)
    # A unit that ran out of GPU memory next to its neighbours gets one rerun with a
    # GPU to itself; the first log is kept as <unit>.log.oom.
    oom = []
    for u in units:
        log = args.out / f"{u.name}.log"
        failed = u.counts["failed"] + u.counts["error"]
        if failed and OOM.search(log.read_text(errors="replace")):
            log.rename(log.with_suffix(".log.oom"))
            oom.append(Unit(u.name, u.file, u.args, u.ids, u.est, retried=True))
    if oom:
        print(f"rerunning {len(oom)} units that ran out of GPU memory, alone")
        drain(oom, args.gpus, 1, args.out)
        units = [u for u in units if u.name not in {o.name for o in oom}] + oom
    wall = time.time() - t0

    print()
    total = dict.fromkeys(KINDS, 0)
    for path in paths:
        mine = [u for u in units if u.file == path]
        counts = {k: sum(u.counts[k] for u in mine) for k in KINDS}
        total = {k: total[k] + counts[k] for k in KINDS}
        line = f"{fmt(counts)} in {sum(u.seconds for u in mine):.2f}s"
        if len(mine) > 1:
            longest = max(u.seconds for u in mine)
            line += f" over {len(mine)} units, longest {longest:.0f}s"
        print(f"{Path(path).stem + '.log':50} {line}")
    longest = max(units, key=lambda u: u.seconds)
    broken = [u for u in units if u.reason]
    late = [u for u in units if u.ids is None and not u.reason]
    collected += sum(sum(u.counts.values()) for u in late)
    ok = not broken and sum(total.values()) == collected
    print(f"total: {fmt(total)}")
    print(
        f"accounted {sum(total.values())} of {collected} collected tests, "
        f"{len(broken)} broken units: {'OK' if ok else 'MISMATCH'}"
    )
    print(f"longest unit: {longest.name} {longest.seconds:.0f}s on HIP {longest.gpu}")
    print(f"wall: {wall:.0f}s, {workers} workers on HIP {' '.join(args.gpus)}")
    for u in late:
        print(
            f"collected only when run whole: {u.name}, {sum(u.counts.values())} tests"
        )
    for u in (u for u in units if u.retried):
        print(f"rerun after running out of GPU memory: {u.name}, see {u.name}.log.oom")
    for u in broken:
        print(f"BROKEN {u.name} on HIP {u.gpu}: {u.reason}")
    (args.out / "units.json").write_text(
        json.dumps(
            {
                "wall": wall,
                "gpus": args.gpus,
                "jobs_per_gpu": JOBS_PER_GPU,
                "unit_seconds": UNIT_SECONDS,
                "collected": collected,
                "units": [asdict(u) | {"start": max(0, u.start - t0)} for u in units],
            },
            indent=1,
        )
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--gpus", default="", help="HIP ordinals, e.g. '4 5 6 7'")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument(
        "--weigh", metavar="RUN", help="run directory or run_suite.sh label"
    )
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if args.weigh:
        run_dir, table = suite_dir(args.weigh), HERE / "suite_weights.json"
        text = weigh(run_dir)
        tmp = table.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        os.replace(tmp, table)  # whole or not at all
        print(f"wrote {table} from {run_dir}")
        return
    args.gpus = args.gpus.replace(",", " ").split()
    if not args.out or not args.gpus:
        ap.error("--out and --gpus are required")
    paths = [os.path.relpath(p) for p in args.files or sorted(glob.glob(FILES))]
    if not paths:
        sys.exit(f"no test files match {FILES} under {os.getcwd()}")
    # Turn these into a normal exit, so the finally below kills every unit. SIGINT
    # already raises KeyboardInterrupt. One the caller ignores stays ignored: nohup
    # ignores SIGHUP so that the run survives a hangup.
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, lambda signum, _: sys.exit(128 + signum))
    try:
        ok = suite(args, paths)
    finally:
        with LOCK:
            STOP.set()
            for proc in LIVE:
                kill(proc)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
