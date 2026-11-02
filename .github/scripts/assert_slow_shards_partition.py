#!/usr/bin/env python3
"""Assert that the slow tier's shard selections still partition it.

The slow tier runs as a matrix of file groups (`slow-shards` in
.github/workflows/tests.yml): three legs name files explicitly and the last
takes `tests/` minus exactly those files. That is two lists which have to agree,
and their failure modes are not symmetric. A file named in a shard but left out
of the catch-all's ignore list merely runs twice -- wasteful, visible, harmless.
A file in the ignore list that no shard names runs *never*: no job fails, no
count looks wrong, and the tier quietly stops covering it. That is the case this
script exists to make impossible to merge.

It re-derives the partition from collection alone, running no tests, and reads
the shard arguments out of the workflow file itself rather than from a copy kept
here -- a hand-copied list could agree with itself while disagreeing with the
matrix that actually runs, which is the whole failure mode again one level up.

Three invariants, each reported separately so a failure says which one broke and
on which node IDs or paths:

  1. the union of the shards equals the unsharded slow set (nothing lost),
  2. the shards are pairwise disjoint (nothing runs twice),
  3. every ignored path is named by some shard (the asymmetric case, caught
     directly rather than inferred -- invariant 1 only notices it once the file
     actually holds a slow test, and the point is to catch the edit, not the
     consequence).

Usage: assert_slow_shards_partition.py [workflow.yml]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_WORKFLOW = Path(".github/workflows/tests.yml")
SHARD_JOB = "slow-shards"
UNSHARDED = ["tests/"]


def collect(args: list[str]) -> set[str]:
    """Node IDs the slow marker selects for one pytest argument list.

    `-o addopts=` clears pyproject's addopts, which carries `-v` and
    `-m 'not slow'`. Without that the output format depends on a setting that
    has nothing to do with sharding: `-q` nets to a tree listing rather than one
    node ID per line, and this script would silently see no tests anywhere.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *args,
        "-m",
        "slow",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # 0 is "collected something", 5 is "collected nothing" -- a legitimate
    # answer here, and one invariant 1 interprets. Anything else is a broken
    # collection, which this script must not report as an empty shard.
    if proc.returncode not in (0, 5):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print(f"::error::collection failed (exit {proc.returncode}) for: pytest {' '.join(args)}")
        raise SystemExit(1)
    return {
        line.strip()
        for line in proc.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def main(argv: list[str]) -> int:
    workflow = Path(argv[1]) if len(argv) > 1 else DEFAULT_WORKFLOW
    legs = yaml.safe_load(workflow.read_text())["jobs"][SHARD_JOB]["strategy"]["matrix"]["include"]
    shards = {leg["group"]: leg["select"].split() for leg in legs}

    unsharded = collect(UNSHARDED)
    collected = {group: collect(args) for group, args in shards.items()}
    for group, node_ids in collected.items():
        print(f"{group}: {len(node_ids)} slow tests")
    print(f"union: {len(set().union(*collected.values()))}  unsharded: {len(unsharded)}")

    failures = 0

    # 1. Nothing lost, nothing invented.
    union = set().union(*collected.values())
    for missing in sorted(unsharded - union):
        print(f"::error::no shard runs this slow test: {missing}")
        failures += 1
    for extra in sorted(union - unsharded):
        print(f"::error::a shard selects a test the slow tier does not: {extra}")
        failures += 1

    # 2. Nothing twice. Reported per node ID with the shards that claim it,
    # because the fix is to drop it from one of them and the message should say
    # which two to look at.
    groups = sorted(collected)
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            for shared in sorted(collected[left] & collected[right]):
                print(f"::error::shards '{left}' and '{right}' both run: {shared}")
                failures += 1

    # 3. Every ignored path is somebody's job. A positional argument is a path a
    # shard runs; `--ignore=` removes one from the catch-all.
    named = {arg for args in shards.values() for arg in args if not arg.startswith("-")}
    ignored = {
        arg.split("=", 1)[1]
        for args in shards.values()
        for arg in args
        if arg.startswith("--ignore=")
    }
    for orphan in sorted(ignored - named):
        print(
            f"::error::{orphan} is ignored by the catch-all shard and named by no "
            "shard, so nothing runs its slow tests"
        )
        failures += 1

    if failures:
        plural = "problem" if failures == 1 else "problems"
        print(
            "::error::the slow shard selections no longer partition the slow "
            f"tier ({failures} {plural})"
        )
        return 1
    print(f"the {len(shards)} shard selections partition the slow tier exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
