#!/usr/bin/env python3
"""Assert that the slow tier's shard selections still partition it.

The slow tier runs as a matrix of selections (`slow-shards` in
.github/workflows/tests.yml): three legs name files and test classes explicitly
and the last takes `tests/` minus exactly those files. That is two lists which
have to agree, and their failure modes are not symmetric. A file named in a
shard but left out of the catch-all's ignore list merely runs twice -- wasteful,
visible, harmless. A file in the ignore list that no shard names runs *never*:
no job fails, no count looks wrong, and the tier quietly stops covering it. That
is the case this script exists to make impossible to merge.

It re-derives the partition from collection alone, running no tests, and reads
the shard arguments out of the workflow file itself rather than from a copy kept
here -- a hand-copied list could agree with itself while disagreeing with the
matrix that actually runs, which is the whole failure mode again one level up.

Four invariants, each reported separately so a failure says which one broke and
on which node IDs, paths or shards:

  1. the union of the shards equals the unsharded slow set (nothing lost),
  2. the shards are pairwise disjoint (nothing runs twice),
  3. every ignored path is named by some shard (the asymmetric case, caught
     directly rather than inferred -- invariant 1 only notices it once the file
     actually holds a slow test, and the point is to catch the edit, not the
     consequence),
  4. no shard collects an empty selection.

Invariant 4 is not implied by the other three, and it is what makes selecting by
test class safe rather than merely possible. Classes are selected as
`<file>::<Class>`, so a shard's whole selection can empty out -- its tests all
gaining the `smoke` marker, or the class being hollowed out -- while the union
still equals the unsharded set and the shards stay disjoint, because the tests
that left are gone from both sides of the comparison. All three of the other
invariants hold and a runner spends two minutes installing this project to run
nothing. Before this check that case was caught only incidentally, by pytest's
exit code 5 in the shard's own test step, which reports after the install and
the model warm rather than before them and says nothing about which selection
emptied.

Invariant 3 compares *file* paths: a shard argument is reduced to the part
before `::`, so naming one class of a file counts as naming the file. The
class-level loss that leaves behind -- a file whose classes are only partly
claimed -- is invariant 1's job, and it catches it exactly, because the
unclaimed class's tests are in the unsharded set and in no shard.

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


def collect(args: list[str], label: str) -> set[str]:
    """Node IDs the slow marker selects for one pytest argument list.

    `label` names what is being collected -- a shard group, or the unsharded
    tier -- so that a collection that fails outright says which selection in the
    matrix to go and look at instead of only which arguments were passed.

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
    # answer here, and the one invariant 4 interprets. Anything else is a broken
    # collection, which this script must not report as an empty shard.
    #
    # Exit 4 is the one worth naming, because selecting by class is what makes
    # it reachable: `<file>::<Class>` for a class that has been renamed or
    # removed is a usage error ("ERROR: not found"), not an empty selection, so
    # it stops here rather than reaching invariant 4. That is still a loud
    # failure naming the shard, and the shard's own test step would exit 4 too.
    if proc.returncode not in (0, 5):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print(
            f"::error::collection failed for shard '{label}' (pytest exit "
            f"{proc.returncode}), so the partition cannot be checked: pytest "
            f"{' '.join(args)} -- a selection naming a file or class that no "
            "longer exists is the usual cause"
        )
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

    unsharded = collect(UNSHARDED, "the unsharded slow tier")
    collected = {group: collect(args, group) for group, args in shards.items()}
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
    # shard runs; `--ignore=` removes one from the catch-all. Compared at file
    # granularity: a shard argument may be `<file>::<Class>`, and the ignore
    # list can only name whole files, so the part before `::` is what the two
    # lists have in common. Naming one class therefore counts as naming the
    # file, and whether the file's *other* classes are claimed by anybody is
    # invariant 1's question, not this one.
    named = {
        arg.split("::", 1)[0] for args in shards.values() for arg in args if not arg.startswith("-")
    }
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

    # 4. Every shard runs something. The other three invariants are all
    # comparisons between the shards and the unsharded tier, and a selection
    # that has emptied out drops off both sides at once: the union still equals
    # the unsharded set, the shards are still disjoint, every ignored path is
    # still named, and one runner installs this project to run no tests. That is
    # the standing risk of selecting by class -- `<file>::<Class>` for a class
    # whose tests have all become `smoke`, or that has been emptied rather than
    # removed, collects nothing while the file it lives in still exists.
    for group, node_ids in sorted(collected.items()):
        if not node_ids:
            print(
                f"::error::shard '{group}' collects no slow tests at all, so its "
                f"runner would run nothing: pytest {' '.join(shards[group])}"
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
