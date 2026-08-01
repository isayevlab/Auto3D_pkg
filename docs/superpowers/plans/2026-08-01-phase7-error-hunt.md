# Phase 7: Error Hunt — Implementation Plan

**Goal:** Find defects the 2026-07-30 audit missed, before 3.5.0 ships. Produce a new, verified findings manifest. Fix only what is Critical; everything else is triaged for Phase 8.

**Why now:** the remediation closed every finding the audit recorded. That is not the same as the package being correct — it is the package being correct *about the things one review looked at*. Two facts from the remediation say the audit under-sampled:

1. **The audit's own search method was incomplete by construction.** It found the same-file-overwrite hazard by grepping `check_gpu_requested` call sites — which can only return functions that are *already guarded*. Two more destructive writers (`ConformerRanker`, `auto3d tautomers`) were invisible to it. Both destroyed user input.
2. **One finding was recorded backwards** (C12), and implementing it as written would have rejected every working custom NNP. If one finding was inverted, the manifest is evidence, not authority.

**Method:** search by **operation**, never by an existing guard. "Who writes files?" finds unguarded writers; "who calls the guard?" cannot.

## Global Constraints

- **Authorship:** commits authored solely by the repository owner. Never add `Co-Authored-By`, `Signed-off-by`, or any AI-attributing trailer; never mention AI, Claude, or Copilot in a commit message or branch name. Never modify git config.
- **Box limits:** ~2 GB RAM; 8 CUDA devices all busy with other work. **NEVER run `pytest -m slow`. NEVER load a neural network potential. NEVER trigger a model download.** Use `-p no:randomly`. `torchani` absent; `ase` 3.27.0 present.
- **Every suite run happens twice** — plain, and with `CUDA_VISIBLE_DEVICES=""`. Every CI runner is CPU-only; this box has 8 GPUs. Two CI-red defects shipped in this effort because a suite was green locally and lying.
- **Slow-tier call sites are audited statically** (grep), since `pytest -m slow` cannot run here.
- **Forbidden git operations:** `git checkout`, `git worktree`, `git restore`, `git stash`, `git commit --amend`, `git add -A`. Stage explicit paths only.
- **Hard invariant:** the suite carries ZERO `xfail` markers and 956 passing tests. Any xfail/xpass appearing is a defect.

## Baseline

`main` after Phase 6: **956 passed, 10 skipped, 66 deselected, 0 xfailed**, identical with and without CUDA. Ruff clean.

## The seven hunts

Each is an independent sweep producing candidate findings. **A candidate is not a finding until it is verified against running code** — this codebase has already produced one inverted audit finding, and reviews here repeatedly found that a plausible-looking claim did not survive contact with the source.

### Hunt 1 — Destructive and lossy operations, by operation

Enumerate every site that **writes, moves, deletes, or truncates** a path, and every site that drops records. For each: what happens if the destination is the source, if the write fails midway, if the process dies between two writes, if two processes race?

Seed set already known: `Chem.SDWriter`, `open(..., "w")`, `os.replace`, `shutil.move`, `os.unlink`, `Path.write_*`. Do not stop at the seed set — grep for the operations, not the known callers.

### Hunt 2 — Silent fallbacks and swallowed failures

Every `except` that does not re-raise, every `getattr(x, name, default)` supplying a *behavioral* default, every `or <fallback>`, every `.get(k, default)` on a config, every bare `continue`/`pass` in a loop over user data.

The question for each: **can this hide a wrong answer, as opposed to a slow one?** Silent CPU fallback (M23), disagreeing padding defaults (C12), and silently-dropped molecules (C6-C8) were all this shape.

### Hunt 3 — Tests that name a guarantee they do not provide

**The highest-yield hunt: ten-plus instances were found during the remediation, every one by accident rather than by looking.** Sweep all of `tests/`:

- assertions on a base exception class where a sibling/ancestor raised earlier can satisfy them
- `pytest.raises(Exception)` or `raises(...)` with no `match=` where several failures are reachable
- tests whose stubs are benign enough that the property named in the title is not actually pinned
- tests that would pass with the production behavior deleted
- attributes read on library objects that may not exist (an ASE `vib.atoms` read raised `AttributeError` instead of checking anything, undetected because it was slow-marked)

Report each with the mutation that proves it: what to break, and the fact that the test stays green.

### Hunt 4 — Documentation that is not true of the code

Treat every claim in `CHANGELOG.md`, `docs/source/**`, `README`, and every docstring that promises behavior as an assertion to verify. This phase found four false claims, including a docstring vouching for Windows safety the code did not have and release notes promising an upgrade path that would have failed at load.

Prioritize: `docs/source/migration-*.rst` (users act on it), then public docstrings, then CHANGELOG.

### Hunt 5 — Same-shape siblings

For each defect the remediation fixed, ask: **where else does this shape occur?** The remediation itself found five same-file writers where the audit found three, and three diverging copies of the tmp+`os.replace` pattern.

Start from `docs/superpowers/follow-ups-after-4.0.0-remediation.md` and the closed-finding list; for each, grep for the *pattern*, not the fixed call site.

### Hunt 6 — Chemistry and numerics correctness

Domain reviewer (`computational-chemist`). Unit handling and conversion factors; charge and spin bookkeeping; stereo/tautomer identity across transformations; energy/force sign conventions; float32 vs float64 boundaries; anything that could produce a *plausible but wrong number*. This is the class where a bug is least likely to announce itself.

### Hunt 7 — Concurrency, resources, and platform

Multiprocessing (`SDF2chunks`, worker pools): shared paths, temp-file collisions, orphaned children on failure. File handles held across `os.replace` (the Windows `74474ed` class). Platform assumptions: case-insensitive filesystems, path separators, `EXDEV` across mounts.

## Process

1. **Sweep.** Each hunt runs independently and writes candidates to its own file. Volume is fine at this stage; precision comes next.
2. **Verify adversarially.** Every candidate is independently checked against running code by someone trying to *refute* it. A candidate that cannot be demonstrated is dropped, not softened. Record the demonstration (inputs → wrong output) with each survivor.
3. **Triage.** Critical = wrong scientific results, data loss, or a break reaching users. Important = real defect, bounded blast radius, or a test that does not test its subject. Minor = naming, style, docs.
4. **Fix Criticals only, in this phase.** Each with a test that fails first, mutation-verified. Everything else feeds Phase 8.
5. **Write the manifest** to `.claude/review-manifests/review-2026-08-01-error-hunt.md`, in the same format as the original audit so the two can be compared.

## Exit criteria

- [ ] All seven hunts complete, candidates recorded.
- [ ] Every surviving finding carries a concrete demonstration, not an argument.
- [ ] Every Critical fixed, each with a mutation-verified test.
- [ ] Suite still 0 xfailed, identical with and without CUDA; ruff clean.
- [ ] New manifest written and committed.
- [ ] A one-paragraph honest answer to: **did this hunt find enough to change the release decision?**
