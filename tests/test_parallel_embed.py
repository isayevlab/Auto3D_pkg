# tests/test_parallel_embed.py
"""Tests for Auto3D.domain.embedding (parallel conformer embedding)."""

import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from rdkit import Chem

import Auto3D.domain.embedding
from Auto3D.domain.embedding import SpeciesSkipped, _embed_single, embed_conformers_parallel

ROOT = Path(__file__).resolve().parent.parent


def _suicide_embed(smi, name, n_conformers, threshold, np_threads):
    """Module-level worker (picklable) that abruptly kills its process, breaking
    the pool. Used to exercise the BrokenProcessPool path."""
    os._exit(1)


@pytest.mark.timeout(180)
def test_an_unguarded_script_gets_the_actionable_broken_pool_note(tmp_path):
    """The hint has to reach a real user, not only a monkeypatched pool.

    ``test_parallel_embed_reraises_broken_pool`` breaks the pool by killing a
    worker, which is the OOM shape. By far the commoner trigger in practice is
    this one: parallel embedding is on by default, the pool spawns, every child
    re-imports the caller's ``__main__``, and a script whose work is not behind
    ``if __name__ == "__main__":`` raises in that re-import before embedding
    anything. The user's only clue used to be a child-process traceback about
    ``freeze_support``, which names neither the guard nor the serial fallback.

    Driven through a real subprocess rather than in-process, because the defect
    is a property of module re-import under ``spawn`` and nothing short of a
    separate interpreter has a ``__main__`` to re-import.
    """
    script = tmp_path / "unguarded.py"
    script.write_text(
        "from Auto3D.domain.embedding import embed_conformers_parallel\n"
        "\n"
        '# No `if __name__ == "__main__":` -- deliberately, that is the defect.\n'
        "print(\n"
        "    list(\n"
        "        embed_conformers_parallel(\n"
        '            [("CCO", "a"), ("CCC", "b")], n_conformers=1, n_workers=2\n'
        "        )\n"
        "    )\n"
        ")\n"
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""),
    )

    assert done.returncode != 0, (
        "test premise: an unguarded script above the pool threshold must fail, "
        f"not succeed. stdout: {done.stdout!r}"
    )
    assert "BrokenProcessPool" in done.stderr, (
        f"the failure did not surface as BrokenProcessPool: {done.stderr!r}"
    )
    assert 'if __name__ == "__main__":' in done.stderr, (
        f"the note naming the guard never reached the user: {done.stderr!r}"
    )
    assert "use_parallel_embedding=False" in done.stderr, (
        f"the note naming the serial fallback never reached the user: {done.stderr!r}"
    )


class TestEmbedSingle:
    """Tests for the _embed_single worker function."""

    def test_embed_single_returns_list_of_tuples(self):
        """_embed_single should return list of (mol, conf_idx, conf_id) tuples."""
        results = _embed_single(
            smi="C",
            name="methane",
            n_conformers=5,
            threshold=0.3,
            np_threads=1,
        )

        assert isinstance(results, list)
        assert len(results) >= 1  # At least one conformer

        for mol, conf_idx, conf_id in results:
            assert isinstance(mol, Chem.Mol)
            assert isinstance(conf_idx, int)
            assert isinstance(conf_id, str)
            assert "methane" in conf_id

    def test_embed_single_with_dynamic_conformers(self):
        """_embed_single with n_conformers=None should use dynamic calculation.

        Compare the actual conformer count against ``calculate_conformer_count``'s
        own formula for this molecule, instead of a bare ">= 1" that would
        pass even if the None branch silently stopped calling that formula
        (e.g. fell back to a fixed conformer count). Hexane is flexible
        enough that the two counts are not degenerate (unlike a rigid/small
        molecule, where embedding + RMSD pruning collapses to 1 regardless of
        the requested count and could hide a formula regression).
        """
        from Auto3D.foundation.utils.molprops import calculate_conformer_count

        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))  # hexane: flexible
        expected_upper_bound = calculate_conformer_count(mol)

        results = _embed_single(
            smi="CCCCCC",
            name="hexane",
            n_conformers=None,
            threshold=0.3,
            np_threads=1,
        )

        assert 1 <= len(results) <= expected_upper_bound, (
            f"expected between 1 and {expected_upper_bound} conformers "
            "(calculate_conformer_count's own dynamic formula for hexane), "
            f"got {len(results)}"
        )

    def test_embed_single_filters_invalid_conformers(self):
        """_embed_single should filter conformers with atom clashes."""
        results = _embed_single(
            smi="CCCC",  # butane
            name="butane",
            n_conformers=10,
            threshold=0.3,
            np_threads=1,
        )

        # All returned conformers should have valid distances
        for mol, conf_idx, conf_id in results:
            positions = mol.GetConformer(conf_idx).GetPositions()
            # min_pairwise_distance should be > 0.9 for all returned conformers
            assert positions.shape[0] > 0

    def test_a_dummy_atom_species_is_refused_with_a_reason(self):
        """N-M3: an R-group placeholder must be named, not quietly dropped.

        Without the skip, `*CCO` embeds and is handed to AIMNet2, which scores
        the dummy atom with its padding embedding (index 0) -- a number for a
        species nobody submitted. Clash relief happens to reject every
        conformer of such a mol today (UFF/MMFF cannot type atom ``*``), so the
        molecule disappears either way; what this pins is that it disappears
        for the stated reason and says so.

        The reason travels as ``SpeciesSkipped`` rather than as a warning logged
        here: this function runs in a spawned pool worker with no run-log
        handler of its own, so a warning from it reached stderr unformatted and
        never the run log, on top of the parent's own line for the same species.
        Raising lets the parent -- which is wired into Auto3D's logging -- emit
        exactly one fully formatted line.
        """
        with pytest.raises(SpeciesSkipped, match="dummy atom"):
            _embed_single("*CCO", "frag", 2, 0.3, 1)

    def test_an_unparseable_smiles_is_refused_with_a_reason(self):
        """The other skip reason travels the same way, and names the SMILES.

        A bare ``return []`` made this indistinguishable in the parent from
        "every conformer was rejected by clash relief", so the run log recorded
        that a species had vanished but not why.
        """
        with pytest.raises(SpeciesSkipped, match="failed to parse"):
            _embed_single("this-is-not-a-smiles", "bad", 2, 0.3, 1)


def test_embed_with_retry_retries_once_with_random_coords(monkeypatch):
    from rdkit import Chem

    import Auto3D.domain.embedding as emb

    seen = []

    def fake_embed(mol, numConfs, params):
        seen.append(bool(params.useRandomCoords))
        return [] if not params.useRandomCoords else [0, 1]

    monkeypatch.setattr(emb.AllChem, "EmbedMultipleConfs", fake_embed)
    n = emb.embed_with_retry(
        Chem.AddHs(Chem.MolFromSmiles("CCO")), n_conformers=2, n_threads=1, prune_rms_thresh=0.3
    )
    assert seen == [False, True] and n == 2


def test_embed_with_retry_gives_up_after_the_retry(monkeypatch):
    from rdkit import Chem

    import Auto3D.domain.embedding as emb

    monkeypatch.setattr(emb.AllChem, "EmbedMultipleConfs", lambda mol, numConfs, params: [])
    assert (
        emb.embed_with_retry(
            Chem.AddHs(Chem.MolFromSmiles("CCO")),
            n_conformers=2,
            n_threads=1,
            prune_rms_thresh=0.3,
        )
        == 0
    )


def test_embed_with_retry_does_not_retry_after_a_timed_out_attempt(monkeypatch):
    """A species that ran out of time must not be given the whole cap again.

    The retry exists for strained systems whose default initial coordinates fail
    *fast* and whose random ones succeed (N-m2). An attempt that produced nothing
    because it burned its entire budget is a different condition: random initial
    coordinates would get no more time than the first attempt had, so a second
    attempt only doubles the wall clock. Unconditional, the per-species worst
    case was 2 x EMBED_TIMEOUT_S = 120 s -- more than the ~67 s the serial path
    spent on the species P-C3 was written to bound, and paid in full by the
    serial path and by the SDF isomer engine, which has no parallel path at all.
    """
    import Auto3D.domain.embedding as emb

    calls = []

    def fake_embed(mol, numConfs, params):
        calls.append(params.timeout)
        time.sleep(2.2)  # past the 2 s cap handed in below
        return []

    monkeypatch.setattr(emb.AllChem, "EmbedMultipleConfs", fake_embed)
    n = emb.embed_with_retry(
        Chem.AddHs(Chem.MolFromSmiles("CCO")),
        n_conformers=2,
        n_threads=1,
        prune_rms_thresh=0.3,
        timeout_s=2,
    )

    assert n == 0
    assert len(calls) == 1, (
        "ETKDG was called again after an attempt that used its whole budget, so "
        f"this species costs two caps instead of one (call timeouts: {calls})"
    )


def test_the_retry_gets_only_the_budget_the_first_attempt_left(monkeypatch):
    """Both attempts together are bounded by one cap, not one each.

    Skipping the retry outright would forgo N-m2's recovery for any species that
    fails slowly but not fatally. Giving the second attempt the remainder keeps
    the recovery and keeps the documented per-species bound honest (to within the
    one second the ceiling rounding can add).
    """
    import Auto3D.domain.embedding as emb

    calls = []

    def fake_embed(mol, numConfs, params):
        calls.append(params.timeout)
        if len(calls) == 1:
            time.sleep(1.5)
            return []
        return [0]

    monkeypatch.setattr(emb.AllChem, "EmbedMultipleConfs", fake_embed)
    n = emb.embed_with_retry(
        Chem.AddHs(Chem.MolFromSmiles("CCO")),
        n_conformers=2,
        n_threads=1,
        prune_rms_thresh=0.3,
        timeout_s=4,
    )

    assert n == 1, "the retry still has to run for a first attempt that failed fast enough"
    assert len(calls) == 2
    assert calls[0] == 4, f"the first attempt should get the whole cap, got {calls[0]}"
    # Not an exact number: the remainder is measured, so it depends on how long
    # the 1.5 s sleep actually took. What must hold is that the retry got less
    # than a fresh cap and still had usable time.
    assert 1 <= calls[1] < 4, (
        f"the retry was given {calls[1]} s against a {4} s cap: a second full cap "
        "doubles the per-species bound the cap exists to set"
    )


@pytest.mark.parametrize(
    "cpus,n_species,requested,threads,expected",
    [
        # A full box, more species than the cap: the cap wins.
        (128, 98, None, 1, 32),
        # Fewer cores than the cap: the cores win.
        (4, 98, None, 1, 4),
        # Fewer species than cores: the species win. Belt-and-braces rather than
        # a cost saving -- CPython >= 3.9 starts ProcessPoolExecutor workers on
        # demand, so a pool sized above the task count never spawns the surplus.
        (128, 3, None, 1, 3),
        # An explicit request is honored verbatim: cap, cores and thread count
        # are all bypassed, because a caller who names a number has a reason.
        (128, 98, 8, 4, 8),
        # A single-core box still gets one worker, never zero.
        (1, 98, None, 1, 1),
        # Each worker threads RDKit's embedding `mpi_np` ways, so the cores have
        # to be shared out between workers rather than handed to each of them:
        # 8 cores at 4 threads apiece is 2 workers, not 8 (which would have put
        # 32 runnable threads on 8 cores).
        (8, 98, None, 4, 2),
        # The division happens before the cap, so a big box with threaded
        # workers still reaches the cap rather than overshooting it.
        (128, 98, None, 4, 32),
        # Single-threaded workers are the unshared case: all cores usable.
        (8, 98, None, 1, 8),
        # More threads per worker than cores: floor at one worker, never zero.
        (4, 98, None, 8, 1),
    ],
)
def test_resolve_embedding_workers(monkeypatch, cpus, n_species, requested, threads, expected):
    """``None`` means "scale to this machine"; an explicit count is obeyed.

    A fixed default of 4 left 124 of 128 cores idle (P-C3), so the resolution
    has to happen where both the machine and the batch size are known rather
    than in a constructor default. "The machine" means cores *per worker*: each
    worker hands ``threads_per_worker`` to ``EmbedMultipleConfs``, so handing
    every core its own worker would oversubscribe the box by that factor.
    """
    from Auto3D.domain.embedding import resolve_embedding_workers

    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert resolve_embedding_workers(requested, n_species, threads_per_worker=threads) == expected


def test_resolve_embedding_workers_defaults_to_one_thread_per_worker(monkeypatch):
    """``threads_per_worker`` is keyword-only and defaults to 1.

    Pinned so the two-argument call stays valid for any caller that does not
    thread inside its workers.
    """
    from Auto3D.domain.embedding import resolve_embedding_workers

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert resolve_embedding_workers(None, 98) == 8


def test_resolve_embedding_workers_survives_an_unknown_core_count(monkeypatch):
    """``os.cpu_count()`` returns None when the platform cannot say."""
    from Auto3D.domain.embedding import resolve_embedding_workers

    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert resolve_embedding_workers(None, 98) == 1


def test_resolve_embedding_workers_survives_a_zero_thread_count(monkeypatch):
    """``threads_per_worker=0`` must not divide by zero.

    ``Auto3DOptions`` bounds ``mpi_np`` at >= 1, but the isomer engine's direct
    callers bypass that validation entirely, and ``self.np`` is whatever they
    passed.
    """
    from Auto3D.domain.embedding import resolve_embedding_workers

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert resolve_embedding_workers(None, 98, threads_per_worker=0) == 8


def test_the_embedding_pool_arms_the_parent_death_initializer(monkeypatch):
    """Every process Auto3D starts has to die with its parent (P-C2).

    The isomer worker, the optimizers, the logger process and the Manager
    servers all pass ``_exit_when_parent_dies``; this pool did not, and this
    branch makes it part of an ordinary run. Under ``spawn`` each pool child
    holds a dup of the call queue's *write* end as well as its read end, so the
    parent's death never produces EOF: an idle worker blocks on ``get()``
    forever and a busy one keeps burning a core on ETKDG. A parent-only signal
    -- the orchestrator's own ``p1.terminate()``, an OOM kill, a plain ``kill``
    -- then strands up to ``PARALLEL_EMBED_MAX_WORKERS`` RDKit processes.

    The executor kwargs are asserted rather than the behavior because the
    behavior needs a real killed parent; that is
    ``tests/test_worker_lifecycle.py::test_a_killed_parent_leaves_no_embedding_pool_workers``,
    which is slow. This one keeps the wiring pinned in the fast tier.
    """
    import Auto3D.domain.embedding as emb

    captured: dict = {}

    class _RecordingExecutor:
        """Enough of ProcessPoolExecutor to run the submit loop in-process."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def submit(self, fn, *args):
            future: Future = Future()
            future.set_result(fn(*args))
            return future

    monkeypatch.setattr(emb, "ProcessPoolExecutor", _RecordingExecutor)
    list(emb.embed_conformers_parallel([("C", "methane")], n_conformers=1, n_workers=2))

    assert captured.get("initializer") is emb._embedding_worker_init, (
        "the embedding pool started its workers with "
        f"initializer={captured.get('initializer')!r}: a worker that does not "
        "arm _exit_when_parent_dies outlives a SIGTERMed or OOM-killed parent"
    )


def test_the_pool_initializer_arms_both_parent_death_and_double_coordinates(monkeypatch):
    """One initializer, two jobs, both of which only matter inside a worker.

    ``_exit_when_parent_dies`` is a no-op in a process with no parent, so this
    substitutes it to see that it is called at all. The pickle property is
    process-global, which is exactly why it is set here rather than at import:
    the worker is where conformers are pickled, and importing
    ``Auto3D.domain.embedding`` must not reconfigure RDKit for an unrelated
    caller (the same stance the module takes on ``set_start_method``).
    """
    import Auto3D.domain.embedding as emb

    armed = []
    monkeypatch.setattr(emb, "_exit_when_parent_dies", lambda: armed.append(True))

    before = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.NoProps)
        emb._embedding_worker_init()
        assert armed == [True], "the pool initializer never armed parent-death detection"
        assert Chem.GetDefaultPickleProperties() & Chem.PropertyPickleOptions.CoordsAsDouble, (
            "the pool initializer left conformer pickling at float32, so a "
            "coordinate crossing the pool boundary loses its low bits"
        )
    finally:
        Chem.SetDefaultPickleProperties(before)


class TestEmbedConformersParallel:
    """Tests for the parallel embedding function."""

    def test_parallel_embed_returns_conformers(self):
        """Parallel embedding should return iterator of (mol, conf_idx, conf_id) tuples."""
        smiles_names = [
            ("C", "methane"),
            ("CC", "ethane"),
        ]
        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=5,
                threshold=0.3,
                np_threads=1,
                n_workers=2,
            )
        )

        assert len(results) >= 2  # At least one conformer per input
        for mol, conf_idx, conf_id in results:
            assert mol is not None
            assert mol.GetNumConformers() > 0
            assert isinstance(conf_idx, int)
            assert isinstance(conf_id, str)

    def test_parallel_embed_with_single_worker(self):
        """Parallel embedding should work with single worker."""
        smiles_names = [
            ("CCC", "propane"),
            ("CCCC", "butane"),
        ]
        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=3,
                n_workers=1,
            )
        )

        assert len(results) >= 2

    def test_parallel_embed_with_empty_input(self):
        """Parallel embedding should handle empty input."""
        results = list(embed_conformers_parallel([], n_conformers=5, n_workers=2))
        assert len(results) == 0

    def test_parallel_embed_conf_id_format(self):
        """Conformer IDs should follow name_idx format."""
        smiles_names = [("C", "mol1")]
        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=3,
                n_workers=1,
            )
        )

        for mol, conf_idx, conf_id in results:
            assert conf_id.startswith("mol1_")
            # conf_id should be name_idx format
            parts = conf_id.split("_")
            assert len(parts) >= 2

    def test_parallel_embed_handles_complex_molecules(self):
        """Parallel embedding should handle more complex molecules."""
        smiles_names = [
            ("c1ccccc1", "benzene"),  # aromatic ring
            ("CCO", "ethanol"),  # small functional group
        ]
        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=5,
                threshold=0.3,
                np_threads=1,
                n_workers=2,
            )
        )

        assert len(results) >= 2

        # Check that benzene conformers have correct atom count
        for mol, conf_idx, conf_id in results:
            if "benzene" in conf_id:
                # Benzene with hydrogens has 12 atoms (6 C + 6 H)
                assert mol.GetNumAtoms() == 12

    def test_parallel_embed_default_parameters(self):
        """Parallel embedding should work with default parameters."""
        smiles_names = [("C", "methane")]
        results = list(embed_conformers_parallel(smiles_names))

        assert len(results) >= 1

    def test_parallel_embed_handles_embedding_errors_gracefully(self):
        """Embedding errors should be caught and logged, not crash the pipeline."""
        # Test with invalid SMILES that will fail embedding
        smiles_names = [("invalid_smiles_xyz", "test_mol")]

        results = list(embed_conformers_parallel(smiles_names, n_conformers=1))
        # Should return empty list, not raise exception
        assert results == []

    def test_parallel_embed_mixed_valid_invalid_smiles(self):
        """Pipeline should continue processing valid SMILES when some fail."""
        smiles_names = [
            ("C", "methane"),  # valid
            ("invalid_smiles", "bad_mol"),  # invalid
            ("CC", "ethane"),  # valid
        ]

        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=3,
                n_workers=2,
            )
        )

        # Should get results from valid molecules only
        assert len(results) >= 2
        conf_ids = [conf_id for _, _, conf_id in results]
        assert any("methane" in cid for cid in conf_ids)
        assert any("ethane" in cid for cid in conf_ids)
        assert not any("bad_mol" in cid for cid in conf_ids)

    def test_parallel_embed_preserves_input_order(self):
        """Output molecule order must match input order, not completion order.

        The parallel path iterates futures in submission order (not
        as_completed), so it matches the deterministic serial path. A larger
        molecule placed first would, under as_completed, finish after the small
        ones and appear out of order.
        """
        smiles_names = [
            ("C1CCCCCCCCCCC1", "ring12"),  # larger -> slower to embed
            ("C", "s1"),
            ("CC", "s2"),
            ("CCC", "s3"),
        ]
        results = list(
            embed_conformers_parallel(
                smiles_names,
                n_conformers=3,
                n_workers=4,
            )
        )

        first_seen = []
        for _, _, conf_id in results:
            name = conf_id.rsplit("_", 1)[0]
            if name not in first_seen:
                first_seen.append(name)
        assert first_seen == ["ring12", "s1", "s2", "s3"]

    def test_parallel_embed_reraises_broken_pool(self, monkeypatch):
        """A killed worker (broken pool) must surface loudly, not be swallowed.

        The per-molecule `except Exception` that catches RDKit failures would
        otherwise also catch BrokenProcessPool on every remaining future and
        silently drop the whole tail of the batch as warnings. An OOM-killed
        worker is the realistic trigger.
        """
        if mp.get_start_method() != "fork":
            pytest.skip("relies on fork to propagate the monkeypatched worker into the pool")

        # Replace the worker with one that kills its process mid-task.
        monkeypatch.setattr(Auto3D.domain.embedding, "_embed_single", _suicide_embed)

        with pytest.raises(BrokenProcessPool) as exc_info:
            list(
                embed_conformers_parallel([("C", "m1"), ("CC", "m2")], n_conformers=1, n_workers=1)
            )

        # ...and it must arrive with the two ways out attached. By far the most
        # common cause in practice is not an OOM kill but an unguarded caller
        # script: the spawn context re-imports `__main__` in every worker, which
        # raises there and breaks the pool before any molecule is embedded. The
        # bare exception named neither the guard nor the serial fallback, and the
        # only clue was a child-process traceback about freeze_support. A note,
        # rather than a different exception type, so the OOM case still surfaces
        # as the same error everything upstream already handles.
        notes = " ".join(getattr(exc_info.value, "__notes__", []))
        assert 'if __name__ == "__main__":' in notes, (
            f"BrokenProcessPool carries no actionable note: {notes!r}"
        )
        assert "use_parallel_embedding=False" in notes, (
            f"the note does not mention the serial fallback: {notes!r}"
        )

    def test_parallel_embedding_returns_the_serial_path_coordinates_exactly(self):
        """Toggling ``--parallel-embedding`` must not move an atom.

        ``_embed_single`` is the serial reference here on purpose: the parallel
        path runs that very same function, so the only difference between the
        two columns is the pickle round trip back from the worker. RDKit pickles
        conformer coordinates as float32 unless ``CoordsAsDouble`` is set, which
        showed up as a 1.0e-4 A difference on one coordinate of
        ``OC(=O)C(N)Cc1c[nH]c2ccccc12`` at SDF write precision -- chemically
        irrelevant, but it made a documented performance switch change the bytes
        of the output, which the repo's other bit-identity guards
        (``test_state_is_bit_identical``, ``TestStepForStepIdentity``) show it
        cares about. Compared at full float64 width, which is where the
        truncation is unambiguous.
        """
        pairs = [
            ("OC(=O)C(N)Cc1c[nH]c2ccccc12", "s_trp"),  # the species the parity probe caught
            ("CCO", "s_ethanol"),
            ("CC(=O)O", "s_acetic"),
            ("Oc1ccccc1", "s_phenol"),
            ("CC(C)CC(N)C(=O)O", "s_leucine"),
            ("OCC(O)CO", "s_glycerol"),
            ("CN1CCC[C@H]1c1cccnc1", "s_nicotine"),
            ("CC(=O)Nc1ccc(O)cc1", "s_paracetamol"),
            ("C1CCCCC1", "s_cyclohexane"),
            ("CSCC[C@H](N)C(=O)O", "s_methionine"),
            ("OC(=O)c1ccccc1O", "s_salicylic"),
            ("CCCCCC", "s_hexane"),
        ]

        expected = {}
        for smi, name in pairs:
            for mol, conf_idx, conf_id in _embed_single(smi, name, 2, 0.3, 1):
                expected[conf_id] = mol.GetConformer(conf_idx).GetPositions().tobytes()
        assert len(expected) >= len(pairs), "test premise: every species embeds something"

        got = {}
        for mol, conf_idx, conf_id in embed_conformers_parallel(
            pairs, n_conformers=2, threshold=0.3, np_threads=1, n_workers=2
        ):
            got[conf_id] = mol.GetConformer(conf_idx).GetPositions().tobytes()

        assert sorted(got) == sorted(expected), (
            "the two paths did not even produce the same conformer set: "
            f"{sorted(set(expected) ^ set(got))}"
        )
        differing = [cid for cid in expected if got[cid] != expected[cid]]
        assert not differing, (
            f"{len(differing)} of {len(expected)} conformers came back from the "
            f"pool with different coordinates than the serial path: {differing}"
        )

    def test_a_skipped_species_is_reported_once_in_the_parent_with_its_reason(self, caplog):
        """One line per skipped species, from the process that has the run log.

        The worker's own ``logger.warning`` reached neither the run log nor
        Auto3D's formatting: under ``spawn`` the pool children have no
        ``QueueHandler`` (``_attach_run_log_handlers`` is for the optimizer
        workers), so the line fell through to ``logging.lastResort`` -- raw on
        stderr, ungoverned by ``--quiet`` -- while the run log recorded only the
        parent's "produced no conformers", which named the species but not the
        reason. Two console lines per bad species where the serial path shows
        one. Now the reason rides a ``SpeciesSkipped`` back to the parent, which
        emits exactly one formatted warning and no second line.
        """
        with caplog.at_level(logging.WARNING, logger="Auto3D"):
            results = list(
                embed_conformers_parallel(
                    [
                        ("C", "methane"),
                        ("this-is-not-a-smiles", "bad_smiles"),
                        ("*CCO", "dummy_atom"),
                    ],
                    n_conformers=2,
                    n_workers=2,
                )
            )

        assert [cid for _, _, cid in results], "test premise: methane still embeds"
        assert not any("bad_smiles" in cid or "dummy_atom" in cid for _, _, cid in results)

        messages = [r.getMessage() for r in caplog.records]
        for name, reason in (("bad_smiles", "failed to parse"), ("dummy_atom", "dummy atom")):
            named = [m for m in messages if name in m]
            assert len(named) == 1, f"expected exactly one warning naming {name!r}, got {named!r}"
            assert reason in named[0], (
                f"the warning for {name!r} does not say why it was skipped: {named[0]!r}"
            )
        assert "produced no conformers" not in caplog.text, (
            "a species skipped with a stated reason also drew the generic "
            f"empty-result warning: {caplog.text!r}"
        )
