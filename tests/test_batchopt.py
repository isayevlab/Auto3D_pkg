# tests/test_batchopt.py
"""Unit tests for the batchopt module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from Auto3D.engines.batch_opt.batchopt import Optimizer
from Auto3D.engines.batch_opt.model_wrapper import BatchedEnergyForce
from tests.helpers_adapter import FakeAdapter


class TestEnForceANI:
    """Tests for BatchedEnergyForce class."""

    def test_enforce_ani_delegates_to_adapter(self):
        """BatchedEnergyForce.forward should delegate to adapter's forward method."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (torch.tensor([1.0, 2.0]), torch.randn(2, 5, 3))

        # BatchedEnergyForce should accept the adapter directly
        model = BatchedEnergyForce(mock_adapter, batchsize_atoms=1024)

        coords = torch.randn(2, 5, 3)
        species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]])
        charges = torch.tensor([0, 0])

        energy, forces = model.forward(coords, species, charges)

        # Verify adapter's forward was called
        mock_adapter.forward.assert_called_once()
        call_args = mock_adapter.forward.call_args
        assert torch.equal(call_args[0][0], coords)
        assert torch.equal(call_args[0][1], species)
        assert torch.equal(call_args[0][2], charges)

    def test_enforce_ani_forward_batched(self):
        """BatchedEnergyForce.forward_batched should batch calls correctly."""
        mock_adapter = MagicMock()

        # Return consistent results for batching
        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        model = BatchedEnergyForce(mock_adapter, batchsize_atoms=10)  # Small batch size

        # Create input that will require multiple batches (5 atoms * 4 batches = 20 atoms > 10)
        coords = torch.randn(4, 5, 3)
        species = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4, dtype=torch.long)

        energy, forces = model.forward_batched(coords, species, charges)

        # batch_size = max(1, batchsize_atoms // N) = max(1, 10 // 5) = 2
        # molecules per sub-batch; 4 molecules split into chunks of 2 -> the
        # adapter must be called exactly twice, not "at least once" (which a
        # single unbatched call would also satisfy, defeating the point of a
        # forward_batched-specific test). Mirrors test_model_wrapper.py's
        # stronger call_count==2 sibling for the same batchsize_atoms/N ratio.
        assert mock_adapter.forward.call_count == 2
        assert energy.shape == (4,)
        assert forces.shape == (4, 5, 3)


@pytest.fixture
def tiny_batch():
    """A minimal ``(net, coord, numbers, charges, param, device)`` tuple.

    Mirrors ``TestConvergenceStatus.test_ensemble_opt_returns_convergence_info``'s
    setup -- a mock adapter returning zero forces (instant convergence) over
    two 3-atom structures -- so ``ensemble_opt(*tiny_batch)`` runs the same
    one-step path without duplicating the setup in every caller.
    """
    mock_adapter = MagicMock()
    mock_adapter.forward.return_value = (
        torch.tensor([0.0, 0.0]),
        torch.zeros(2, 3, 3),  # Zero forces = instant convergence
    )
    model = BatchedEnergyForce(mock_adapter, batchsize_atoms=1024)
    coord = torch.randn(2, 3, 3)
    numbers = torch.tensor([[6, 1, 1], [6, 1, 1]], dtype=torch.long)
    charges = torch.tensor([0, 0], dtype=torch.long)
    param = {"opt_steps": 10, "convergence_threshold": 0.01, "patience": 5}
    device = torch.device("cpu")
    return model, coord, numbers, charges, param, device


def test_ensemble_opt_returns_exactly_the_keys_its_callers_read(tiny_batch):
    """``ensemble_opt`` must return only the keys its callers read.

    ``ids``, ``he``, ``close`` and ``timing`` were constructed and returned
    but never read by any caller -- ``Optimizer._optimize_bucket`` reads
    ``energy``, ``fmax``, ``converged_mask``, ``oscillating_count``, ``coord``
    and ``non_finite`` only, and the direct test callers in this file read no
    more than that either. Dead weight in both the state dict and the return.
    """
    from Auto3D.engines.batch_opt.batchopt import ensemble_opt

    result = ensemble_opt(*tiny_batch)
    assert set(result) == {
        "coord",
        "numbers",
        "converged_mask",
        "oscillating_count",
        "fmax",
        "energy",
        "non_finite",
    }


def test_ensemble_opt_accepts_the_legacy_opttol_key_for_one_release(tiny_batch):
    """A hand-written param dict may still spell the key ``opttol`` until 4.0.

    Dict-style access on ``Auto3DOptions`` gets a one-release deprecation; a
    raw ``param`` dict built by hand (not through ``OptimizationConfig``)
    deserves the same courtesy rather than a hard ``KeyError`` the moment
    ``to_dict()`` renamed its key.
    """
    from Auto3D.engines.batch_opt.batchopt import ensemble_opt

    net, coord, numbers, charges, param, device = tiny_batch
    legacy_param = {k: v for k, v in param.items() if k != "convergence_threshold"}
    legacy_param["opttol"] = param["convergence_threshold"]

    with pytest.warns(DeprecationWarning, match="opttol"):
        ensemble_opt(net, coord, numbers, charges, legacy_param, device)


def test_ensemble_opt_with_convergence_threshold_key_warns_nothing(tiny_batch):
    """The current key name is read silently -- no deprecation warning."""
    import warnings

    from Auto3D.engines.batch_opt.batchopt import ensemble_opt

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ensemble_opt(*tiny_batch)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


class TestConvergenceStatus:
    """Tests for convergence status determination (Issue #90)."""

    def test_ensemble_opt_returns_convergence_info(self, tiny_batch):
        """ensemble_opt should return converged_mask and oscillating_count."""
        from Auto3D.engines.batch_opt.batchopt import ensemble_opt

        result = ensemble_opt(*tiny_batch)

        # Verify new fields are present, with their actual VALUES: zero force
        # on step 1 means fmax (0.0) is at once below convergence_threshold
        # (0.01) for both structures, so both must be reported converged and
        # neither must have been counted as oscillating -- checking only key
        # presence/type/length (as before) would pass even if the values
        # were transposed, all-False, or a stray increment leaked into
        # oscillating_count.
        assert "converged_mask" in result, "converged_mask missing from ensemble_opt return"
        assert "oscillating_count" in result, "oscillating_count missing from ensemble_opt return"
        assert isinstance(result["converged_mask"], list)
        assert isinstance(result["oscillating_count"], list)
        assert result["converged_mask"] == [True, True]
        assert result["oscillating_count"] == [0, 0]


@pytest.mark.slow
class TestConvergenceFlagDerivation:
    """The Converged/Dropped_Oscillating flags must come from production code.

    The previous tests in this class re-implemented the derivation inside the
    test body and asserted it against itself, so no production code executed.
    """

    def test_oscillating_structure_is_not_reported_converged(self, job_dir):
        """A structure at/over the patience limit must not be Converged=True."""
        pytest.importorskip("torchani")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.engines.batch_opt.batchopt import Optimizer

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "osc_test")

        sdf = job_dir / "in.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        # patience=1 guarantees the oscillation path is taken on any structure
        # that does not reduce fmax on its very first step.
        from Auto3D.engines.model_factory import create_model

        opt = Optimizer(
            in_f=str(sdf),
            out_f=str(job_dir / "out.sdf"),
            adapter=create_model("ANI2xt", torch.device("cpu")),
            device=torch.device("cpu"),
            config={
                "opt_steps": 5,
                "convergence_threshold": 1e-9,
                "patience": 1,
                "batchsize_atoms": 1024,
            },
        )
        opt.run()

        out = [m for m in Chem.SDMolSupplier(str(job_dir / "out.sdf"), removeHs=False) if m]
        assert out, "optimizer produced no output"

        # Invariant: this must hold for every molecule regardless of which path
        # its trajectory took, so it can never no-op. It fails if a regression
        # decouples Converged/Dropped_Oscillating (e.g. dropping the
        # `osc_count_i < patience` guard from `convergence_i`).
        for m in out:
            if m.HasProp("Dropped_Oscillating") and m.GetProp("Dropped_Oscillating") == "True":
                assert m.GetProp("Converged") != "True", (
                    "a structure dropped for oscillation was reported as converged"
                )

        # Non-vacuity check: the invariant above is only meaningful if the
        # oscillation path was actually taken. If this fails, the fixture
        # (patience=1, convergence_threshold=1e-9, ethanol's FIRE trajectory) has stopped
        # reproducing the oscillating condition on this platform/torch
        # version -- it does not mean the Converged/Dropped_Oscillating
        # invariant itself is broken.
        assert any(
            m.HasProp("Dropped_Oscillating") and m.GetProp("Dropped_Oscillating") == "True"
            for m in out
        ), (
            "fixture did not reproduce an oscillating structure (patience=1 "
            "should force this on the very first non-converging step); the "
            "invariant above was never exercised"
        )


def _optimizing_for_buckets(tmp_path, sizes):
    """An ``Optimizer`` instance built only to call ``_make_buckets``.

    ``_make_buckets`` is pure-Python, so a conforming double is enough and no
    model is loaded. ``Optimizer`` no longer builds its own adapter, so there
    is no ``create_model`` seam left to patch.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.engines.batch_opt.batchopt import Optimizer

    inp = tmp_path / "in.sdf"
    mols = []
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(sizes):
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i))
            w.write(m)
            mols.append(m)
    eng = Optimizer(
        str(inp),
        str(tmp_path / "o.sdf"),
        adapter=FakeAdapter(),
        device=torch.device("cpu"),
        config={
            "opt_steps": 1,
            "convergence_threshold": 0.01,
            "patience": 1,
            "batchsize_atoms": 1024,
        },
    )
    return eng, mols


def test_make_buckets_groups_by_size(tmp_path):
    """R60: a 4x atom-count spread shares one bucket; past it starts a new one.

    methane (5 atoms), ethane (8), propane (11) span 11/5 = 2.2x -- inside
    ``BUCKET_SIZE_FACTOR`` (4.0) -- so all three must land in ONE bucket, each
    index appearing exactly once. The 20-carbon ring (~60 atoms, ~12x methane,
    well past 4x) must start its own bucket rather than share theirs.
    """
    sizes = ["C", "CC", "CCC", "C1CCCCCCCCCCCCCCCCCCC1"]
    eng, mols = _optimizing_for_buckets(tmp_path, sizes)
    buckets = eng._make_buckets(mols)

    assert len(buckets) == 2
    ring_bucket = [b for b in buckets if 3 in b][0]
    small_bucket = [b for b in buckets if 3 not in b][0]
    assert ring_bucket == [3]
    assert sorted(small_bucket) == [0, 1, 2]


def test_make_buckets_splits_at_max_count(tmp_path):
    """``BUCKET_MAX_COUNT + 1`` same-size molecules form two buckets, not one."""
    from Auto3D.engines.batch_opt.batchopt import Optimizer

    sizes = ["C"] * (Optimizer.BUCKET_MAX_COUNT + 1)
    eng, mols = _optimizing_for_buckets(tmp_path, sizes)
    buckets = eng._make_buckets(mols)

    assert len(buckets) == 2
    assert sorted(i for b in buckets for i in b) == list(range(len(mols)))


def test_optimizing_preserves_input_order(tmp_path, monkeypatch):
    """Bucketing reorders internally but output order must match input."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import Auto3D.engines.batch_opt.batchopt as bo

    inp = tmp_path / "in.sdf"
    smis = ["CCCCCCCC", "C", "CCC"]  # 8,1,3 heavy atoms - deliberately unsorted
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(smis):
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i))
            w.write(m)

    from tests.helpers_batchopt import make_fake_ensemble_opt

    monkeypatch.setattr(bo, "ensemble_opt", make_fake_ensemble_opt())

    out = tmp_path / "out.sdf"
    eng = bo.Optimizer(
        str(inp),
        str(out),
        adapter=FakeAdapter(),
        device=torch.device("cpu"),
        config={
            "opt_steps": 1,
            "convergence_threshold": 0.01,
            "patience": 1,
            "batchsize_atoms": 1024,
        },
    )
    eng.run()
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert names == ["0", "1", "2"]  # original input order


@pytest.fixture
def two_bucket_run(tmp_path, monkeypatch):
    """A full ``Optimizer.run()`` over two molecules sized far enough apart
    that ``_make_buckets`` splits them into two buckets (methane vs. a
    20-carbon ring, well past ``BUCKET_SIZE_FACTOR``), using the fake
    ``ensemble_opt`` double so no model is loaded. Returns a no-argument
    callable that performs the run, so a caller can monkeypatch
    ``torch.cuda`` first.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import Auto3D.engines.batch_opt.batchopt as bo
    from tests.helpers_batchopt import make_fake_ensemble_opt

    monkeypatch.setattr(bo, "ensemble_opt", make_fake_ensemble_opt())

    sizes = ["C", "C1CCCCCCCCCCCCCCCCCCC1"]  # methane vs. a 20-carbon ring
    inp = tmp_path / "two_bucket_in.sdf"
    mols = []
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(sizes):
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i))
            w.write(m)
            mols.append(m)

    eng = bo.Optimizer(
        str(inp),
        str(tmp_path / "two_bucket_out.sdf"),
        adapter=FakeAdapter(),
        device=torch.device("cpu"),
        config={
            "opt_steps": 1,
            "convergence_threshold": 0.01,
            "patience": 1,
            "batchsize_atoms": 1024,
        },
    )
    assert len(eng._make_buckets(mols)) == 2  # this run must exercise two buckets

    return eng.run


def test_the_cuda_cache_is_released_once_per_chunk_not_once_per_bucket(monkeypatch, two_bucket_run):
    """One release per chunk, however many size buckets the chunk holds.

    The optimizer released the cache after every bucket as well; measured on
    and off against the production path, that release lowered neither peak
    reserved memory nor the allocator's retry and out-of-memory counts
    (``benchmarks/results-notes/2026-10-16-empty-cache.md``), and it is gone.
    """
    calls = {"n": 0}
    # This test must also pass on a host with no CUDA at all, where the guard
    # reads ``is_available() == False``; patch it True so the call can fire
    # regardless of what hardware is actually present.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1))
    two_bucket_run()  # the file's smallest run that produces two buckets on CPU
    assert calls["n"] == 1  # the per-chunk call in run(), and nothing per bucket


class TestBatchOptDependsDownwards:
    """``batch_opt`` must depend on ``models/``, never on ``model_factory``.

    ``batchopt.py`` imported ``Auto3D.engines.model_factory.create_model`` at module
    scope and called it in ``Optimizer.__init__``: the numerical layer
    constructing its own dependency, and reaching UP into the layer that is
    supposed to sit above it. The visible symptom was in the tests -- every one
    of them had to monkeypatch ``Auto3D.engines.batch_opt.batchopt.create_model``, a seam
    that existed only because the arrow pointed the wrong way.
    """

    def test_importing_batchopt_does_not_pull_in_the_factory(self):
        """Asserted in a fresh interpreter: an already-imported ``model_factory``
        would make this vacuous inside the test session."""
        import subprocess
        import sys

        program = (
            "import sys; import Auto3D.engines.batch_opt.batchopt as b; "
            "assert 'Auto3D.engines.model_factory' not in sys.modules, "
            "sorted(m for m in sys.modules if m.startswith('Auto3D')); "
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout

    def test_batchopt_exposes_no_create_model_seam(self):
        import Auto3D.engines.batch_opt.batchopt as bo

        assert not hasattr(bo, "create_model")


class TestOptimizingTakesAnAdapterNotAName:
    """Construction is the caller's job, and the caller must be in the worker.

    ``Optimizer`` no longer knows an engine name at all: ``pad_from_mols`` asks
    the adapter, so there is nothing left for a name to decide.
    """

    @staticmethod
    def _config():
        return {
            "opt_steps": 100,
            "convergence_threshold": 0.003,
            "patience": 1000,
            "batchsize_atoms": 1024,
        }

    def test_an_engine_name_is_rejected(self):
        """The parameters after ``out_f`` are keyword-only, so a stale positional
        call fails at the call rather than silently binding a string into the
        slot that supplies the padding values."""
        with pytest.raises(TypeError):
            Optimizer("dummy.sdf", "out.sdf", "AIMNET", torch.device("cpu"), self._config())

    def test_padding_values_come_from_the_injected_adapter(self):
        from tests.helpers_adapter import FakeAdapter

        adapter = FakeAdapter(coord_pad=1.5, species_pad=-2)
        opt = Optimizer(
            "dummy.sdf",
            "out.sdf",
            adapter=adapter,
            device=torch.device("cpu"),
            config=self._config(),
        )
        assert opt.model is adapter
        assert opt.coord_pad == 1.5
        assert opt.species_pad == -2

    def test_optimizing_no_longer_carries_an_engine_name(self):
        from tests.helpers_adapter import FakeAdapter

        opt = Optimizer(
            "dummy.sdf",
            "out.sdf",
            adapter=FakeAdapter(),
            device=torch.device("cpu"),
            config=self._config(),
        )
        assert not hasattr(opt, "name")


def test_charges_reach_the_model_as_float32():
    """``pad_from_mols`` builds charges float32; ``ensemble_opt`` must not narrow it.

    ``pad_from_mols`` documents the float32 choice at its construction site, and
    the AIMNet2 adapter casts to the coordinate dtype on arrival. Between them
    ``ensemble_opt`` re-cast to ``torch.long``, so the documented contract was
    false on the ``main()`` path and any non-integral charge was truncated
    toward zero rather than carried.
    """
    from Auto3D.engines.batch_opt.batchopt import ensemble_opt

    seen_dtypes = []

    class _RecordingAdapter(FakeAdapter):
        """Only the dtype recording is this test's own.

        Was a hand-rolled duplicate of every ``FakeAdapter`` member; that is the
        shape ``tests/helpers_adapter`` exists to absorb, and it went red the
        first time the adapter contract gained a member.
        """

        def forward(self, coords, species, charges, atom_mask=None):
            seen_dtypes.append(charges.dtype)
            return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

    model = BatchedEnergyForce(_RecordingAdapter(), batchsize_atoms=1024)
    coord = torch.randn(2, 3, 3)
    numbers = torch.tensor([[6, 1, 1], [6, 1, 1]], dtype=torch.long)
    charges = torch.tensor([0.0, 0.0], dtype=torch.float32)

    ensemble_opt(
        model,
        coord,
        numbers,
        charges,
        {"opt_steps": 10, "convergence_threshold": 0.01, "patience": 5},
        torch.device("cpu"),
    )

    assert seen_dtypes, "the model was never called"
    assert seen_dtypes[0] == torch.float32, (
        f"charges reached the model as {seen_dtypes[0]}, not float32"
    )
