# tests/test_optimization_engine_indexing.py
"""Gate the M6 host-device-sync reduction in ``n_steps``.

``n_steps`` used to subset the batch with boolean masks: six masked reads, six
masked writes and four more inside ``FIRE.clean``, each of which forces a
GPU->CPU synchronization on CUDA because ATen has to ``nonzero()`` the mask and
copy the element count to the host to size the output. Measured with
``tests/helpers_sync_count.py``: **exactly 18 per step**, every step, up to 2000
steps per bucket.

The rewrite computes ``torch.nonzero(not_converged)`` **once** and feeds the
resulting int64 index to ``index_select`` / ``index_copy_``, which do not sync.
``nonzero`` is itself a sync -- that is the very mechanism by which boolean-mask
indexing syncs -- so this is **18 -> 2 per step**, not 18 -> 0: one ``nonzero``
for the active subset and a second inside the step for ``FIRE.clean``, whose
mask is indexed within the active subset rather than the full batch.

These tests assert two independent things:

* **Results are unchanged.** ``TestEngineMatchesBooleanMaskReference`` runs a
  test-local reimplementation of the *old* boolean-mask loop against production
  ``n_steps`` in the same process and asserts ``torch.equal`` on every state
  tensor. Same process and same hardware, so there is no cross-platform
  float-determinism hazard -- which is exactly why a checked-in golden
  trajectory file is *not* used here.
* **The win does not silently regress.** ``TestHotLoopDoesNotSync`` counts
  dispatched ops and fails the moment someone reintroduces
  ``state['coord'][mask]``. CI has no GPU and can never time this loop, but it
  can count it exactly.
"""

from __future__ import annotations

import pytest
import torch

from Auto3D.batch_opt.fire_optimizer import FIRE
from Auto3D.batch_opt.optimization_engine import n_steps
from tests.helpers_sync_count import BOOL_MASK_LABELS, NONZERO, SyncCounter

# --------------------------------------------------------------------------- #
# Hermetic potentials. No NNP, no model download, no GPU, CPU-only, exact.
# --------------------------------------------------------------------------- #


class _AnisotropicHarmonic:
    """``E = sum_i k_i * r_i^2`` with a different ``k`` per molecule.

    The per-molecule force constant is what makes this a real test: molecules
    converge at *different* steps, so the subset gather is a genuine partial
    gather rather than a whole-batch no-op. A homogeneous batch hides exactly
    the bug this file exists to catch (the same trap documented on
    ``test_convergence_outcome_never_depends_on_energy_stability``).

    ``k`` is keyed on ``numbers[:, 0]`` so it follows a molecule through every
    re-gather, which a positional lookup would not.
    """

    def __init__(self, k: dict[int, float]) -> None:
        self.k = k

    def forward_batched(self, coord, numbers, charges, atom_mask=None):
        kk = torch.tensor(
            [self.k[int(numbers[row, 0])] for row in range(coord.shape[0])],
            dtype=coord.dtype,
        ).reshape(-1, 1, 1)
        energy = (kk * coord**2).sum(dim=(1, 2)).to(torch.double)
        forces = -2.0 * kk * coord
        return energy, forces


class _ConstantForce:
    """Force never decreases, so every molecule leaves via the oscillation drop.

    This is the only way to exercise the ``patience`` path, and (with a
    ``patience`` larger than ``n``) the only way to keep every molecule active
    for a known number of steps -- which the sync count below depends on.
    """

    def forward_batched(self, coord, numbers, charges, atom_mask=None):
        forces = torch.zeros_like(coord)
        forces[..., 0] = 0.5
        return torch.zeros(coord.shape[0], dtype=torch.double), forces


def _make_state(nn, batch: int, natoms: int, seed: int, start: float = 2.0) -> dict:
    """Build a fresh ``n_steps`` state dict with reproducible coordinates."""
    torch.manual_seed(seed)
    numbers = torch.arange(batch).reshape(batch, 1).expand(batch, natoms).contiguous()
    return {
        "coord": torch.rand(batch, natoms, 3, dtype=torch.float) * start,
        "numbers": numbers.to(torch.long),
        "charges": torch.zeros(batch, dtype=torch.long),
        "nn": nn,
        "converged_mask": torch.zeros(batch, dtype=torch.bool),
        "fmax": torch.full((batch,), 999.0),
        "energy": torch.full((batch,), float("inf"), dtype=torch.double),
    }


# --------------------------------------------------------------------------- #
# T1 / T2 -- the primitive claims the rewrite rests on.
# --------------------------------------------------------------------------- #

_SHAPES = {
    "1d": (7,),
    "2d-col": (7, 1),
    "3d-coords": (7, 4, 3),
}
_MASKS = {
    "all-true": [True] * 7,
    "all-false": [False] * 7,
    "single-true": [False, False, True, False, False, False, False],
    "alternating": [True, False, True, False, True, False, True],
    "random": [False, True, True, False, False, True, False],
}


def _sample(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(11)
    if dtype is torch.bool:
        return torch.rand(shape) > 0.5
    if dtype is torch.long:
        return torch.randint(-50, 50, shape)
    return torch.randn(shape, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.long, torch.bool])
@pytest.mark.parametrize("shape_name", list(_SHAPES))
@pytest.mark.parametrize("mask_name", list(_MASKS))
def test_index_select_matches_boolean_mask_read(dtype, shape_name, mask_name):
    """``x.index_select(0, nonzero(m))`` equals ``x[m]``, bit for bit.

    Both emit rows in ascending index order (``nonzero`` returns sorted
    indices), so the gathered subset is identical, not merely equivalent. This
    is the read half of M6 and it must hold for every dtype in ``state``:
    float coordinates, float64 energies, int64 counters, bool masks.
    """
    x = _sample(_SHAPES[shape_name], dtype)
    mask = torch.tensor(_MASKS[mask_name])
    idx = torch.nonzero(mask, as_tuple=True)[0]

    assert idx.dtype is torch.int64
    assert torch.equal(x.index_select(0, idx), x[mask])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.long, torch.bool])
@pytest.mark.parametrize("shape_name", list(_SHAPES))
@pytest.mark.parametrize("mask_name", list(_MASKS))
def test_index_copy_matches_boolean_mask_write(dtype, shape_name, mask_name):
    """``dst.index_copy_(0, nonzero(m), v)`` equals ``dst[m] = v``, bit for bit.

    ``nonzero`` yields unique indices, so unlike ``index_add_`` / ``scatter_add_``
    there is no accumulation-order ambiguity to worry about. The all-false case
    is included on purpose: an empty index is a valid no-op, which is what the
    mid-loop "everything just converged" path relies on.
    """
    shape = _SHAPES[shape_name]
    mask = torch.tensor(_MASKS[mask_name])
    idx = torch.nonzero(mask, as_tuple=True)[0]
    source = _sample((int(idx.numel()), *shape[1:]), dtype)

    expected = _sample(shape, dtype).clone()
    expected[mask] = source
    actual = _sample(shape, dtype).clone()
    actual.index_copy_(0, idx, source)

    assert torch.equal(actual, expected)


def test_index_copy_and_index_put_dtype_strictness_measured():
    """Pin the exact dtype strictness of both write forms. Measured, not assumed.

    The intuition "``x[mask] = v`` silently casts, ``index_copy_`` raises" is only
    half right, and the wrong half matters:

    * ``x[mask] = scalar`` **does** cast silently, because ATen's
      ``canDispatchToMaskedFill`` fast path lowers a single-element value to
      ``masked_fill_``.
    * ``x[mask] = tensor`` with a mismatched dtype **already raises** --
      ``"Index put requires the source and destination dtypes match"`` -- in
      both directions, narrowing and widening.
    * ``index_copy_`` raises in every mismatched case, with its own message.

    So switching to ``index_copy_`` does not introduce a new class of failure for
    tensor writes; it only removes the scalar fast path's silent cast. The
    consequence for ``n_steps`` is the same either way: cast explicitly at every
    write site, which is what it now does.
    """
    mask = torch.tensor([True, True, False])
    index = torch.tensor([0, 1])

    # Scalar value: index_put_ casts silently via masked_fill_.
    scalar_dst = torch.zeros(3, dtype=torch.float32)
    scalar_dst[torch.tensor([True, False, False])] = torch.ones(1, dtype=torch.float64)
    assert scalar_dst.dtype is torch.float32
    assert scalar_dst[0] == 1.0

    # Tensor value: BOTH forms raise, with different messages.
    put_dst = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Index put requires the source and destination"):
        put_dst[mask] = torch.ones(2, dtype=torch.float64)

    copy_dst = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="expected to have the same dtype"):
        copy_dst.index_copy_(0, index, torch.ones(2, dtype=torch.float64))

    # Widening is rejected too, so ".to(dtype)" is required, not merely tidy.
    widen_dst = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="expected to have the same dtype"):
        widen_dst.index_copy_(0, index, torch.ones(2, dtype=torch.float32))
    widen_dst.index_copy_(0, index, torch.ones(2, dtype=torch.float32).to(torch.float64))
    assert widen_dst.tolist() == [1.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# T3 / T4 -- the two masked assignments that became torch.where.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fmax_values",
    [
        [0.5, 2.0, 0.1, 3.0],
        [9.0, 9.0, 9.0, 9.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, float("nan"), 0.2, float("inf")],
    ],
)
def test_smallest_fmax_where_matches_masked_assignment(fmax_values):
    """The ``torch.where`` form of the ``smallest_fmax`` update is exact.

    Including NaN, which is the whole point. ``<`` is ``False`` for NaN, so the
    original masked assignment *kept* the previous ``smallest_fmax`` when a
    molecule's force went NaN. ``torch.minimum`` would instead propagate the NaN
    and poison that molecule's oscillation tracking for the remainder of the
    run. This test is what fails if someone "simplifies" the ``where`` into a
    ``minimum``.
    """
    fmax = torch.tensor(fmax_values)
    smallest = torch.tensor([[1.0], [1.0], [1.0], [1.0]])

    fmax_col = fmax.reshape(-1, 1)
    reduced = (fmax_col < smallest).reshape(-1)

    reference = smallest.clone()
    reference[reduced] = fmax_col[reduced]
    rewrite = torch.where(reduced.unsqueeze(-1), fmax_col, smallest)

    assert torch.equal(rewrite, reference)
    # And the guard against the tempting simplification:
    if torch.isnan(fmax).any():
        assert not torch.equal(torch.minimum(fmax_col, smallest), reference)


@pytest.mark.parametrize(
    "patterns",
    [
        [[True, False, True, False]],
        [[False, False, False, False]],
        [[True, True, True, True]],
        [[False, False, True, False], [False, True, False, False], [False] * 4],
    ],
)
def test_oscillating_count_where_matches_masked_assignment(patterns):
    """The fused ``torch.where`` counter update is integer-exact.

    The original was two statements -- zero the reduced entries, then increment
    the non-reduced ones -- and the ordering mattered: a reduced molecule was
    zeroed and then incremented by ``False``, landing on 0. The single ``where``
    (reduced -> 0, else -> old + 1) reproduces that, including across multiple
    accumulating steps.
    """
    reference = torch.zeros(4, dtype=torch.long)
    rewrite = torch.zeros(4, dtype=torch.long)

    for pattern in patterns:
        reduced = torch.tensor(pattern)

        reference[reduced] = 0
        reference += ~reduced

        rewrite = torch.where(reduced, torch.zeros_like(rewrite), rewrite + 1)

        assert torch.equal(rewrite, reference)


# --------------------------------------------------------------------------- #
# T5 -- the keystone: bit-identity against the old boolean-mask loop.
# --------------------------------------------------------------------------- #


def _reference_n_steps(state, n, opttol, patience, atom_mask=None):
    """The pre-M6 boolean-mask loop, reimplemented here on purpose.

    This is a *reference*, not a copy of the code under test: it imports nothing
    from ``optimization_engine`` and does all of its subsetting with boolean
    masks, including inlining the old body of ``FIRE.clean`` (which now takes an
    int64 index). Only ``FIRE.__call__`` is shared, because the FIRE step math
    is not what M6 changed and is covered by ``test_fire_optimizer.py``.

    Keeping the reference in-process rather than checking in a golden
    trajectory is deliberate: a golden file would compare float results across
    platforms and BLAS versions, and would fail for reasons that have nothing
    to do with this rewrite.
    """
    numbers = state["numbers"]
    coord = state["coord"]
    if atom_mask is None:
        atom_mask = torch.ones_like(numbers, dtype=torch.bool)
    state["atom_mask"] = atom_mask

    optimizer = FIRE(coord)
    smallest_fmax0 = torch.full((len(coord), 1), 999.0, dtype=torch.float, device=coord.device)
    state["oscillating_count"] = torch.zeros(len(coord), dtype=torch.long, device=coord.device)

    istep = 0
    for istep in range(1, n + 1):
        not_converged = ~state["converged_mask"]
        if istep % 10 == 0 and not not_converged.any():
            break

        coord = state["coord"][not_converged]
        if coord.shape[0] == 0:
            break
        numbers = state["numbers"][not_converged]
        charges = state["charges"][not_converged]
        atom_mask_subset = state["atom_mask"][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscillating_count = state["oscillating_count"][not_converged]

        coord.requires_grad_(True)
        energy, forces = state["nn"].forward_batched(
            coord, numbers, charges, atom_mask=atom_mask_subset
        )
        coord.requires_grad_(False)

        forces = forces.masked_fill(~atom_mask_subset.unsqueeze(-1), 0.0)
        fmax = forces.norm(dim=-1).max(dim=-1)[0]
        not_converged_post1 = fmax > opttol
        stepped = optimizer(coord, forces).detach()
        coord = torch.where(not_converged_post1.view(-1, 1, 1), stepped, coord)

        fmax_reduced = (fmax.reshape(-1, 1) < smallest_fmax).reshape(-1)
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        oscillating_count[fmax_reduced] = 0
        oscillating_count += ~fmax_reduced
        not_oscillating = oscillating_count < patience
        not_converged_post = not_converged_post1 & not_oscillating

        # The old FIRE.clean body, verbatim, so the reference does not depend on
        # the new int64 signature.
        optimizer.v = optimizer.v[not_converged_post]
        optimizer.Nsteps = optimizer.Nsteps[not_converged_post]
        optimizer.dt = optimizer.dt[not_converged_post]
        optimizer.a = optimizer.a[not_converged_post]

        state["converged_mask"][not_converged] = ~not_converged_post
        state["fmax"][not_converged] = fmax
        state["energy"][not_converged] = energy.detach().to(state["energy"].dtype)
        state["coord"][not_converged] = coord
        smallest_fmax0[not_converged] = smallest_fmax
        state["oscillating_count"][not_converged] = oscillating_count

    final_coord = state["coord"].detach().clone().requires_grad_(True)
    e_final, f_final = state["nn"].forward_batched(
        final_coord, state["numbers"], state["charges"], atom_mask=state["atom_mask"]
    )
    state["energy"] = e_final.detach().to(state["energy"].dtype)
    f_final = f_final.detach().masked_fill(~state["atom_mask"].unsqueeze(-1), 0.0)
    state["fmax"] = f_final.norm(dim=-1).max(dim=-1)[0].to(state["fmax"].dtype)
    return istep


_STAGGERED_K = {i: 1.0 + 3.0 * i for i in range(8)}
_WIDE_K = {i: 0.5 + 0.37 * i for i in range(64)}

# (label, nn factory, batch, natoms, n, opttol, patience, seed, atom_mask)
_SCENARIOS = [
    (
        "staggered convergence",
        lambda: _AnisotropicHarmonic(_STAGGERED_K),
        8,
        6,
        2000,
        0.01,
        5000,
        0,
        None,
    ),
    (
        "staggered plus oscillation drops",
        lambda: _AnisotropicHarmonic(_STAGGERED_K),
        8,
        6,
        2000,
        1e-9,
        20,
        1,
        None,
    ),
    ("all oscillating", _ConstantForce, 6, 5, 200, 0.01, 7, 2, None),
    ("single molecule", lambda: _AnisotropicHarmonic(_STAGGERED_K), 1, 4, 500, 0.01, 5000, 3, None),
    (
        "n=0, loop never runs",
        lambda: _AnisotropicHarmonic(_STAGGERED_K),
        4,
        4,
        0,
        0.01,
        5000,
        4,
        None,
    ),
    ("large batch", lambda: _AnisotropicHarmonic(_WIDE_K), 64, 9, 800, 0.02, 300, 5, None),
    (
        "padded batch, 2 ghost slots",
        lambda: _AnisotropicHarmonic(_STAGGERED_K),
        8,
        6,
        2000,
        0.01,
        5000,
        6,
        "two-ghosts",
    ),
]
_SCENARIOS += [
    (
        f"randomized seed {seed}",
        lambda: _AnisotropicHarmonic(_STAGGERED_K),
        8,
        6,
        400,
        0.03,
        60,
        seed,
        None,
    )
    for seed in range(7, 17)
]

_STATE_KEYS = ("coord", "energy", "fmax", "converged_mask", "oscillating_count")


class TestEngineMatchesBooleanMaskReference:
    """``n_steps`` is bit-identical to the boolean-mask loop it replaced."""

    @pytest.mark.parametrize(
        "label,nn_factory,batch,natoms,n,opttol,patience,seed,mask_kind",
        _SCENARIOS,
        ids=[s[0] for s in _SCENARIOS],
    )
    def test_state_is_bit_identical(
        self, label, nn_factory, batch, natoms, n, opttol, patience, seed, mask_kind
    ):
        """Every state tensor matches exactly -- ``torch.equal``, not ``allclose``.

        ``index_select(0, nonzero(m))`` and ``x[m]`` gather the same rows in the
        same order, ``index_copy_`` and ``x[m] = v`` write the same rows, and no
        arithmetic or reduction order changed. So the correct assertion is exact
        equality; a tolerance would hide precisely the kind of reordering bug
        this is meant to detect.
        """
        atom_mask = None
        if mask_kind == "two-ghosts":
            atom_mask = torch.ones(batch, natoms, dtype=torch.bool)
            atom_mask[:, -2:] = False

        reference_state = _make_state(nn_factory(), batch, natoms, seed)
        production_state = _make_state(nn_factory(), batch, natoms, seed)

        _reference_n_steps(
            reference_state,
            n=n,
            opttol=opttol,
            patience=patience,
            atom_mask=None if atom_mask is None else atom_mask.clone(),
        )
        n_steps(
            production_state,
            n=n,
            opttol=opttol,
            patience=patience,
            atom_mask=None if atom_mask is None else atom_mask.clone(),
        )

        mismatched = [
            key
            for key in _STATE_KEYS
            if not torch.equal(reference_state[key], production_state[key])
        ]
        assert not mismatched, (
            f"{label}: state differs from the boolean-mask reference in "
            f"{mismatched}; max abs deltas "
            + ", ".join(
                f"{key}="
                f"{(reference_state[key].to(torch.double) - production_state[key].to(torch.double)).abs().max().item():.3e}"
                for key in mismatched
            )
        )

    def test_scenarios_actually_exercise_partial_gathers(self):
        """Guard the guard: the staggered scenario must converge unevenly.

        If every molecule converged on the same step, the subset gather would
        always be the whole batch and the bit-identity above would be vacuous --
        it would never compare a *partial* gather, which is the only thing M6
        changed. Observe the width of the batch handed to the model each step:
        a staggered run shrinks in several stages.
        """

        class _WidthRecorder:
            def __init__(self, inner):
                self.inner = inner
                self.widths: list[int] = []

            def forward_batched(self, coord, numbers, charges, atom_mask=None):
                self.widths.append(int(coord.shape[0]))
                return self.inner.forward_batched(coord, numbers, charges, atom_mask=atom_mask)

        recorder = _WidthRecorder(_AnisotropicHarmonic(_STAGGERED_K))
        state = _make_state(recorder, 8, 6, seed=0)
        n_steps(state, n=2000, opttol=0.01, patience=5000)

        # The last call is the end-of-function recompute, which is always
        # full-width; the loop body is everything before it.
        in_loop = recorder.widths[:-1]
        assert in_loop[0] == 8
        assert len(set(in_loop)) >= 3, (
            "the active set never shrank in stages, so no partial gather was "
            f"ever exercised: observed batch widths {sorted(set(in_loop))}"
        )
        assert min(in_loop) < 8


class TestStepForStepIdentity:
    """Trajectories agree at *every* prefix length, not just at the end.

    Final-state equality can in principle hide two errors that cancel. Comparing
    after 1, 2, 3, ... steps localizes any divergence to the first step where it
    appears, which is what makes the decomposition of ``n_steps`` into
    ``_step_active_subset`` / ``_scatter_back`` / ``_recompute_final_energy_and_fmax``
    checkable rather than merely plausible: the refactor and the indexing change
    landed together, so nothing may be taken on faith about either.

    ``n_steps`` builds a fresh ``FIRE`` sized to the full batch on entry, so it
    cannot be driven one step at a time; each prefix length is a fresh run from
    identical initial conditions instead.
    """

    @pytest.mark.parametrize("mask_kind", [None, "two-ghosts"])
    def test_every_prefix_of_the_trajectory_matches(self, mask_kind):
        """40 prefixes x 5 state tensors, exact equality, on a hermetic potential."""
        batch, natoms = 8, 6
        atom_mask = None
        if mask_kind == "two-ghosts":
            atom_mask = torch.ones(batch, natoms, dtype=torch.bool)
            atom_mask[:, -2:] = False

        for steps in range(1, 41):
            reference_state = _make_state(_AnisotropicHarmonic(_STAGGERED_K), batch, natoms, seed=0)
            production_state = _make_state(
                _AnisotropicHarmonic(_STAGGERED_K), batch, natoms, seed=0
            )
            _reference_n_steps(
                reference_state,
                n=steps,
                opttol=0.05,
                patience=12,
                atom_mask=None if atom_mask is None else atom_mask.clone(),
            )
            n_steps(
                production_state,
                n=steps,
                opttol=0.05,
                patience=12,
                atom_mask=None if atom_mask is None else atom_mask.clone(),
            )

            for key in _STATE_KEYS:
                assert torch.equal(reference_state[key], production_state[key]), (
                    f"diverged at step {steps} in {key!r}"
                )

    def test_the_prefixes_are_not_all_the_same_state(self):
        """Guard the guard: the trajectory must actually move across prefixes.

        If every prefix produced identical state (already converged at step 1),
        the 40 comparisons above would be 40 copies of one comparison.
        """
        seen = set()
        for steps in (1, 5, 20, 40):
            state = _make_state(_AnisotropicHarmonic(_STAGGERED_K), 8, 6, seed=0)
            n_steps(state, n=steps, opttol=0.05, patience=12)
            seen.add(tuple(state["coord"].reshape(-1).tolist()))
        assert len(seen) == 4


class TestDecomposedHelpers:
    """The four pieces ``n_steps`` was split into are callable on their own.

    The point of the split is not line count -- it is that each piece can be
    driven directly with a state it did not build itself, which the 200-line
    original could not be.
    """

    def _state(self):
        state = _make_state(_AnisotropicHarmonic(_STAGGERED_K), 4, 5, seed=0)
        state["atom_mask"] = torch.ones_like(state["numbers"], dtype=torch.bool)
        state["oscillating_count"] = torch.zeros(4, dtype=torch.long)
        return state

    def test_step_active_subset_returns_rows_aligned_with_the_index(self):
        """Every field of the result has the active subset's leading dimension."""
        from Auto3D.batch_opt.optimization_engine import _step_active_subset

        state = self._state()
        optimizer = FIRE(state["coord"].index_select(0, torch.tensor([1, 3])))
        smallest = torch.full((4, 1), 999.0)
        active = torch.tensor([1, 3])

        result = _step_active_subset(state, optimizer, active, smallest, opttol=0.01, patience=100)

        assert result.coord.shape == (2, 5, 3)
        for field in (result.energy, result.fmax, result.still_active, result.oscillating_count):
            assert field.shape[0] == 2
        assert result.smallest_fmax.shape == (2, 1)
        # And it left the full-batch state alone: writing is _scatter_back's job.
        assert torch.equal(state["converged_mask"], torch.zeros(4, dtype=torch.bool))

    def test_scatter_back_writes_only_the_active_rows(self):
        """Inactive rows keep their previous values, exactly.

        A scatter that touched an untouched row would silently overwrite a
        converged structure's final geometry with a stale one.
        """
        from Auto3D.batch_opt.optimization_engine import _scatter_back, _StepResult

        state = self._state()
        state["coord"] = torch.zeros(4, 5, 3)
        state["fmax"] = torch.full((4,), 7.0)
        state["energy"] = torch.full((4,), 3.0, dtype=torch.double)
        smallest = torch.full((4, 1), 999.0)
        active = torch.tensor([1, 3])

        _scatter_back(
            state,
            active,
            smallest,
            _StepResult(
                coord=torch.ones(2, 5, 3),
                energy=torch.full((2,), -1.0, dtype=torch.double),
                fmax=torch.full((2,), 0.5),
                still_active=torch.tensor([True, False]),
                smallest_fmax=torch.full((2, 1), 0.5),
                oscillating_count=torch.tensor([0, 4]),
            ),
        )

        assert state["converged_mask"].tolist() == [False, False, False, True]
        assert state["fmax"].tolist() == [7.0, 0.5, 7.0, 0.5]
        assert state["energy"].tolist() == [3.0, -1.0, 3.0, -1.0]
        assert torch.equal(state["coord"][0], torch.zeros(5, 3))
        assert torch.equal(state["coord"][1], torch.ones(5, 3))
        assert smallest.reshape(-1).tolist() == [999.0, 0.5, 999.0, 0.5]
        assert state["oscillating_count"].tolist() == [0, 0, 0, 4]

    def test_scatter_back_casts_to_the_destination_dtype(self):
        """float64 sources land in float32 destinations without raising.

        ``index_copy_`` refuses to cast in either direction, so the cast has to be
        at the call site. This is the unit-level statement of what
        ``test_float64_model_outputs_do_not_raise`` checks end to end.
        """
        from Auto3D.batch_opt.optimization_engine import _scatter_back, _StepResult

        state = self._state()
        smallest = torch.full((4, 1), 999.0)
        _scatter_back(
            state,
            torch.tensor([0]),
            smallest,
            _StepResult(
                coord=torch.ones(1, 5, 3, dtype=torch.float64),
                energy=torch.zeros(1, dtype=torch.float32),
                fmax=torch.full((1,), 0.25, dtype=torch.float64),
                still_active=torch.tensor([True]),
                smallest_fmax=torch.full((1, 1), 0.25, dtype=torch.float64),
                oscillating_count=torch.tensor([2], dtype=torch.int32),
            ),
        )
        assert state["coord"].dtype is torch.float32
        assert state["fmax"].dtype is torch.float32
        assert state["energy"].dtype is torch.float64
        assert state["oscillating_count"].dtype is torch.long
        assert state["fmax"][0].item() == 0.25

    def test_emit_progress_is_a_no_op_without_a_callback(self):
        """No callback means no ``optimization_counts``, hence no sync at all."""
        from Auto3D.batch_opt.optimization_engine import _emit_progress

        state = self._state()
        counter = SyncCounter()
        with counter:
            _emit_progress(state, patience=100, progress_cb=None, istep=5)
        assert counter.total == 0, counter.report()

    def test_emit_progress_swallows_a_failing_callback(self):
        """A broken progress display must never abort an optimization."""
        from Auto3D.batch_opt.optimization_engine import _emit_progress

        def explode(event):
            raise RuntimeError("display is on fire")

        _emit_progress(self._state(), patience=100, progress_cb=explode, istep=5)

    def test_emit_progress_reports_the_counts(self):
        """The event carries the five documented keys."""
        from Auto3D.batch_opt.optimization_engine import _emit_progress

        state = self._state()
        state["converged_mask"] = torch.tensor([True, True, False, False])
        events: list[dict] = []
        _emit_progress(state, patience=100, progress_cb=events.append, istep=7)
        assert events == [{"step": 7, "total": 4, "converged": 2, "dropped": 0, "active": 2}]

    def test_recompute_final_energy_and_fmax_uses_the_stored_coordinates(self):
        """Reported energy/fmax describe the reported geometry, not the pre-step one."""
        from Auto3D.batch_opt.optimization_engine import (
            _recompute_final_energy_and_fmax,
        )

        state = self._state()
        state["coord"] = torch.full((4, 5, 3), 0.5)
        state["energy"] = torch.full((4,), 12345.0, dtype=torch.double)
        state["fmax"] = torch.full((4,), 999.0)

        _recompute_final_energy_and_fmax(state)

        expected_energy = torch.tensor(
            [_STAGGERED_K[i] * 5 * 3 * 0.25 for i in range(4)], dtype=torch.double
        )
        assert torch.allclose(state["energy"], expected_energy)
        expected_fmax = torch.tensor([(2.0 * _STAGGERED_K[i] * 0.5) * (3**0.5) for i in range(4)])
        assert torch.allclose(state["fmax"], expected_fmax, rtol=1e-5)

    def test_recompute_ignores_padded_atom_forces(self):
        """Ghost slots cannot inflate the reported fmax."""
        from Auto3D.batch_opt.optimization_engine import (
            _recompute_final_energy_and_fmax,
        )

        state = self._state()
        state["coord"] = torch.full((4, 5, 3), 0.1)
        state["coord"][:, -1, :] = 50.0  # a wildly displaced ghost atom
        state["atom_mask"][:, -1] = False

        _recompute_final_energy_and_fmax(state)

        quiet = torch.tensor([(2.0 * _STAGGERED_K[i] * 0.1) * (3**0.5) for i in range(4)])
        assert torch.allclose(state["fmax"], quiet, rtol=1e-5)


# --------------------------------------------------------------------------- #
# T6 -- the regression lock.
# --------------------------------------------------------------------------- #


def _loop_body_syncs(steps: int) -> SyncCounter:
    """Count syncs for exactly ``steps`` full-width loop iterations.

    ``patience`` is huge and the force is constant, so nothing ever converges
    and every step runs the full body. ``steps < 10`` keeps the throttled
    ``not_converged.any()`` (every 10 steps) and ``print_stats`` (``n >= 10``)
    off the hot path, so the count is the loop body alone plus a fixed
    end-of-function constant -- which cancels in the delta below.
    """
    counter = SyncCounter(attribute=True)
    state = _make_state(_ConstantForce(), 8, 6, seed=0)
    with counter:
        n_steps(state, n=steps, opttol=0.0, patience=10**9)
    return counter


class TestHotLoopDoesNotSync:
    """The counted, CI-enforceable half of this cluster."""

    def test_hot_loop_does_no_boolean_mask_indexing(self):
        """Zero boolean-mask reads and zero boolean-mask writes in ``n_steps``.

        This fails the moment someone writes ``state['coord'][mask]`` again.
        It is the durable part of the change: CI cannot time the loop, but it
        can prove the sync-forcing ops are gone.
        """
        counter = _loop_body_syncs(9)
        assert counter.bool_mask_ops == 0, (
            "n_steps performed boolean-mask indexing, which forces a GPU->CPU "
            "sync per call. Use index_select / index_copy_ with the int64 index "
            f"from the single torch.nonzero instead.\n{counter.report()}"
        )

    def test_hot_loop_syncs_at_most_twice_per_step(self):
        """At most 2 sync-forcing ops per step, and they are ``nonzero`` calls.

        Two, not zero: one ``nonzero`` for the active subset (reused by six
        gathers and six scatters) and one for ``FIRE.clean``, whose mask is
        indexed within the active subset rather than the full batch. The delta
        between two step counts cancels every fixed end-of-function cost.
        """
        few, many = 4, 9
        delta = _loop_body_syncs(many).total - _loop_body_syncs(few).total
        per_step = delta / (many - few)
        assert per_step <= 2.0, (
            f"{per_step:.1f} sync-forcing ops per optimization step, expected "
            f"<= 2.0\n{_loop_body_syncs(many).report()}"
        )

    def test_the_two_remaining_syncs_are_both_nonzero(self):
        """Name the survivors, so the accounting stays honest.

        ``nonzero`` *is* a sync on CUDA -- it is the mechanism by which
        boolean-mask indexing syncs in the first place. The win is that one
        ``nonzero`` result is reused twelve times instead of each gather and
        scatter computing its own, i.e. 18 -> 2, never 18 -> 0.
        """
        counter = _loop_body_syncs(9)
        non_nonzero = {
            label: count for label, count in counter.counts.items() if label != NONZERO and count
        }
        # print_stats runs once at the end of n_steps and reads two scalars.
        readbacks = sum(non_nonzero.values())
        assert readbacks <= 2, (
            "unexpected sync-forcing ops beyond the two nonzero calls and the "
            f"final print_stats: {non_nonzero}\n{counter.report()}"
        )
        assert counter.counts[NONZERO] == 2 * 9, (
            f"expected 2 nonzero calls per step over 9 steps, got "
            f"{counter.counts[NONZERO]}\n{counter.report()}"
        )

    def test_counter_would_catch_a_reintroduced_boolean_mask(self):
        """Prove the detector detects. Otherwise T6 could pass vacuously.

        A counter that silently stopped classifying ``aten.index.Tensor`` would
        make every assertion above trivially true, so exercise it on a known
        boolean-mask read and a known masked write.
        """
        counter = SyncCounter()
        with counter:
            x = torch.randn(4, 3)
            mask = torch.tensor([True, False, True, False])
            _ = x[mask]
            x[mask] = torch.zeros(2, 3)
        assert counter.bool_mask_ops == 2, counter.report()
        assert set(counter.counts) <= set(BOOL_MASK_LABELS) | {NONZERO}


# --------------------------------------------------------------------------- #
# The dtype hazard index_copy_ introduces, end to end.
# --------------------------------------------------------------------------- #


class _Float64Model:
    """A model returning float64 energies *and* float64 forces.

    This is the realistic custom-NNP case, and on the pre-M6 loop it **crashed**:
    ``smallest_fmax`` is allocated float32, ``fmax`` inherits float64 from the
    forces, and ``smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]``
    raised ``"Index put requires the source and destination dtypes match"``
    whenever two or more molecules reduced their force in the same step. With
    exactly one reducing molecule the value had ``numel() == 1``, took ATen's
    ``masked_fill_`` fast path, and silently cast -- so the failure was
    *batch-size dependent* and invisible in a single-molecule test.

    Casting explicitly at every ``index_copy_`` site fixes that as a side effect.
    This is the one place the rewrite is deliberately **not** bit-identical to
    its predecessor: it succeeds where the predecessor raised.
    """

    def forward_batched(self, coord, numbers, charges, atom_mask=None):
        energy = (coord.to(torch.float64) ** 2).sum(dim=(1, 2))
        forces = -2.0 * coord.to(torch.float64)
        return energy, forces


def test_float64_model_outputs_do_not_raise():
    """A float64 NNP runs to completion and ``state`` dtypes are preserved."""
    state = _make_state(_Float64Model(), 4, 5, seed=0)
    n_steps(state, n=400, opttol=0.01, patience=5000)

    assert state["coord"].dtype is torch.float32
    assert state["fmax"].dtype is torch.float32
    assert state["energy"].dtype is torch.float64
    assert state["oscillating_count"].dtype is torch.long
    assert torch.isfinite(state["coord"]).all()
    # Not merely "did not raise": the run must actually reach the minimum, so
    # the multi-molecule force-reduction path (the one that used to raise) is
    # genuinely exercised many times over.
    assert state["converged_mask"].all()
