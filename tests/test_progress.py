# tests/test_progress.py
"""Unit tests for the live-progress plumbing: the count helper that feeds events,
the display's aggregation, and what the two of them actually render.

No neural network potential is ever loaded here. The one test that drives the
real optimization loop (``n_steps``) does it against a stub force field whose
forces are chosen so the *exact* number of converged structures is known in
advance -- which is the point, because the defect this file now pins is a
progress report whose numbers did not correspond to any real work.
"""

from __future__ import annotations

import io

import torch
from rich.console import Console

from Auto3D.batch_opt.optimization_engine import n_steps, optimization_counts
from Auto3D.cli.progress import OptimizationDisplay


def _render(panel) -> str:
    """Render a Rich renderable to plain text, with styling forced off.

    ``color_system=None`` is load-bearing, not decoration. Writing to a
    non-tty ``StringIO`` does NOT by itself guarantee plain text: Rich also
    honors ``FORCE_COLOR``, which GitHub Actions sets, so under CI this
    helper emitted escapes and every ``assert "Converged 3" in rendered``
    below failed while passing locally. An earlier version of this docstring
    claimed "no ANSI: the file is not a tty", which was the guarantee it did
    not provide.
    """
    console = Console(file=io.StringIO(), width=100, color_system=None)
    console.print(panel)
    return console.file.getvalue()


def test_optimization_counts():
    # 5 structures; converged_mask marks 3 as done; one of those (osc>=patience)
    # is a drop, so converged=2, dropped=1, active=5-3=2.
    state = {
        "numbers": torch.zeros(5, 3),
        "converged_mask": torch.tensor([True, True, False, False, True]),
        "oscillating_count": torch.tensor([0, 5, 0, 0, 1]),
    }
    assert optimization_counts(state, patience=3) == (5, 2, 1, 2)


def test_optimization_counts_none_converged():
    state = {
        "numbers": torch.zeros(4, 3),
        "converged_mask": torch.tensor([False, False, False, False]),
        "oscillating_count": torch.tensor([0, 0, 0, 0]),
    }
    assert optimization_counts(state, patience=3) == (4, 0, 0, 4)


def test_display_single_job():
    d = OptimizationDisplay(0)
    d.update_from_jobs({1: {"total": 5, "converged": 2, "dropped": 1, "active": 2, "step": 30}})
    assert (d.total, d.converged, d.dropped, d.active, d.step) == (5, 2, 1, 2, 30)
    d.make_panel()  # must not raise


def test_display_multi_job_aggregates():
    d = OptimizationDisplay(0)
    d.update_from_jobs(
        {
            1: {"total": 5, "converged": 2, "dropped": 1, "active": 2, "step": 30},
            2: {"total": 3, "converged": 3, "dropped": 0, "active": 0, "step": 50},
        }
    )
    assert d.total == 8
    assert d.converged == 5
    assert d.dropped == 1
    assert d.active == 2
    assert d.step == 50  # furthest along


def test_display_empty_jobs_is_noop():
    """An empty ``jobs`` dict must leave the prior counts untouched, not reset
    them to zero.

    Starting from an all-zero display and then checking for all-zero after
    ``update_from_jobs({})`` cannot tell a real no-op from a reset: summing
    over zero jobs also yields zero, so that assertion would pass even if the
    ``if not jobs: return`` guard in ``update_from_jobs`` were deleted
    outright. Seed nonzero state first so a reset is distinguishable from a
    no-op.
    """
    d = OptimizationDisplay(0)
    d.update_from_jobs({1: {"total": 5, "converged": 2, "dropped": 1, "active": 2, "step": 7}})
    before = (d.total, d.converged, d.dropped, d.active, d.step)
    assert before == (5, 2, 1, 2, 7)

    d.update_from_jobs({})
    assert (d.total, d.converged, d.dropped, d.active, d.step) == before


# --- what the numbers are measured against -----------------------------------


class _SpeciesForceNN:
    """Stub force field: force magnitude chosen per species, constant forever.

    Keyed on species rather than row index because ``n_steps`` gathers a SUBSET
    of the batch once molecules start converging, so row 1 of a later step is
    not molecule 1. Every force vector points along x, so ``f.norm(dim=-1)`` is
    the x-component exactly and ``fmax`` sits at an exactly known position
    relative to ``opttol`` for the whole run.
    """

    def __init__(self, force_by_species: dict[int, float]) -> None:
        self.force_by_species = force_by_species

    def forward_batched(self, coord, numbers, charges, atom_mask=None):
        e = torch.zeros(coord.shape[0])
        f = torch.zeros_like(coord)
        for row in range(coord.shape[0]):
            f[row, :, 0] = self.force_by_species[int(numbers[row, 0])]
        return e, f


def _state_with_known_convergence():
    """5 structures of which exactly 3 converge on the first step, 2 never do.

    Species 1 feels zero force (``fmax = 0 <= opttol``: converged immediately);
    species 6 feels a force of 5.0, three orders of magnitude above ``opttol``,
    and it never decreases, so those two stay active for the whole run.
    """
    numbers = torch.stack(
        [
            torch.full((3,), 1, dtype=torch.long),
            torch.full((3,), 6, dtype=torch.long),
            torch.full((3,), 1, dtype=torch.long),
            torch.full((3,), 6, dtype=torch.long),
            torch.full((3,), 1, dtype=torch.long),
        ]
    )
    return {
        "numbers": numbers,
        "charges": torch.zeros(5, dtype=torch.long),
        "coord": torch.zeros(5, 3, 3),
        "nn": _SpeciesForceNN({1: 0.0, 6: 5.0}),
        "converged_mask": torch.zeros(5, dtype=torch.bool),
        "fmax": torch.full((5,), 999.0),
        "energy": torch.full((5,), 999.0, dtype=torch.double),
    }


def test_reported_counts_equal_the_work_actually_done():
    """The events the optimizer emits must describe real structures.

    Known work: 5 structures in, exactly 3 of them converge (their force is
    zero), 2 never do, none is dropped (``patience`` is far above ``n``). Every
    event -- and therefore everything the panel can show -- must say exactly
    that, at every step it reports.

    This is the assertion the old display could not have satisfied. It reported
    a *fraction*: ``tqdm`` over ``range(1, n+1)`` divided the current step by
    the step budget, so this run (converging at step 1, then grinding to step
    40) would have shown 2.5% climbing to 100% while the count of converged
    structures never moved off 3.
    """
    state = _state_with_known_convergence()
    events: list[dict] = []

    # patience far above n: the constant force never decreases, so the
    # oscillation counter climbs every step and would otherwise drop the two
    # active structures and change the counts under test.
    n_steps(state, n=40, opttol=0.01, patience=10_000, progress_cb=events.append)

    assert events, "the optimizer emitted no progress events at all"
    for event in events:
        assert (event["total"], event["converged"], event["dropped"], event["active"]) == (
            5,
            3,
            0,
            2,
        ), f"event {event} does not describe the 5 structures actually optimized"
    assert events[-1]["step"] == 40


def test_the_panel_shows_those_counts_and_claims_no_fraction():
    """Feed the real events to the display: the panel must show the real
    numbers, and must not translate them into a percentage.

    There is no honest denominator at this layer (enumeration is still
    producing structures while the optimizer consumes the earlier ones), so the
    display reports counts only. ``%`` appearing here again would mean some
    denominator was reinvented.
    """
    state = _state_with_known_convergence()
    events: list[dict] = []
    n_steps(state, n=40, opttol=0.01, patience=10_000, progress_cb=events.append)

    display = OptimizationDisplay(0)
    display.update_from_jobs({0: events[-1]})
    rendered = _render(display.make_panel())

    assert "Converged 3" in " ".join(rendered.split())
    assert "Active 2" in " ".join(rendered.split())
    assert "Dropped 0" in " ".join(rendered.split())
    assert "5 structures in this batch" in " ".join(rendered.split())
    assert "%" not in rendered, f"the panel reported a fraction again:\n{rendered}"


def test_the_panel_does_not_run_backwards_across_successive_batches():
    """The sawtooth, pinned.

    One worker optimizes chunk after chunk, and ``n_steps`` starts over for
    each: a fresh ``total`` and the step back at 1. The old panel divided
    converged by the current batch's total, so a worker finishing a 4-structure
    chunk (100%) and picking up a 100-structure chunk showed ``100% -> 2%``.

    The fix is that no fraction is reported, so there is nothing to run
    backwards; what the panel shows is the batch in flight, correctly.
    """
    display = OptimizationDisplay(0)

    display.update_from_jobs(
        {0: {"total": 4, "converged": 4, "dropped": 0, "active": 0, "step": 210}}
    )
    finished_small_chunk = _render(display.make_panel())

    display.update_from_jobs(
        {0: {"total": 100, "converged": 2, "dropped": 0, "active": 98, "step": 10}}
    )
    fresh_big_chunk = _render(display.make_panel())

    assert "%" not in finished_small_chunk
    assert "%" not in fresh_big_chunk
    assert "Converged 2" in " ".join(fresh_big_chunk.split())
    assert "100 structures in this batch" in " ".join(fresh_big_chunk.split())


def test_the_optimizer_writes_no_progress_bar_to_stderr(capsys):
    """``n_steps`` must not paint a bar over the step budget.

    The bar it used to paint was wrong twice over: it measured the step budget
    rather than the work (a run converging at step 300 of 2000 showed 15% and
    then vanished; a run where nothing converged reached 100%), and ``tqdm``
    only auto-disables on ``disable=None``, so it wrote carriage returns into
    every redirected stderr, log file and CI transcript regardless.

    Keyed on the control characters and the rate suffix rather than on a
    percentage, because those are what a bar unavoidably emits and what a
    redirected log unavoidably collects.
    """
    state = _state_with_known_convergence()
    n_steps(state, n=40, opttol=0.01, patience=10_000)

    err = capsys.readouterr().err
    assert "\r" not in err, f"a progress bar repainted stderr: {err!r}"
    assert "it/s" not in err, f"a progress bar wrote a rate to stderr: {err!r}"


# --- which stream the panel lands on ------------------------------------------


def _plain(text: str) -> str:
    """Strip ANSI and collapse whitespace.

    Both are load-bearing. Rich colorizes whenever it believes it is on a
    terminal (``FORCE_COLOR`` in CI is enough), and it styles ``Converged`` and
    its count separately -- so the escape sequence lands *between* the two and a
    naive ``"Converged 4" in out`` passes locally and fails in CI. Collapsing
    whitespace afterwards absorbs the panel's own column padding and wrapping.
    """
    import re

    return " ".join(re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text).split())


def test_the_live_panel_goes_to_stderr_and_never_to_stdout(tmp_path, monkeypatch):
    """Progress belongs on stderr; stdout carries the result.

    The parent ``Live`` used to render onto the CLI's reserved stdout while the
    child optimizer's own status went to stderr. Two consequences, both fixed
    here: under a pty the two interleaved and tore the panel border apart, and
    ``auto3d run > log`` -- the case where a live panel matters most, because
    stdout is not on screen -- filed the panel into the log and showed the user
    nothing.

    It also kept stdout non-empty for the duration of a run, on the same stream
    ``--json`` promises carries only the document.

    Hermetic: ``Auto3D.auto3D.main`` is replaced by a stub that fires one
    progress event and returns a clean result, so no potential is loaded.
    """
    from typer.testing import CliRunner

    import Auto3D.auto3D as a3d
    from Auto3D.cli.app import app
    from Auto3D.results import WorkflowResult

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")
    out = tmp_path / "out.sdf"

    def fake_main(options, progress_callback=None, **kwargs):
        assert progress_callback is not None, "interactive run passed no progress callback"
        progress_callback(
            {"job": 0, "total": 7, "converged": 4, "dropped": 1, "active": 2, "step": 120}
        )
        return WorkflowResult(str(out), failures=[])

    monkeypatch.setattr(a3d, "main", fake_main)

    # ANI2xt, not the default AIMNET: that name short-circuits engine
    # resolution in resolve_engine_name's first branch, so nothing imports
    # `aimnet` (and through it `warp`, which initializes every visible CUDA
    # device at import time) for a test that never optimizes anything.
    result = CliRunner().invoke(
        app, ["run", str(smi), "--k", "1", "--no-gpu", "--engine", "ANI2xt"]
    )
    assert result.exit_code == 0, result.output

    err, out_text = _plain(result.stderr), _plain(result.stdout)

    assert "Optimizing" in err
    assert "Converged 4" in err
    assert "Active 2" in err

    assert "Optimizing" not in out_text, f"the live panel reached stdout:\n{out_text}"
    assert "Converged" not in out_text, f"the live panel reached stdout:\n{out_text}"
    # Control: the results summary is what stdout is *for*, so a test that
    # passed simply because nothing at all reached stdout would be vacuous.
    assert "Molecules:" in out_text
