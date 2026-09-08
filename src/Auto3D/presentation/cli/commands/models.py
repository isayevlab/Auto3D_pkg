"""Model information commands."""

from __future__ import annotations

import importlib

from rich.panel import Panel
from rich.table import Table

from Auto3D.engines.models.policy import ANI_ELEMENTS, check_gpu_requested
from Auto3D.engines.models.species import format_elements
from Auto3D.foundation.exceptions import ConfigurationError, NumericalError
from Auto3D.presentation.cli.console import console
from Auto3D.presentation.cli.errors import handle_error

#: What ANI2x and ANI2xt accept, quoted from the gate that enforces it rather
#: than retyped -- see ``format_elements``. Rendering at import pulls rdkit in
#: with this module; that is a knowing choice and costs nothing, since rdkit is a
#: required dependency and the CLI already imports torch.
#:
#: The four AIMNet2 entries below stay literals on purpose. Their source of truth
#: is each model file's own ``implemented_species`` metadata, and deriving them
#: here would mean loading four NNPs to print a help table.
#: ``tests/test_element_sets.py`` pins them to that metadata in the slow tier
#: instead.
_ANI_ELEMENT_STRING = format_elements(ANI_ELEMENTS)


def check_dependency_status(name: str) -> tuple[bool, str]:
    """Report whether an optional dependency is importable, and its version.

    The probe is by import, for whatever name it is handed. This used to
    special-case ``"torchani"`` and return an unconditional
    ``(True, "Available")`` for anything else -- an answer arrived at without
    looking. That was unreachable while torchani was the only optional engine,
    but it made the next entry in the engine table silently report itself
    installed.
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        return False, "[yellow]Not installed[/yellow]"
    except Exception as exc:  # noqa: BLE001 - a status probe must not raise
        # Installed but unusable is a third state, and it is common for
        # CUDA-linked packages: torchani can raise OSError or RuntimeError from
        # a broken driver or a version mismatch. `auto3d models list` exists to
        # report status, so letting that propagate would kill the one command a
        # user runs to find out what is wrong.
        return False, f"[red]Import failed: {type(exc).__name__}[/red]"
    version = getattr(module, "__version__", None)
    return True, f"[green]v{version}[/green]" if version else "[green]Available[/green]"


def execute_models_list() -> None:
    """Display available optimization engines."""
    table = Table(title="Available Optimization Engines", show_header=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Networks/step", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Status")

    # AIMNET (alias for the aimnet2 registry default) - auto-downloaded on first use
    table.add_row(
        "AIMNET",
        "1",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )

    # ANI2x - requires torchani
    _, ani_status = check_dependency_status("torchani")
    table.add_row(
        "ANI2x",
        "[yellow]8 (ensemble)[/yellow]",
        "[green]★★★★☆[/green]",
        ani_status,
    )

    # ANI2xt - requires torchani
    table.add_row(
        "ANI2xt",
        "1",
        "[yellow]★★★☆☆[/yellow]",
        ani_status,
    )

    # AIMNet2 registry families - auto-downloaded on first use
    table.add_section()
    table.add_row(
        "aimnet2-2025",
        "1",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )
    table.add_row(
        "aimnet2-nse",
        "1",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )
    table.add_row(
        "aimnet2-pd",
        "1",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )

    console.print(table)
    console.print()
    console.print("[dim]aimnet2-2025: B97-3c, improved non-covalent interactions[/dim]")
    console.print(
        "[dim]aimnet2-nse: open-shell chemistry model (Auto3D uses default multiplicity)[/dim]"
    )
    console.print("[dim]aimnet2-pd: palladium catalysis[/dim]")
    console.print(
        "[dim]Any aimnet registry name is accepted; registry models are "
        "downloaded on first use into ~/.cache/aimnet[/dim]"
    )
    console.print(
        "[dim]Networks/step is how many networks the engine evaluates per "
        "optimization step -- no engine speed benchmark is maintained in "
        "this repository.[/dim]"
    )
    console.print()
    console.print("[dim]Run 'auto3d models info <engine>' for details[/dim]")


ENGINE_INFO = {
    "AIMNET": {
        "name": "AIMNet2",
        "description": "State-of-the-art neural network potential for organic and main-group chemistry.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "Single model (not benchmarked here)",
        "accuracy": "Best for organic molecules",
        "reference": "https://github.com/isayevlab/AIMNet2",
        "notes": [
            "Default engine (recommended)",
            "A single registry model is used; pick aimnet2-2025/-nse/-pd for different chemistry",
        ],
    },
    "AIMNET2-2025": {
        "name": "aimnet2-2025",
        "description": "AIMNet2 model trained on B97-3c with improved non-covalent interactions.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "Single model (not benchmarked here)",
        "accuracy": "Improved non-covalent interactions over the default model",
        "reference": "https://github.com/isayevlab/aimnetcentral",
        "notes": [
            "Registry model: downloaded on first use into ~/.cache/aimnet",
            "B97-3c level of theory",
            "Recommended when non-covalent interactions matter",
        ],
    },
    "AIMNET2-NSE": {
        "name": "aimnet2-nse",
        "description": "AIMNet2 model trained with open-shell chemistry. Auto3D evaluates all models at default multiplicity (singlet).",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "Single model (not benchmarked here)",
        "accuracy": "Trained on open-shell species but Auto3D does not yet expose spin-state control",
        "reference": "https://github.com/isayevlab/aimnetcentral",
        "notes": [
            "Registry model: downloaded on first use into ~/.cache/aimnet",
            "Model supports open-shell chemistry but Auto3D currently evaluates at default (singlet) multiplicity",
            "Spin-state control is not yet available through the Auto3D API",
        ],
    },
    "AIMNET2-PD": {
        "name": "aimnet2-pd",
        "description": "AIMNet2 model with palladium support for organometallic catalysis.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I",
        "speed": "Single model (not benchmarked here)",
        "accuracy": "Best for Pd organometallic / catalytic systems",
        "reference": "https://github.com/isayevlab/aimnetcentral",
        "notes": [
            "Registry model: downloaded on first use into ~/.cache/aimnet",
            "Replaces As with Pd vs the standard AIMNet2 element set",
            "Only transition metal supported is Pd",
        ],
    },
    "ANI2X": {
        "name": "ANI-2x",
        "description": "Accurate neural network potential for organic molecules.",
        "elements": _ANI_ELEMENT_STRING,
        "speed": "8-model ensemble (not benchmarked here)",
        "accuracy": "Excellent for covered elements",
        "reference": "https://github.com/aiqm/torchani",
        "notes": [
            "Requires torchani: pip install torchani",
            "8-model ensemble",
            "Well-validated for drug-like molecules",
        ],
    },
    "ANI2XT": {
        "name": "ANI-2xt",
        "description": "Extended ANI-2x with improved torsion handling.",
        "elements": _ANI_ELEMENT_STRING,
        "speed": "Single model (not benchmarked here)",
        "accuracy": "Good for conformer generation",
        "reference": "https://github.com/aiqm/torchani",
        "notes": [
            "Requires torchani: pip install torchani",
            "Single model",
            "Optimized for conformer search",
        ],
    },
}


def execute_models_info(engine: str, verbose: int = 0) -> None:
    """Display detailed information about an engine.

    An unrecognized engine name is the same user mistake ``resolve_engine_name``
    already reports as a ``ConfigurationError`` (exit 2) from ``auto3d run``
    and ``auto3d energy``; it exited 1 here only because this raise site
    predated the funnel.
    """
    try:
        engine_upper = engine.upper()
        # Any aimnet2-* registry name describes an AIMNet2 model. Variants with
        # their own ENGINE_INFO block (aimnet2-2025/-nse/-pd) are shown
        # directly; any other aimnet2* name (e.g. the bare 'aimnet2' alias or a
        # future registry variant) falls back to the base AIMNET entry instead
        # of printing "Unknown engine".
        if engine_upper not in ENGINE_INFO and engine_upper.startswith("AIMNET2"):
            engine_upper = "AIMNET"

        if engine_upper not in ENGINE_INFO:
            from Auto3D.engines.model_factory import ModelFactory

            raise ConfigurationError(
                f"Unknown engine: {engine}",
                hint=f"Available: {', '.join(ModelFactory.available_models())}",
            )
    except Exception as e:  # noqa: BLE001 - funnel everything to the error panel
        handle_error(e, verbose=verbose)

    info = ENGINE_INFO[engine_upper]

    content = f"""[bold]{info["name"]}[/bold]

{info["description"]}

[bold]Supported Elements:[/bold] {info["elements"]}

[bold]Speed:[/bold] {info["speed"]}
[bold]Accuracy:[/bold] {info["accuracy"]}

[bold]Reference:[/bold] {info["reference"]}

[bold]Notes:[/bold]
"""
    for note in info["notes"]:
        content += f"  - {note}\n"

    console.print(Panel(content, title=f"[cyan]{engine_upper}[/cyan]", border_style="blue"))


def execute_models_test(engine: str, gpu: bool = True, gpu_idx: int = 0, verbose: int = 0) -> None:
    """Health-check an engine: load it and run one tiny forward pass.

    Catches the common environment problems up front -- a missing torchani for
    ANI, a failed/blocked aimnet registry download, or a broken custom model
    file -- instead of having them surface deep inside a run.
    """
    # handle_error is already imported at module level (used identically by
    # execute_models_info above); this used to re-import it locally too, a
    # dead duplicate of the same binding.
    from Auto3D.presentation.cli.console import print_success

    try:
        import time

        import torch

        from Auto3D.engines.model_factory import create_model, get_device

        # `energy`/`optimize`/`thermo` (cli/commands/properties.py) already call
        # this before doing any work; `models test` reached
        # model_factory.get_device directly and silently fell back to CPU on a
        # CPU-only box instead of failing the same way -- the last M23 gap.
        check_gpu_requested(gpu)
        device = get_device(gpu_idx, use_gpu=gpu)
        with console.status(f"[bold]Loading {engine} on {device}..."):
            t0 = time.time()
            adapter = create_model(engine, device)
            # A single methane molecule (H, C only -> supported by every engine).
            coords = torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.63, 0.63, 0.63],
                        [-0.63, -0.63, 0.63],
                        [0.63, -0.63, -0.63],
                        [-0.63, 0.63, -0.63],
                    ]
                ],
                dtype=torch.float,
                device=device,
            )
            # Build species in the engine's own convention, asked of the
            # adapter that will consume them. Passing raw atomic numbers made the
            # ANI2xt check evaluate a Cl+4C species and report success (audit
            # C4); asking a name-keyed helper instead of the model left the
            # convention and the model as two independently-resolved things.
            species = torch.tensor([adapter.to_species([6, 1, 1, 1, 1])], device=device)
            charges = torch.tensor([0.0], device=device)
            energy, forces = adapter.forward(coords, species, charges)
            elapsed = time.time() - t0

        if not (torch.isfinite(energy).all() and torch.isfinite(forces).all()):
            raise NumericalError(
                f"{engine} produced non-finite energy/forces on the test molecule."
            )

        e_ev = float(energy.reshape(-1)[0])
        print_success(
            f"{engine} is working on {device} "
            f"(methane E = {e_ev:.4f} eV, {elapsed:.1f}s incl. load)."
        )
    except Exception as e:  # noqa: BLE001 - present every failure as a clean panel
        handle_error(e, verbose=verbose)
