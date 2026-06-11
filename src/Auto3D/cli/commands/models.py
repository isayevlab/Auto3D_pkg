# src/Auto3D/cli/commands/models.py
"""Model information commands."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from Auto3D.cli.console import console, print_error


def check_dependency_status(name: str) -> tuple[bool, str]:
    """Check if an optional dependency is available."""
    if name == "torchani":
        try:
            import torchani
            return True, f"[green]v{torchani.__version__}[/green]"
        except ImportError:
            return False, "[yellow]Not installed[/yellow]"
    return True, "[green]Available[/green]"


def execute_models_list() -> None:
    """Display available optimization engines."""
    table = Table(title="Available Optimization Engines", show_header=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Speed", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Status")

    # AIMNET (alias for the aimnet2 registry default) - auto-downloaded on first use
    table.add_row(
        "AIMNET",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )

    # ANI2x - requires torchani
    ani_available, ani_status = check_dependency_status("torchani")
    table.add_row(
        "ANI2x",
        "[yellow]★★★☆☆[/yellow]",
        "[green]★★★★☆[/green]",
        ani_status,
    )

    # ANI2xt - requires torchani
    table.add_row(
        "ANI2xt",
        "[yellow]★★★★☆[/yellow]",
        "[yellow]★★★☆☆[/yellow]",
        ani_status,
    )

    # AIMNet2 registry families - auto-downloaded on first use
    table.add_section()
    table.add_row(
        "aimnet2-2025",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )
    table.add_row(
        "aimnet2-nse",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )
    table.add_row(
        "aimnet2-pd",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[cyan]Auto-download[/cyan]",
    )

    console.print(table)
    console.print()
    console.print(
        "[dim]aimnet2-2025: B97-3c, improved non-covalent interactions[/dim]"
    )
    console.print("[dim]aimnet2-nse: open-shell / radicals (spin support)[/dim]")
    console.print("[dim]aimnet2-pd: palladium catalysis[/dim]")
    console.print(
        "[dim]Any aimnet registry name is accepted; registry models are "
        "downloaded on first use into ~/.cache/aimnet[/dim]"
    )
    console.print()
    console.print("[dim]Run 'auto3d models info <engine>' for details[/dim]")


ENGINE_INFO = {
    "AIMNET": {
        "name": "AIMNet2",
        "description": "State-of-the-art neural network potential with excellent speed and accuracy.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "~35x faster than ANI2x",
        "accuracy": "Best for organic molecules",
        "reference": "https://github.com/isayevlab/AIMNet2",
        "notes": [
            "Default engine (recommended)",
            "Single model by default for speed",
            "Use --use-ensemble for highest accuracy",
        ],
    },
    "AIMNET2-2025": {
        "name": "aimnet2-2025",
        "description": "AIMNet2 model trained on B97-3c with improved non-covalent interactions.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "~35x faster than ANI2x",
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
        "description": "AIMNet2 model with open-shell support for radicals and spin states.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I",
        "speed": "~35x faster than ANI2x",
        "accuracy": "Best for open-shell / radical species",
        "reference": "https://github.com/isayevlab/aimnetcentral",
        "notes": [
            "Registry model: downloaded on first use into ~/.cache/aimnet",
            "Adds open-shell (spin) support for radicals",
            "Use when modeling radicals or non-singlet spin states",
        ],
    },
    "AIMNET2-PD": {
        "name": "aimnet2-pd",
        "description": "AIMNet2 model with palladium support for organometallic catalysis.",
        "elements": "H, B, C, N, O, F, Si, P, S, Cl, Se, Br, Pd, I",
        "speed": "~35x faster than ANI2x",
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
        "elements": "H, C, N, O, F, S, Cl",
        "speed": "Moderate",
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
        "elements": "H, C, N, O, F, S, Cl",
        "speed": "Faster than ANI2x",
        "accuracy": "Good for conformer generation",
        "reference": "https://github.com/aiqm/torchani",
        "notes": [
            "Requires torchani: pip install torchani",
            "Single model (faster)",
            "Optimized for conformer search",
        ],
    },
}


def execute_models_info(engine: str) -> None:
    """Display detailed information about an engine."""
    engine_upper = engine.upper()

    if engine_upper not in ENGINE_INFO:
        print_error(
            f"Unknown engine: {engine}",
            hint=f"Available: {', '.join(ENGINE_INFO.keys())}",
        )
        raise SystemExit(1)

    info = ENGINE_INFO[engine_upper]

    content = f"""[bold]{info['name']}[/bold]

{info['description']}

[bold]Supported Elements:[/bold] {info['elements']}

[bold]Speed:[/bold] {info['speed']}
[bold]Accuracy:[/bold] {info['accuracy']}

[bold]Reference:[/bold] {info['reference']}

[bold]Notes:[/bold]
"""
    for note in info["notes"]:
        content += f"  - {note}\n"

    console.print(Panel(content, title=f"[cyan]{engine_upper}[/cyan]", border_style="blue"))
