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

    # AIMNET - always available (bundled)
    table.add_row(
        "AIMNET",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[green]Available[/green]",
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

    console.print(table)
    console.print()
    console.print("[dim]Run 'auto3d models info <engine>' for details[/dim]")


ENGINE_INFO = {
    "AIMNET": {
        "name": "AIMNet2",
        "description": "State-of-the-art neural network potential with excellent speed and accuracy.",
        "elements": "H, C, N, O, F, Si, P, S, Cl, Br, I",
        "speed": "~35x faster than ANI2x",
        "accuracy": "Best for organic molecules",
        "reference": "https://github.com/isayevlab/AIMNet2",
        "notes": [
            "Default engine (recommended)",
            "Single model by default for speed",
            "Use --use-ensemble for highest accuracy",
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
