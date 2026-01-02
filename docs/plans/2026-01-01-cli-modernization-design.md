# CLI Modernization Design

**Date:** 2026-01-01
**Status:** Approved
**Scope:** Full CLI modernization with Typer + Rich

## Overview

Modernize Auto3D's CLI from basic argparse to a full-featured modern CLI with:
- Typer for command parsing and subcommand structure
- Rich for visual output (progress bars, panels, tables, syntax highlighting)
- Pydantic for configuration validation
- Auto-detection of terminal capabilities

## Decisions Summary

| Aspect | Decision |
|--------|----------|
| Framework | Typer + Rich |
| Structure | Subcommands (run, config, models, validate) |
| Config | YAML-first with CLI overrides, Pydantic validation |
| Output | Auto-detect (Rich/plain/JSON) |
| Verbosity | Balanced default, -v/-q flags |
| Progress | Live panels with converged/active/dropped stats |
| Errors | Contextual panels with actionable hints |
| Compat | Legacy YAML mode with deprecation warning |

## Architecture

### Framework Stack

```
┌─────────────────────────────────────────────┐
│  CLI Layer (Typer)                          │
│  - Subcommand routing                       │
│  - Argument parsing & validation            │
│  - Type-safe options via hints              │
├─────────────────────────────────────────────┤
│  Console Layer (Rich)                       │
│  - Auto-detect terminal capabilities        │
│  - Progress bars, panels, tables            │
│  - Live status updates                      │
├─────────────────────────────────────────────┤
│  Config Layer                               │
│  - YAML loading with validation             │
│  - CLI override merging                     │
│  - Pydantic models for type safety          │
├─────────────────────────────────────────────┤
│  Existing Core (unchanged)                  │
│  - workflow.py, batch_opt, isomer_engine    │
│  - Just receives validated config dict      │
└─────────────────────────────────────────────┘
```

### New File Structure

```
src/Auto3D/
├── cli/
│   ├── __init__.py
│   ├── app.py           # Typer app, subcommand registration
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── run.py       # Main workflow command
│   │   ├── config.py    # Config management (init/show/validate)
│   │   └── models.py    # Model info commands
│   ├── console.py       # Rich console singleton, output helpers
│   ├── progress.py      # Progress bars, live status components
│   ├── results.py       # Results display formatting
│   ├── errors.py        # Error handling and display
│   └── config_schema.py # Pydantic models for config validation
├── auto3Dcli.py         # Slim entry point, imports from cli/
```

## Command Structure

```
auto3d
├── run <input>              # Main workflow (default if no subcommand)
├── config
│   ├── init                 # Generate default config file
│   ├── show [file]          # Display config with syntax highlighting
│   └── validate <file>      # Validate config without running
├── models
│   ├── list                 # Show available engines
│   └── info <name>          # Details about specific model
└── validate <file>          # Quick SMILES/SDF validation
```

### Command Signatures

```python
# run command - the workhorse
@app.command()
def run(
    input: Path,
    config: Annotated[Path, Option("-c", "--config")] = None,
    k: Annotated[int, Option("--k", help="Top-k conformers")] = None,
    window: Annotated[float, Option(help="Energy window kcal/mol")] = None,
    engine: Annotated[Engine, Option(help="AIMNET|ANI2x|ANI2xt")] = None,
    gpu: Annotated[bool, Option("--gpu/--no-gpu")] = None,
    gpu_idx: Annotated[str, Option(help="GPU index(es): 0 or 0,1,2")] = None,
    verbose: Annotated[int, Option("-v", "--verbose", count=True)] = 0,
    quiet: Annotated[bool, Option("-q", "--quiet")] = False,
    json_output: Annotated[bool, Option("--json")] = False,
): ...

# config init - generate starter config
@config_app.command("init")
def config_init(
    output: Annotated[Path, Option("-o")] = Path("auto3d.yaml"),
    engine: Annotated[Engine, Option()] = Engine.AIMNET,
    preset: Annotated[Preset, Option(help="quick|balanced|thorough")] = None,
): ...
```

### Shorthand Support

```bash
# These are equivalent:
auto3d run molecules.smi
auto3d molecules.smi          # Implicit 'run' for convenience
```

## Configuration Management

### Pydantic Schema

```python
# cli/config_schema.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from pathlib import Path

class Auto3DConfig(BaseModel):
    # Required
    path: Path

    # Output control
    k: int | None = Field(None, ge=1, description="Top-k conformers")
    window: float | None = Field(None, gt=0, description="Energy window kcal/mol")

    # Engine settings
    optimizing_engine: Literal["AIMNET", "ANI2x", "ANI2xt"] = "AIMNET"
    use_gpu: bool = True
    gpu_idx: int | list[int] = 0

    # Isomer settings
    enumerate_tautomer: bool = False
    tauto_engine: Literal["rdkit", "oechem"] = "rdkit"
    enumerate_isomer: bool = True
    isomer_engine: Literal["rdkit", "omega"] = "rdkit"

    # Optimization
    opt_steps: int = Field(2000, ge=1)
    convergence_threshold: float = Field(0.01, gt=0)
    patience: int = Field(250, ge=1)
    threshold: float = Field(0.3, gt=0, description="RMSD threshold")

    @field_validator("gpu_idx", mode="before")
    @classmethod
    def parse_gpu_idx(cls, v):
        if isinstance(v, str):
            return [int(x) for x in v.split(",")]
        return v
```

### Config Loading Priority

```
1. CLI arguments (highest priority)
2. Config file (-c auto3d.yaml)
3. Defaults from schema (lowest priority)
```

### Merge Logic

```python
def load_config(config_file: Path | None, cli_overrides: dict) -> Auto3DConfig:
    # Start with defaults
    config_dict = {}

    # Layer config file
    if config_file:
        config_dict.update(yaml.safe_load(config_file.read_text()))

    # Layer CLI overrides (only non-None values)
    config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

    # Validate and return
    return Auto3DConfig(**config_dict)
```

## Console & Output

### Console Singleton with Auto-Detection

```python
# cli/console.py
from rich.console import Console
import sys

def create_console() -> Console:
    """Auto-detect terminal capabilities."""
    force_terminal = None
    if not sys.stdout.isatty():
        force_terminal = False

    return Console(
        force_terminal=force_terminal,
        highlight=True,
        emoji=False,
    )

console = create_console()
```

### Output Helpers

```python
def print_error(message: str, hint: str | None = None):
    panel = Panel(
        f"[red]{message}[/red]" + (f"\n\n[dim]{hint}[/dim]" if hint else ""),
        title="[red]Error[/red]",
        border_style="red",
    )
    console.print(panel, stderr=True)

def print_success(message: str):
    console.print(f"[green]✓[/green] {message}")

def print_warning(message: str):
    console.print(f"[yellow]⚠[/yellow] {message}")
```

### Startup Banner

```python
def print_banner(config: Auto3DConfig):
    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()

    grid.add_row("Input:", f"[cyan]{config.path.name}[/cyan]")
    grid.add_row("Engine:", config.optimizing_engine)
    grid.add_row("GPU:", f"CUDA:{config.gpu_idx}" if config.use_gpu else "CPU")
    grid.add_row("Output:", f"k={config.k}" if config.k else f"window={config.window}")

    console.print(Panel(grid, title="[bold]Auto3D v3.0[/bold]", border_style="blue"))
```

### Verbosity Control

```python
class OutputManager:
    def __init__(self, verbosity: int, quiet: bool, json_mode: bool):
        self.verbosity = 0 if quiet else verbosity
        self.json_mode = json_mode
        self.events = []

    def status(self, msg: str, level: int = 0):
        if self.json_mode:
            self.events.append({"type": "status", "message": msg})
        elif self.verbosity >= level:
            console.print(f"[dim]│[/dim] {msg}")
```

## Progress Display

### Progress Bars

```python
# cli/progress.py
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
)

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
```

### Live Optimization Status

```python
class OptimizationDisplay:
    """Live-updating display during geometry optimization."""

    def __init__(self, total_structures: int):
        self.total = total_structures
        self.converged = 0
        self.active = total_structures
        self.dropped = 0
        self.step = 0
        self.best_energy = None

    def make_panel(self) -> Panel:
        pct = (self.converged + self.dropped) / self.total
        bar = "━" * int(pct * 30) + "╺" + "─" * (29 - int(pct * 30))

        stats = Table.grid(padding=(0, 3))
        stats.add_row(
            f"[green]✓ Converged[/green]",  f"[green]{self.converged}[/green]",
            f"[yellow]◉ Active[/yellow]",    f"[yellow]{self.active}[/yellow]",
            f"[red]✗ Dropped[/red]",         f"[red]{self.dropped}[/red]",
        )

        content = f"{bar} {pct:.0%}  Step {self.step}\n\n{stats}"
        if self.best_energy:
            content += f"\n\n[dim]Best energy: {self.best_energy:.2f} kcal/mol[/dim]"

        return Panel(content, title="[bold]Optimizing[/bold]", border_style="blue")

    def update(self, converged: int, active: int, dropped: int, step: int, best_e: float):
        self.converged = converged
        self.active = active
        self.dropped = dropped
        self.step = step
        self.best_energy = best_e
```

### Callback Adapter

```python
class ProgressCallback:
    def __init__(self, display: OptimizationDisplay, live: Live):
        self.display = display
        self.live = live

    def on_step(self, stats: dict):
        self.display.update(**stats)
        self.live.update(self.display.make_panel())
```

## Results Display

### Summary Panel

```python
def print_results_summary(results: WorkflowResults):
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()

    stats.add_row("Molecules:", f"[green]{results.success_count}[/green] succeeded")
    if results.failed_count:
        stats.add_row("", f"[red]{results.failed_count}[/red] failed")
    stats.add_row("Conformers:", f"{results.total_conformers} generated")
    stats.add_row("Output:", f"[cyan]{results.output_path}[/cyan]")
    stats.add_row("Time:", format_duration(results.elapsed_seconds))

    console.print(Panel(stats, title="[bold green]Results[/bold green]", border_style="green"))
```

### Detailed View (with -v flag)

```python
def print_detailed_results(results: WorkflowResults):
    tree = Tree("[bold]Output Summary[/bold]")

    for mol in results.molecules[:10]:
        mol_branch = tree.add(f"[cyan]{mol.name}[/cyan] ({mol.conformer_count} conformers)")
        for conf in mol.conformers[:3]:
            energy_style = "green" if conf.is_lowest else "dim"
            mol_branch.add(f"[{energy_style}]{conf.energy:+.2f} kcal/mol[/{energy_style}]")
        if mol.conformer_count > 3:
            mol_branch.add(f"[dim]... +{mol.conformer_count - 3} more[/dim]")

    if len(results.molecules) > 10:
        tree.add(f"[dim]... +{len(results.molecules) - 10} more molecules[/dim]")

    console.print(tree)
```

### JSON Output Mode

```python
def output_json(results: WorkflowResults):
    data = {
        "success": results.failed_count == 0,
        "molecules": results.success_count,
        "conformers": results.total_conformers,
        "output_file": str(results.output_path),
        "elapsed_seconds": results.elapsed_seconds,
        "failures": [{"name": f.name, "error": f.error} for f in results.failures],
    }
    console.print_json(data=data)
```

## Error Handling

### Error Panel Display

```python
def handle_cli_error(error: Auto3DError):
    error_types = {
        ConfigurationError: ("Configuration Error", "yellow", config_hint),
        InputValidationError: ("Invalid Input", "red", input_hint),
        ModelNotFoundError: ("Model Not Found", "red", model_hint),
        GPUError: ("GPU Error", "yellow", gpu_hint),
        DependencyError: ("Missing Dependency", "yellow", dependency_hint),
    }

    title, color, hint_fn = error_types.get(
        type(error),
        ("Error", "red", lambda e: None)
    )

    content = f"[{color}]{error}[/{color}]"
    hint = hint_fn(error)
    if hint:
        content += f"\n\n[dim]{hint}[/dim]"

    console.print(Panel(content, title=f"[{color}]{title}[/{color}]", border_style=color))
    raise SystemExit(1)
```

### Contextual Hints

```python
def config_hint(e: ConfigurationError) -> str:
    return "Run 'auto3d config init' to generate a valid config file"

def input_hint(e: InputValidationError) -> str:
    return "Run 'auto3d validate <file>' to check your input file"

def model_hint(e: ModelNotFoundError) -> str:
    return "Available engines: AIMNET, ANI2x, ANI2xt\nRun 'auto3d models list' for details"

def gpu_hint(e: GPUError) -> str:
    return "Try --no-gpu to run on CPU, or check CUDA installation"

def dependency_hint(e: DependencyError) -> str:
    dep = e.dependency_name
    hints = {
        "openeye": "Install: conda install -c openeye openeye-toolkits",
        "torchani": "Install: pip install torchani",
        "ase": "Install: pip install ase",
    }
    return hints.get(dep, f"Install the missing dependency: {dep}")
```

## Utility Commands

### Models List

```python
@models_app.command("list")
def models_list():
    table = Table(title="Available Engines", show_header=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Speed")
    table.add_column("Accuracy")
    table.add_column("Status")

    table.add_row(
        "AIMNET",
        "[green]★★★★★[/green]",
        "[green]★★★★★[/green]",
        "[green]✓ Available[/green]"
    )
    table.add_row(
        "ANI2x",
        "[yellow]★★★☆☆[/yellow]",
        "[green]★★★★☆[/green]",
        check_torchani_status()
    )
    table.add_row(
        "ANI2xt",
        "[yellow]★★★★☆[/yellow]",
        "[yellow]★★★☆☆[/yellow]",
        check_torchani_status()
    )

    console.print(table)
```

### Config Init

```python
@config_app.command("init")
def config_init(
    output: Annotated[Path, Option("-o", "--output")] = Path("auto3d.yaml"),
    preset: Annotated[Preset, Option("-p", help="quick|balanced|thorough")] = None,
):
    presets = {
        "quick": {"k": 1, "opt_steps": 500, "patience": 100},
        "balanced": {"k": 5, "opt_steps": 2000},
        "thorough": {"k": 10, "window": 5.0, "opt_steps": 5000},
    }

    config = DEFAULT_CONFIG.copy()
    if preset:
        config.update(presets[preset])

    yaml_content = generate_commented_yaml(config)
    output.write_text(yaml_content)

    console.print(f"[green]✓[/green] Created [cyan]{output}[/cyan]")
    console.print(Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True))
```

## Dependencies

### New Dependencies

```toml
# pyproject.toml additions
dependencies = [
    # ... existing deps ...
    "typer>=0.12.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
]
```

## Backwards Compatibility

### Legacy Support

```python
# auto3Dcli.py
def cli():
    # Legacy: single YAML file argument
    if len(sys.argv) == 2 and _is_yaml_file(sys.argv[1]):
        console.print("[dim]Hint: New syntax is 'auto3d run -c config.yaml input.smi'[/dim]\n")
        return legacy_yaml_run(sys.argv[1])

    # New Typer app
    from Auto3D.cli.app import app
    app()
```

### Deprecation Warning

```python
def legacy_yaml_run(yaml_path: str):
    console.print(Panel(
        "[yellow]Deprecation Notice[/yellow]\n\n"
        "The 'auto3d config.yaml' syntax will be removed in v4.0\n\n"
        "New syntax:\n"
        "  auto3d run input.smi -c config.yaml",
        border_style="yellow"
    ))
```

### Environment Variable Support

```python
engine: Annotated[str, Option(
    envvar="AUTO3D_ENGINE",
    help="AIMNET|ANI2x|ANI2xt"
)] = "AIMNET"

gpu: Annotated[bool, Option(
    "--gpu/--no-gpu",
    envvar="AUTO3D_USE_GPU",
)] = True
```

### Shell Completion

```bash
# Enable with:
auto3d --install-completion bash  # or zsh/fish/powershell

# Usage:
auto3d r<TAB>        → auto3d run
auto3d run --e<TAB>  → auto3d run --engine
```

## Visual Examples

### Startup Banner

```
╭─ Auto3D v3.0 ──────────────────────────────────────────────╮
│  Input:   molecules.smi                                    │
│  Engine:  AIMNET                                           │
│  GPU:     CUDA:0                                           │
│  Output:  k=5                                              │
╰────────────────────────────────────────────────────────────╯
```

### Balanced Output (Default)

```
╭─ Auto3D v3.0 ──────────────────────────────────────────────╮
│ Input: molecules.smi (150 SMILES)                          │
│ Engine: AIMNET │ GPU: CUDA:0 │ k=5                         │
╰────────────────────────────────────────────────────────────╯

⠋ Enumerating stereoisomers...
✓ Generated 423 stereoisomers from 150 SMILES

Optimizing ━━━━━━━━━━━━━━━━━━━━━━ 100% 423/423  02:34

⚠ 3 molecules failed optimization (see --verbose for details)

╭─ Results ──────────────────────────────────────────────────╮
│ ✓ 147 molecules → 735 conformers                           │
│ ✗ 3 molecules failed                                       │
│ Output: molecules_out.sdf                                  │
│ Time: 2m 41s                                               │
╰────────────────────────────────────────────────────────────╯
```

### Live Optimization Panel

```
╭─ Optimizing ───────────────────────────────────────────────╮
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺───── 75%  Step 750         │
│                                                            │
│ ✓ Converged  45    ◉ Active  28    ✗ Dropped  7           │
│                                                            │
│ Best energy: -342.51 kcal/mol                              │
╰────────────────────────────────────────────────────────────╯
```

### Error Display

```
╭─ Configuration Error ──────────────────────────────────────╮
│  Invalid value for 'optimizing_engine': ANI3              │
│  Must be one of: AIMNET, ANI2x, ANI2xt                    │
│                                                            │
│  Run 'auto3d config init' to generate a valid config file │
╰────────────────────────────────────────────────────────────╯
```

### Models List

```
        Available Engines
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Engine  ┃ Speed   ┃ Accuracy ┃ Status        ┃
┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ AIMNET  │ ★★★★★   │ ★★★★★    │ ✓ Available   │
│ ANI2x   │ ★★★☆☆   │ ★★★★☆    │ ✓ Available   │
│ ANI2xt  │ ★★★★☆   │ ★★★☆☆    │ ✓ Available   │
└─────────┴─────────┴──────────┴───────────────┘
```
