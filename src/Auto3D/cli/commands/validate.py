# src/Auto3D/cli/commands/validate.py
"""Input file validation command."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from Auto3D.cli.console import console, emit_json, print_error


@dataclass
class ValidationResult:
    """Result of file validation."""

    valid: bool
    file_format: str
    total_count: int
    valid_count: int
    errors: list[tuple[int, str, str]]  # (line_num, content, error_msg)


def validate_smiles_file(file_path: Path) -> ValidationResult:
    """Validate a SMILES file.

    Must reject exactly what the runner (encode_ids/iter_smi_records,
    file_ops.py) rejects, or a passing `auto3d validate` is not trustworthy
    (M25): a line needs both a SMILES and an ID (whitespace-separated), and
    '#'-prefixed lines are treated as comments -- both checks mirrored from
    file_ops.iter_smi_records so the two never drift apart again.
    """
    from rdkit import Chem

    errors: list[tuple[int, str, str]] = []
    valid_count = 0
    total_count = 0

    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            total_count += 1
            parts = line.split()
            if len(parts) < 2:
                errors.append(
                    (i, line[:50], "Missing molecule ID (expected 'SMILES ID')")
                )
                continue
            smiles = parts[0]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors.append((i, smiles[:50], "Invalid SMILES"))
            else:
                valid_count += 1

    return ValidationResult(
        valid=len(errors) == 0,
        file_format="SMI",
        total_count=total_count,
        valid_count=valid_count,
        errors=errors,
    )


def validate_sdf_file(file_path: Path) -> ValidationResult:
    """Validate an SDF file."""
    from rdkit import Chem

    errors: list[tuple[int, str, str]] = []
    valid_count = 0
    total_count = 0

    suppl = Chem.SDMolSupplier(str(file_path), removeHs=False)
    for i, mol in enumerate(suppl, 1):
        total_count += 1
        if mol is None:
            errors.append((i, f"Molecule #{i}", "Failed to parse"))
        else:
            valid_count += 1

    return ValidationResult(
        valid=len(errors) == 0,
        file_format="SDF",
        total_count=total_count,
        valid_count=valid_count,
        errors=errors,
    )


def _validate_json_document(input_file: Path, result: ValidationResult) -> dict:
    """Build the ``--json`` document for a completed validation.

    Same shape as every other ``--json`` document (``success`` first, see
    ``cli/commands/properties.py::_report``). Unlike the human table, the
    error list is *not* truncated to the first ten: a person reading a
    terminal wants a summary, a program parsing stdout wants the whole
    finding, and silently emitting a prefix of it would make `auto3d validate
    --json` disagree with itself about how many entries are broken.
    """
    return {
        "success": result.valid,
        "command": "validate",
        "input_file": str(input_file),
        "format": result.file_format,
        "molecules": result.total_count,
        "valid_molecules": result.valid_count,
        "errors": [
            {"line": line_num, "content": content, "error": msg}
            for line_num, content, msg in result.errors
        ],
    }


def execute_validate(input_file: Path, json_output: bool = False) -> None:
    """Validate an input file.

    Args:
        input_file: Path to the .smi/.sdf file to check.
        json_output: Emit the result as a JSON document on stdout instead of
            a Rich panel. Every other command that produces a result already
            had ``--json``; this one did not, so it was the single hole in
            ``auto3d validate x.smi --json && auto3d run ...`` pipelines.
    """
    suffix = input_file.suffix.lower()

    # The status spinner is a Live render on the document stream, so it must
    # not run while stdout is reserved for a JSON document.
    spinner = (
        contextlib.nullcontext()
        if json_output
        else console.status("[bold]Validating input file...")
    )
    with spinner:
        if suffix == ".smi":
            result = validate_smiles_file(input_file)
        elif suffix == ".sdf":
            result = validate_sdf_file(input_file)
        else:
            # The panel still goes to stderr, where diagnostics belong, but a
            # --json caller must get a parseable document on stdout on this
            # path too rather than an empty stream.
            if json_output:
                emit_json({
                    "success": False,
                    "command": "validate",
                    "input_file": str(input_file),
                    "format": suffix.lstrip(".").upper(),
                    "molecules": 0,
                    "valid_molecules": 0,
                    "errors": [
                        {
                            "line": 0,
                            "content": str(input_file),
                            "error": f"Unsupported file format: {suffix}",
                        }
                    ],
                })
            print_error(
                f"Unsupported file format: {suffix}",
                hint="Supported formats: .smi, .sdf",
            )
            raise SystemExit(1)

    if json_output:
        emit_json(_validate_json_document(input_file, result))
        if not result.valid:
            raise SystemExit(1)
        return

    if result.valid:
        console.print(Panel(
            f"[green]Valid {result.file_format} file[/green]\n\n"
            f"Molecules: {result.total_count}\n"
            f"All entries parsed successfully",
            title="Validation Passed",
            border_style="green",
        ))
    else:
        error_table = Table(show_header=True, header_style="bold red")
        error_table.add_column("Line")
        error_table.add_column("Content")
        error_table.add_column("Error")

        for line_num, content, msg in result.errors[:10]:
            error_table.add_row(str(line_num), content, msg)

        more_msg = ""
        if len(result.errors) > 10:
            more_msg = f"\n[dim]... and {len(result.errors) - 10} more errors[/dim]"

        console.print(Panel(
            f"[red]{len(result.errors)} invalid entries found[/red]\n\n"
            f"Valid: {result.valid_count}/{result.total_count}",
            title="Validation Failed",
            border_style="red",
        ))
        console.print(error_table)
        if more_msg:
            console.print(more_msg)

        raise SystemExit(1)
