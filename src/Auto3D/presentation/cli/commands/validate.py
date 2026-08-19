# src/Auto3D/cli/commands/validate.py
"""Input file validation command."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from Auto3D.foundation.exceptions import InputValidationError
from Auto3D.presentation.cli.console import console, emit_json
from Auto3D.presentation.cli.errors import handle_error


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
                errors.append((i, line[:50], "Missing molecule ID (expected 'SMILES ID')"))
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


def execute_validate(input_file: Path, json_output: bool = False, verbose: int = 0) -> None:
    """Validate an input file.

    Every failure leaves through ``handle_error``, so this command's exit
    codes come from ``exit_code_for`` like every other command's. It had no
    error handling at all before: a bad file format and a file full of
    unparseable SMILES both raised a hard-coded ``SystemExit(1)``, and
    anything the reader itself raised -- a non-UTF-8 ``.smi``, say -- escaped
    as a raw ``UnicodeDecodeError`` traceback. A file the runner will reject
    is an ``InputValidationError`` (exit **2**), the same verdict and the same
    code ``auto3d run`` gives the same file, which is the entire point of a
    pre-flight checker.

    Args:
        input_file: Path to the .smi/.sdf file to check.
        json_output: Emit the result as a JSON document on stdout instead of
            a Rich panel. Every other command that produces a result already
            had ``--json``; this one did not, so it was the single hole in
            ``auto3d validate x.smi --json && auto3d run ...`` pipelines.
        verbose: CLI verbosity, forwarded to ``handle_error`` so an
            unexpected internal failure can be turned into a traceback with
            ``-v`` -- the panel tells the user to do exactly that.
    """
    # Tracks whether stdout has already received this command's own JSON
    # document. `handle_error` emits a *second* (failure-shaped) document when
    # told to, and two documents on one stream is not parseable JSON -- so the
    # richer validate document, which names every bad line, wins wherever it
    # was already written.
    document_emitted = False
    try:
        suffix = input_file.suffix.lower()

        # The status spinner is a Live render on the document stream, so it
        # must not run while stdout is reserved for a JSON document.
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
                # The panel still goes to stderr, where diagnostics belong,
                # but a --json caller must get a parseable document on stdout
                # on this path too rather than an empty stream.
                if json_output:
                    emit_json(
                        {
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
                        }
                    )
                    document_emitted = True
                raise InputValidationError(
                    f"Unsupported file format: {suffix}",
                    hint="Supported formats: .smi, .sdf",
                )

        if json_output:
            emit_json(_validate_json_document(input_file, result))
            document_emitted = True
        elif result.valid:
            console.print(
                Panel(
                    f"[green]Valid {result.file_format} file[/green]\n\n"
                    f"Molecules: {result.total_count}\n"
                    f"All entries parsed successfully",
                    title="Validation Passed",
                    border_style="green",
                )
            )
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

            console.print(
                Panel(
                    f"[red]{len(result.errors)} invalid entries found[/red]\n\n"
                    f"Valid: {result.valid_count}/{result.total_count}",
                    title="Validation Failed",
                    border_style="red",
                )
            )
            console.print(error_table)
            if more_msg:
                console.print(more_msg)

        if not result.valid:
            raise InputValidationError(
                f"{len(result.errors)} invalid entries in {input_file} "
                f"({result.valid_count}/{result.total_count} valid).",
                # InputValidationError's class hint is "Run 'auto3d validate
                # <file>' to check your input file", which is absurd advice
                # to print at the end of `auto3d validate <file>`. The table
                # above already lists what is wrong and where.
                hint="",
            )
    except Exception as e:  # noqa: BLE001 - funnel everything to the error panel
        handle_error(e, verbose=verbose, json_output=json_output and not document_emitted)
