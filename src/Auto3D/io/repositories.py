"""Repository pattern implementations for molecule file I/O."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol, runtime_checkable

from rdkit import Chem


@runtime_checkable
class MoleculeRepository(Protocol):
    """Protocol for molecule file repositories.

    Defines the interface for reading and writing molecular data
    in different file formats.
    """

    def read(self, path: str) -> Iterator[Chem.Mol]:
        """Read molecules from a file.

        Args:
            path: Path to the input file.

        Yields:
            RDKit Mol objects.
        """
        ...

    def write(self, path: str, molecules: list[Chem.Mol]) -> None:
        """Write molecules to a file.

        Args:
            path: Path to the output file.
            molecules: List of RDKit Mol objects to write.
        """
        ...


class SDFRepository:
    """Repository for SDF (Structure-Data File) format.

    Handles reading and writing molecules in SDF format with
    support for molecular properties.

    Example:
        >>> repo = SDFRepository()
        >>> mols = list(repo.read("input.sdf"))
        >>> repo.write("output.sdf", mols)
    """

    def __init__(self, remove_hs: bool = False) -> None:
        """Initialize the SDF repository.

        Args:
            remove_hs: Whether to remove hydrogens when reading.
        """
        self.remove_hs = remove_hs

    def read(self, path: str) -> Iterator[Chem.Mol]:
        """Read molecules from an SDF file.

        Args:
            path: Path to the SDF file.

        Yields:
            RDKit Mol objects (skips None/invalid molecules).
        """
        supplier = Chem.SDMolSupplier(path, removeHs=self.remove_hs)
        for mol in supplier:
            if mol is not None:
                yield mol

    def write(self, path: str, molecules: list[Chem.Mol]) -> None:
        """Write molecules to an SDF file.

        Args:
            path: Path to the output SDF file.
            molecules: List of RDKit Mol objects to write.
        """
        with Chem.SDWriter(path) as writer:
            for mol in molecules:
                if mol is not None:
                    writer.write(mol)

    def read_with_properties(
        self,
        path: str,
        properties: list[str] | None = None,
    ) -> Iterator[tuple[Chem.Mol, dict[str, str]]]:
        """Read molecules with their properties.

        Args:
            path: Path to the SDF file.
            properties: List of property names to extract. None for all.

        Yields:
            Tuples of (molecule, properties_dict).
        """
        for mol in self.read(path):
            if properties is None:
                props = {name: mol.GetProp(name) for name in mol.GetPropsAsDict()}
            else:
                props = {}
                for name in properties:
                    if mol.HasProp(name):
                        props[name] = mol.GetProp(name)
            yield mol, props


class SMIRepository:
    """Repository for SMILES file format.

    Handles reading and writing SMILES strings with associated IDs.

    Example:
        >>> repo = SMIRepository()
        >>> smiles_data = list(repo.read("input.smi"))  # [(smiles, id), ...]
        >>> repo.write_smiles("output.smi", smiles_data)
    """

    def read(self, path: str) -> Iterator[Chem.Mol]:
        """Read molecules from a SMILES file.

        Args:
            path: Path to the SMILES file.

        Yields:
            RDKit Mol objects with _Name property set.
        """
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles, name = parts[0], parts[1]
                elif len(parts) == 1:
                    smiles, name = parts[0], ""
                else:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol.SetProp("_Name", name)
                    yield mol

    def write(self, path: str, molecules: list[Chem.Mol]) -> None:
        """Write molecules to a SMILES file.

        Args:
            path: Path to the output SMILES file.
            molecules: List of RDKit Mol objects to write.
        """
        with open(path, "w") as f:
            for mol in molecules:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    f.write(f"{smiles}\t{name}\n")

    def read_raw(self, path: str) -> Iterator[tuple[str, str]]:
        """Read raw SMILES strings with IDs.

        Args:
            path: Path to the SMILES file.

        Yields:
            Tuples of (smiles, id).
        """
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    yield parts[0], parts[1]
                elif len(parts) == 1:
                    yield parts[0], ""

    def write_raw(self, path: str, data: list[tuple[str, str]]) -> None:
        """Write raw SMILES strings with IDs.

        Args:
            path: Path to the output file.
            data: List of (smiles, id) tuples.
        """
        with open(path, "w") as f:
            for smiles, name in data:
                f.write(f"{smiles}\t{name}\n")


def read_molecules(path: str, remove_hs: bool = False) -> Iterator[Chem.Mol]:
    """Convenience function to read molecules from any supported format.

    Args:
        path: Path to the input file (SDF or SMI).
        remove_hs: Whether to remove hydrogens (SDF only).

    Yields:
        RDKit Mol objects.

    Raises:
        ValueError: If file format is not supported.
    """
    ext = path.lower().rsplit(".", 1)[-1]

    if ext == "sdf":
        repo = SDFRepository(remove_hs=remove_hs)
    elif ext in ("smi", "smiles"):
        repo = SMIRepository()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    yield from repo.read(path)


def write_molecules(path: str, molecules: list[Chem.Mol]) -> None:
    """Convenience function to write molecules to any supported format.

    Args:
        path: Path to the output file (SDF or SMI).
        molecules: List of RDKit Mol objects.

    Raises:
        ValueError: If file format is not supported.
    """
    ext = path.lower().rsplit(".", 1)[-1]

    if ext == "sdf":
        repo = SDFRepository()
    elif ext in ("smi", "smiles"):
        repo = SMIRepository()
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    repo.write(path, molecules)
