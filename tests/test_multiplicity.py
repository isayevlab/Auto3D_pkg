"""Spin multiplicity resolution, moved to the domain layer.

These cover the pure-Python resolver in ``Auto3D.domain.multiplicity`` (moved
here from ``tests/test_thermo_helpers.py`` when the resolver moved out of the
thermochemistry layer so the optimizer, in the engines layer, could reach it
too) and the identity check that the thermo layer calls this module's
resolver directly and shares one record-naming helper with it.
"""

from __future__ import annotations

import logging

import pytest
from rdkit import Chem

from Auto3D.domain.multiplicity import electron_count, resolve_multiplicity


def test_the_thermo_modules_use_the_domain_resolver_and_the_shared_record_name():
    from Auto3D.domain import multiplicity
    from Auto3D.entry.ASE.thermo import driver, properties
    from Auto3D.foundation.utils import molprops

    assert driver.resolve_multiplicity is multiplicity.resolve_multiplicity
    assert properties.record_name is molprops.record_name
    assert multiplicity.record_name is molprops.record_name


def test_resolve_multiplicity_closed_shell_is_singlet():
    m = Chem.MolFromSmiles("CCO")
    assert resolve_multiplicity(m) == 1


def test_resolve_multiplicity_radical_is_doublet():
    m = Chem.MolFromSmiles("[CH3]")  # methyl radical, 1 unpaired electron
    assert resolve_multiplicity(m) == 2


def test_resolve_multiplicity_open_shell_warning_names_the_molecule(caplog):
    """The open-shell warning is the one sibling in this function that did
    not name the record; every other warning here passes ``record_name(mol)``.
    """
    m = Chem.MolFromSmiles("[CH3]")  # methyl radical, 1 unpaired electron
    m.SetProp("_Name", "my_radical")
    with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
        assert resolve_multiplicity(m) == 2
    assert any("my_radical" in r.message for r in caplog.records), caplog.text


def test_resolve_multiplicity_does_not_write_the_derived_value_back():
    m = Chem.MolFromSmiles("[CH3]")  # methyl radical, as the doublet case above
    assert resolve_multiplicity(m) == 2
    assert not m.HasProp("multiplicity")


def test_resolve_multiplicity_respects_explicit_property():
    m = Chem.MolFromSmiles("[CH3]")  # would derive 2 ...
    m.SetUnsignedProp("multiplicity", 4)  # ... but an explicit value wins
    assert resolve_multiplicity(m) == 4
    # A valid explicit property is the one actually used, so it stays.
    assert m.GetUnsignedProp("multiplicity") == 4


@pytest.mark.parametrize(
    "value",
    ["0", "99", "7", "two"],
    ids=["below_1", "above_electron_bound", "wrong_parity", "unparseable"],
)
def test_resolve_multiplicity_invalid_property_is_dropped_from_the_record(value):
    """Every one of the four rejection branches must drop the invalid
    'multiplicity' property from the record once the radical-derived value
    is used instead: a record carrying a multiplicity that was never fed
    into G would be advertising a number nothing used.

    The methyl radical is a 9-electron (odd) species: the valid range is
    1 to n_electrons + 1 = 10, and parity must be even (2S+1 requires even
    multiplicity for an odd-electron species). The four ``value``s each land
    in one rejection branch: "0" is below 1, "99" is above the 10-electron
    bound, "7" is odd and so fails parity, and "two" is unparseable.
    """
    m = Chem.MolFromSmiles("[CH3]")
    assert electron_count(m) == 9, "test premise: a 9-electron, odd-electron species"
    m.SetProp("multiplicity", value)
    assert resolve_multiplicity(m) == 2
    assert not m.HasProp("multiplicity")


def _mol(smiles, **props):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for key, value in props.items():
        mol.SetProp(key, str(value))
    return mol


class TestMultiplicity:
    def test_a_malformed_property_falls_back_to_the_radical_count(self):
        """The accessor was unguarded where _symmetry_number's is guarded."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "triplet")
        assert resolve_multiplicity(mol) == 1

    def test_negative_multiplicity_falls_back_to_the_radical_count(self, caplog):
        """GetUnsignedProp("-1") wraps around to 4294967295 rather than
        raising, feeding spin = 2147483647.0 into IdealGasThermo's
        R*ln(multiplicity) term. The value must be validated, not just
        parsed."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "-1")
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_zero_multiplicity_falls_back_to_the_radical_count(self, caplog):
        """GetUnsignedProp("0") parses cleanly to 0, giving spin = -0.5."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "0")
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_dioxygen_is_flagged_as_ambiguous(self, caplog):
        """O=O draws closed-shell but is a ground-state triplet.

        The radical-electron count is 0 here, so nothing signals that the
        closed-shell assumption is wrong for this molecule.
        """
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            multiplicity = resolve_multiplicity(_mol("O=O"))
        assert multiplicity == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records), (
            f"an ambiguous open-shell drawing was not flagged: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_ordinary_closed_shell_molecule_is_not_flagged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            resolve_multiplicity(_mol("CCO"))
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_an_unsigned_wraparound_value_falls_back_to_the_radical_count(self, caplog):
        """The other side of the guard: int("4294967295") parses cleanly (no
        wraparound -- that only afflicts GetUnsignedProp) to a value that is
        ">= 1" and so slipped past a lower-bound-only check, feeding
        spin = 2147483647.0 into R*ln(multiplicity) with no warning -- a 13.1
        kcal/mol shift in Gibbs energy at 298.15 K. It must be rejected by the
        upper bound (n_electrons + 1)."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "4294967295")
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_value_just_above_the_electron_bound_falls_back(self, caplog):
        """n_electrons + 1 is the physical ceiling (every electron unpaired);
        one above that is invalid regardless of parity."""
        mol = _mol("CCO")
        n_electrons = electron_count(mol)
        mol.SetProp("multiplicity", str(n_electrons + 3))  # same parity, over the cap
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_wrong_parity_value_falls_back(self, caplog):
        """CCO is a 26-electron (even) closed-shell species, so a valid
        multiplicity must be odd; 2 (even) is unreachable by any spin state
        and must be rejected even though it is within the electron-count
        bound."""
        mol = _mol("CCO", multiplicity=2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_legitimate_high_multiplicity_on_an_even_electron_species_passes(self, caplog):
        """A triplet (multiplicity 3) on an even-electron species is
        physically legitimate (e.g. a diradical or an excited state) and
        must pass through unchanged and unwarned."""
        mol = _mol("CCO", multiplicity=3)
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 3
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_legitimate_doublet_on_a_radical_passes(self, caplog):
        """The methyl radical is a 9-electron (odd) species, so multiplicity
        2 (even) is exactly the physically expected doublet and must pass
        through unchanged and unwarned."""
        mol = _mol("[CH3]", multiplicity=2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            assert resolve_multiplicity(mol) == 2
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)


class TestDerivedMultiplicityIsAlsoChecked:
    """The bounds and parity checks only ever ran on a *supplied* multiplicity.

    Both sit inside ``if mol.HasProp("multiplicity")``, so a multiplicity Auto3D
    derived from the radical-electron count was returned unchecked -- and 2S+1
    requires odd multiplicity for an even-electron species, even for an odd-electron
    one, which the radical count can violate when the drawing hides an open shell.
    Getting it wrong is worth RT ln((m+1)/m) or RT ln(m/(m-1)) in T*S_elec --
    0.41 kcal/mol at 298 K for a doublet taken for a singlet.

    RDKit re-derives radical electrons on sanitization and is self-consistent for
    every ordinary species (checked: CH2, CH3, O2, N, ethanol), which is why this
    needed a deliberately inconsistent molecule rather than a realistic one --
    and why the check is worth having: the case it catches is precisely the one no
    ordinary input produces.
    """

    def test_a_parity_inconsistent_derived_multiplicity_warns(self, caplog):
        # CH3 has 9 electrons (odd), so 2S+1 must be even. Forcing two unpaired
        # electrons makes the derived multiplicity 3 -- odd, and impossible.
        mol = Chem.MolFromSmiles("[CH3]")
        mol.GetAtomWithIdx(0).SetNumRadicalElectrons(2)
        mol.SetProp("_Name", "impossible_radical")
        assert electron_count(mol) % 2 == 1, "test premise: odd electrons"

        with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
            multiplicity = resolve_multiplicity(mol)

        assert multiplicity == 3, "the derived value is still returned, not replaced"
        assert any("parity" in r.message for r in caplog.records), (
            f"an impossible derived multiplicity was returned with no warning: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_a_consistent_derived_multiplicity_is_not_warned_about(self, caplog):
        """Every ordinary radical must stay quiet, or the check is noise."""
        for smiles in ("[CH3]", "[CH2]", "CCO", "[N]"):
            caplog.clear()
            mol = Chem.MolFromSmiles(smiles)
            with caplog.at_level(logging.WARNING, logger="Auto3D.domain.multiplicity"):
                resolve_multiplicity(mol)
            assert not any("parity" in r.message for r in caplog.records), (
                f"{smiles} triggered a parity warning it should not have"
            )
