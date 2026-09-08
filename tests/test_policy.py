# tests/test_policy.py
"""``Auto3D.engines.models.policy``: the open-shell gate and the CLI import cost.

Two properties pinned here, both from issue #10/#14 in the multi-review audit:

* ``_requires_aimnet`` / ``check_engine_supports_molecules`` used to test
  elements and net charge only. A radical (``[CH3]``, benzyl, ...) requested
  through ANI2x/ANI2xt passed silently and was scored as a different,
  closed-shell species -- wrong energy AND wrong geometry, with nothing in the
  output to notice by. The fix routes radicals through the SAME hard-error
  mechanism charged species already trigger, and adds a warning (once per
  call, not once per molecule) on the AIMNET/custom-NNP path, which has no
  element/charge restriction but is still closed-shell by default.
* ``policy.py`` sits on the CLI's ``--help`` path
  (``presentation.cli.app`` -> ``commands/properties.py`` -> here), so ``torch``
  and ``rdkit`` must not load merely by importing this module -- see
  ``tests/test_import_boundaries.py::test_cli_app_import_does_not_load_torch_or_rdkit``
  for the module-level assertion; the mol-shaped tests below only need rdkit
  because THEY construct molecules, not because importing the module cost it.

Nothing here loads torch models or an NNP: `_requires_aimnet` and
`check_engine_supports_molecules` are pure RDKit + policy logic.
"""

from __future__ import annotations

import logging

import pytest
from rdkit import Chem

from Auto3D.engines.models.policy import _requires_aimnet, check_engine_supports_molecules
from Auto3D.foundation.exceptions import ConfigurationError


def _mol(smiles: str, name: str | None = None) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test premise: {smiles!r} must parse"
    if name is not None:
        mol.SetProp("_Name", name)
    return mol


# Methyl radical: a single unpaired electron, net charge 0, every atom inside
# the ANI element set -- so if this molecule requires AIMNET, the ONLY
# possible reason is the radical-electron gate, not charge or element.
METHYL_RADICAL = "[CH3]"
BENZYL_RADICAL = "c1ccccc1[CH2]"


class TestRequiresAimnetCountsRadicalElectrons:
    def test_a_radical_with_in_set_elements_and_no_charge_requires_aimnet(self):
        mol = _mol(METHYL_RADICAL)
        assert Chem.rdmolops.GetFormalCharge(mol) == 0, "test premise: charge must be zero"
        assert _requires_aimnet(mol) is True

    def test_a_closed_shell_in_set_neutral_molecule_does_not_require_aimnet(self):
        assert _requires_aimnet(_mol("CCO")) is False

    def test_charge_alone_still_requires_aimnet(self):
        """The pre-existing gate (C11) must survive the radical addition."""
        assert _requires_aimnet(_mol("[O-]C(=O)C")) is True

    def test_out_of_set_element_alone_still_requires_aimnet(self):
        """Same: phosphorus is outside ANI_ELEMENTS regardless of spin state."""
        assert _requires_aimnet(_mol("CP(C)C")) is True


class TestCheckEngineSupportsMoleculesRadicalGate:
    """The three cases issue #10 names explicitly."""

    @pytest.mark.parametrize("engine", ["ANI2x", "ANI2xt", "ani2x", "ANI2XT"])
    def test_radical_requested_for_ani_raises_naming_radicals(self, engine):
        """Same hard error charged species already get (issue #10), and the
        message now names radicals as a possible cause -- not just the
        molecule list charged species already produced."""
        mol = _mol(METHYL_RADICAL, name="methyl_radical")

        with pytest.raises(ConfigurationError) as exc_info:
            check_engine_supports_molecules(mol, engine)

        message = str(exc_info.value)
        assert "methyl_radical" in message
        assert "radical" in message.lower()

    def test_radical_requested_for_aimnet_warns_and_does_not_raise(self, caplog):
        mol = _mol(METHYL_RADICAL, name="methyl_radical")

        with caplog.at_level(logging.WARNING):
            check_engine_supports_molecules(mol, "AIMNET")  # must not raise

        assert "open-shell" in caplog.text.lower()

    def test_closed_shell_molecule_is_unaffected_on_either_path(self, caplog):
        mol = _mol("CCO", name="ethanol")

        with caplog.at_level(logging.WARNING):
            check_engine_supports_molecules(mol, "ANI2x")  # must not raise
            check_engine_supports_molecules(mol, "AIMNET")  # must not raise

        assert caplog.text == ""


class TestOpenShellWarningIsBatchedOncePerCall:
    """One warning line per ``check_engine_supports_molecules`` call, not one
    per molecule -- a run over a large SDF must not turn into a wall of
    near-identical log lines (issue #10)."""

    def test_multiple_radicals_in_one_call_produce_exactly_one_warning(self, caplog):
        mols = [
            _mol(METHYL_RADICAL, name="a"),
            _mol(BENZYL_RADICAL, name="b"),
            _mol("CCO", name="c"),  # closed-shell, must not add a second warning
        ]

        with caplog.at_level(logging.WARNING):
            check_engine_supports_molecules(mols, "AIMNET")

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_custom_nnp_path_also_warns_like_aimnet(self, caplog):
        """Any name outside BUILTIN_ANI_MODELS is the same no-restriction,
        closed-shell-by-default path -- a custom NNP path string included."""
        mol = _mol(METHYL_RADICAL, name="methyl_radical")

        with caplog.at_level(logging.WARNING):
            check_engine_supports_molecules(mol, "/path/to/my_model.pt")

        assert "open-shell" in caplog.text.lower()
