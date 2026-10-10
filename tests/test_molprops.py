#!/usr/bin/env python
"""Tests for Auto3D.foundation.utils.molprops module."""

from __future__ import annotations

import pytest
from rdkit import Chem

from Auto3D.foundation.utils.molprops import has_dummy_atoms


class TestHasDummyAtoms:
    """Truth table for the dummy-atom predicate, pinned at its defining surface.

    ``has_dummy_atoms`` is also pinned through the ``policy.py`` delegate in
    ``tests/test_policy.py``; this is the direct-import case for the function
    itself.
    """

    @pytest.mark.parametrize(
        ("smiles", "expected"),
        [
            ("*CCO", True),
            ("[3*]CCO", True),
            ("CCO", False),  # ethanol - no dummy atom
        ],
    )
    def test_has_dummy_atoms(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert has_dummy_atoms(mol) is expected
