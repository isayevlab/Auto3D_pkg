"""Stereochemical identity must survive the pipeline.

Auto3D's primary value proposition is stereoisomer enumeration, so a molecule
emitted with a different configuration than it was given is a correctness
failure, not a quality one. Every test here is hermetic: these defects occur
during enumeration, before any neural network potential runs.

Findings: C1 (E/Z collapse), C2 (tautomer stereo loss), M19 (SDF path).
"""

from __future__ import annotations

import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from Auto3D.foundation.utils.stereochemistry import enantiomer, enantiomer_helper


def _enumerate(smiles: str) -> list[str]:
    """Enumerate unassigned stereocenters the way the pipeline does."""
    opts = StereoEnumerationOptions(unique=True, maxIsomers=64, onlyUnassigned=True)
    mol = Chem.MolFromSmiles(smiles)
    return sorted(Chem.MolToSmiles(m) for m in EnumerateStereoisomers(mol, options=opts))


def _embedded(smiles: str, seed: int = 1) -> Chem.Mol:
    """An H-explicit, embedded molecule -- the form the duplicate filters see."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, f"embedding failed: {smiles}"
    return mol


def _pnictogen(mol: Chem.Mol) -> Chem.Atom:
    """The molecule's single phosphorus or arsenic atom."""
    found = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in (15, 33)]
    assert len(found) == 1, f"expected exactly one P or As: {len(found)}"
    return found[0]


def _assert_recovered_tag_matches_parsed(epimers: tuple[str, ...]) -> None:
    """The helper must recover exactly the tag RDKit parsed from the SMILES.

    Shared by the phosphorus and arsenic agreement tests: the geometric argument
    is the same for both elements, so the check is too, and duplicating it per
    element would let the two drift apart.
    """
    from Auto3D.foundation.utils.stereo_check import _assign_pnictogen_tags

    for smiles in epimers:
        parsed = _pnictogen(Chem.MolFromSmiles(smiles)).GetChiralTag()
        assert parsed != Chem.ChiralType.CHI_UNSPECIFIED, f"no parsed tag: {smiles}"

        for seed in range(1, 6):
            mol = _embedded(smiles, seed=seed)
            # Production reads tags after this call, which wipes the tag RDKit
            # itself cannot re-derive -- exactly the gap being filled.
            Chem.AssignStereochemistryFrom3D(mol)
            assert _pnictogen(mol).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

            _assign_pnictogen_tags(mol)
            assert _pnictogen(mol).GetChiralTag() == parsed, (
                f"{smiles} seed {seed}: recovered {_pnictogen(mol).GetChiralTag()}, parsed {parsed}"
            )


class TestEnantiomerPredicate:
    """The enantiomer predicate must not treat 'no stereocenters' as 'enantiomers'."""

    def test_two_achiral_molecules_are_not_enantiomers(self):
        """Molecules with no stereocenters cannot be an enantiomeric pair."""
        assert enantiomer([], []) is False


class TestEZIsomersSurvive:
    """E/Z configuration is invariant under reflection, so it is never enantiomeric."""

    def test_but_2_ene_keeps_both_geometric_isomers(self):
        """CC=CC must yield both E and Z after enantiomer filtering."""
        enumerated = _enumerate("CC=CC")
        assert enumerated == ["C/C=C/C", "C/C=C\\C"], f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"

    def test_fumaric_and_maleic_acid_both_survive(self):
        """The two diacids are distinct compounds, not an enantiomeric pair."""
        enumerated = _enumerate("OC(=O)C=CC(=O)O")
        assert len(enumerated) == 2, f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"


class TestTautomerStereoPreservation:
    """Tautomer enumeration must not silently erase a specified stereocenter."""

    def test_specified_center_survives_tautomer_enumeration(self, job_dir):
        """At least one output tautomer must retain the input's specified center.

        This drives Auto3D's real ``rdkit`` tautomer engine -- the same
        ``RDKitOrOEChemTautomerEngine.rd_taut()`` the pipeline dispatches to,
        reached via
        ``Auto3D.engines.isomers.factory.create_tautomer_engine`` -- rather than a
        bare RDKit ``TautomerEnumerator``, so the defect is attributed to
        Auto3D's tautomer path and not to RDKit in isolation.

        This center is itself alpha to the ketone, so a tautomer genuinely
        produced by enolizing through the stereocenter's own alpha-hydrogen
        could legitimately racemize it -- that would be real chemistry, not a
        bug. The defect under test is that RDKit's ``SetRemoveSp3Stereo(True)``
        strips the center indiscriminately from EVERY output tautomer,
        including ones produced by enolizing the ketone's other,
        non-stereogenic alpha carbon, which cannot touch this center at all.
        That is unconditional information loss, not equilibrium modeling, and
        the eventual fix must not over-correct to "preserve stereo across all
        tautomers unconditionally."
        """
        from Auto3D.engines.isomers.factory import create_tautomer_engine

        in_smi = job_dir / "taut_stereo.smi"
        in_smi.write_text("C[C@H](C(=O)C)N taut_test\n")
        out_smi = job_dir / "taut_stereo_out.smi"

        create_tautomer_engine("rdkit", str(in_smi), str(out_smi), pka_norm=False).run()

        outputs = out_smi.read_text().splitlines()
        assert outputs, "tautomer enumeration returned nothing"
        assert any("@" in line for line in outputs), (
            f"every tautomer lost the specified stereocenter: {sorted(outputs)}"
        )


class TestSdfInputStereo:
    """A 2D SDF with an unspecified center must not be silently randomized."""

    def test_unspecified_center_is_enumerated_or_refused(self, job_dir):
        """Drive Auto3D's real SDF isomer engine on a flat, unspecified center.

        This writes a genuine flat (2D, no wedge bonds, no parity flags) SDF
        record for alanine to disk and feeds it through the production
        ``rdkit_sdf`` engine -- the same ``RDKitSdfIsomer.run()`` the pipeline
        dispatches to for SDF input --
        via ``Auto3D.engines.isomers.IsomerEngineFactory.create``. It then inspects
        the SDF file Auto3D actually writes, grouped by species name (the
        conformer-index suffix stripped). Either the two configurations must
        come out as distinct, internally consistent species, or ambiguous
        input must be explicitly refused (a ``ValueError``). Instead, both
        configurations are written as numbered conformers under one species
        name -- the defect this test targets.
        """
        from Auto3D.engines.isomers import IsomerEngineFactory

        # Alanine drawn flat (2D), with no stereo specified anywhere: no
        # wedge/hash bonds, no parity flags in the mol block.
        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        mol.SetProp("_Name", "alanine_flat")
        AllChem.Compute2DCoords(mol)

        input_sdf = job_dir / "alanine_flat.sdf"
        with Chem.SDWriter(str(input_sdf)) as writer:
            writer.write(mol)

        output_sdf = job_dir / "alanine_enumerated.sdf"
        engine = IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=12,
            threshold=0.3,
            n_jobs=1,
        )

        try:
            engine.run()
        except ValueError:
            # Explicit refusal of ambiguous stereochemistry is an acceptable
            # resolution; there is nothing further to check.
            return

        per_species: dict[str, set[str]] = {}
        for out_mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
            if out_mol is None:
                continue
            name = out_mol.GetProp("_Name")
            species = name.rsplit("_", 1)[0]
            Chem.AssignStereochemistryFrom3D(out_mol)
            found = Chem.FindMolChiralCenters(out_mol, useLegacyImplementation=False)
            per_species.setdefault(species, set()).update(code for _, code in found)

        mixed = {name: sorted(codes) for name, codes in per_species.items() if len(codes) > 1}
        assert not mixed, (
            f"RDKitSdfIsomer wrote a stereochemical mixture under a single species "
            f"name: {mixed}; the pipeline must enumerate distinct species or refuse "
            f"ambiguous input instead of silently mixing configurations as "
            f"conformers of one species"
        )

    def test_two_stereocenters_one_unspecified_yields_two_distinct_species(self, job_dir):
        """A second fixture for the same defect that alanine cannot exercise.

        Alanine has exactly one stereocenter, so ``RDKitSdfIsomer.stereoisomers``'s
        own enantiomer-pair removal collapses its two configurations down to
        ONE before the naming defect above ever has a second configuration to
        mix -- ``assert not mixed`` in the test above is structurally
        unfalsifiable for a single-stereocenter input: it would pass even if
        stereoisomer enumeration were disabled entirely (a mutation that
        truncates ``stereoisomers()`` to its first element is byte-identical
        for alanine). This fixture specifies one stereocenter and leaves a
        second one unspecified, so the two surviving configurations are
        diastereomers, not mirror images of each other, and enantiomer dedup
        keeps both -- giving the "mixed under one name" check something to
        actually falsify.

        Tracking is also different from the test above on purpose: pooling
        R/S codes across a molecule's own multiple stereocenters into one set
        (as the alanine test does) reads a single genuine two-center
        configuration as "mixed" against itself the moment its two centers
        happen to carry different labels. Track the full (atom_idx, code)
        configuration per conformer instead.
        """
        from Auto3D.engines.isomers import IsomerEngineFactory

        # C2 (attached to OH) is specified via @; C3 (attached to NH2) is left
        # unspecified. EnumerateStereoisomers(onlyUnassigned=True) then varies
        # only C3, producing two diastereomers (not enantiomers of each other,
        # since C2 is fixed), so enantiomer_key dedup does not collapse them.
        mol = Chem.MolFromSmiles("C[C@H](O)C(N)C(=O)O")
        mol.SetProp("_Name", "aminobutanol_flat")
        AllChem.Compute2DCoords(mol)

        input_sdf = job_dir / "aminobutanol_flat.sdf"
        with Chem.SDWriter(str(input_sdf)) as writer:
            writer.write(mol)

        output_sdf = job_dir / "aminobutanol_enumerated.sdf"
        engine = IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=12,
            threshold=0.3,
            n_jobs=1,
        )

        try:
            engine.run()
        except ValueError:
            # Explicit refusal of ambiguous stereochemistry is an acceptable
            # resolution; there is nothing further to check.
            return

        per_species: dict[str, set[frozenset]] = {}
        for out_mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
            if out_mol is None:
                continue
            name = out_mol.GetProp("_Name")
            species = name.rsplit("_", 1)[0]
            Chem.AssignStereochemistryFrom3D(out_mol)
            found = frozenset(Chem.FindMolChiralCenters(out_mol, useLegacyImplementation=False))
            per_species.setdefault(species, set()).add(found)

        # Unlike the alanine case, both diastereomers must actually survive --
        # if stereoisomer enumeration silently truncated to one (the mutation
        # the alanine test above cannot catch), there would be nothing left
        # to check for mixing either.
        assert len(per_species) == 2, (
            f"expected the two diastereomers to survive as two distinct "
            f"species, got {sorted(per_species)}"
        )

        mixed = {name: sorted(configs) for name, configs in per_species.items() if len(configs) > 1}
        assert not mixed, (
            f"RDKitSdfIsomer wrote more than one stereochemical configuration "
            f"under a single species name: {mixed}"
        )


class TestChiralPhosphorusIdentity:
    """P(III) epimers are distinct compounds and must not share a species key.

    A pyramidal trivalent phosphine is a configurationally stable stereocenter
    -- inversion costs tens of kcal/mol for an ordinary tertiary phosphine,
    against roughly 6 for the analogous amine -- so its two epimers are
    separable compounds, not conformers. The barrier is substituent-dependent,
    and an aromatic phosphorus is not stable at all, which is why that case is
    excluded rather than tagged. RDKit perceives none of this: ``AssignStereochemistryFrom3D``
    leaves degree-3 P untagged under both perception modes, and
    ``AssignAtomChiralTagsFromStructure`` leaves it untagged too, so there is
    no label for ``rdCIPLabeler`` to compute either. With no tag, both epimers
    canonicalize to the same SMILES, the duplicate filters read them as
    conformers of one species, and whichever has the higher energy is dropped
    with nothing logged (finding N-M10).
    """

    def test_phosphorus_epimers_have_distinct_species_keys(self, stereo_perception):
        from Auto3D.foundation.utils.stereo_check import species_key

        a = _embedded("CC[P@@](C)CCC")
        b = _embedded("CC[P@](C)CCC")
        assert species_key(a) != species_key(b), (
            "the two phosphine epimers share a species key, so a duplicate "
            "filter will drop one of two distinct compounds"
        )

    def test_phosphine_epimer_keys_are_stable_across_embeddings(self, stereo_perception):
        """One epimer must key identically from two independent conformers.

        The key is the filters' answer to "same compound?", so it has to be a
        property of the configuration and not of the conformer that happened to
        be embedded. A sign convention read off the wrong reference -- neighbor
        index order rather than bond order, say -- can still separate the two
        epimers above while varying between conformers of one of them, which
        would split a species into singletons instead of merging two.
        """
        from Auto3D.foundation.utils.stereo_check import species_key

        assert species_key(_embedded("CC[P@@](C)CCC", seed=1)) == species_key(
            _embedded("CC[P@@](C)CCC", seed=7)
        )

    def test_non_stereogenic_phosphine_gets_no_tag(self, stereo_perception):
        """A phosphorus with two identical substituents is not a stereocenter.

        ``CP(C)CC`` has two methyls, so its signed volume still has a sign and a
        tag assigned from that sign alone would flip between conformers --
        splitting one compound across two keys, the mirror of the defect above.
        Symmetry-aware canonical ranks (``breakTies=False``) are what rule it
        out: two neighbors sharing a rank means there is nothing to resolve.
        """
        from Auto3D.foundation.utils.stereo_check import species_key

        first, second = _embedded("CP(C)CC", seed=1), _embedded("CP(C)CC", seed=7)
        assert species_key(first) == species_key(second), (
            "a non-stereogenic phosphorus was given a conformer-dependent tag"
        )
        assert "@" not in species_key(first), (
            f"a non-stereogenic phosphorus was tagged: {species_key(first)}"
        )

    def test_phosphine_tag_matches_the_parsed_smiles_tag(self, stereo_perception):
        """Pin the sign convention against RDKit's own reading of ``[P@]``/``[P@@]``.

        ``CHI_TETRAHEDRAL_CW``/``CCW`` are defined relative to the atom's bond
        ordering, with RDKit's own rule for where the implicit fourth position
        (here the lone pair) sits, so which sign of the triple product means CW
        is not derivable from the enum names. It was determined empirically and
        is pinned here: RDKit parses the tag from the SMILES, ETKDG honors it
        when it embeds, and the helper must recover exactly that tag from the
        resulting coordinates. If a future RDKit flips the convention this test
        fails rather than the species keys quietly swapping identities.
        """
        _assert_recovered_tag_matches_parsed(("CC[P@@](C)CCC", "CC[P@](C)CCC"))

    def test_phosphine_oxide_epimers_have_distinct_species_keys(self, stereo_perception):
        """The P(V) path must keep working, and must not go through the helper.

        A four-coordinate phosphorus is an ordinary tetrahedral center that
        ``AssignStereochemistryFrom3D`` tags on its own, so the helper has to
        leave it alone -- a degree filter that caught it would overwrite a
        correct tag with one derived from only three of its four substituents.
        """
        from Auto3D.foundation.utils.stereo_check import species_key

        a = _embedded("CC[P@@](=O)(C)CCC")
        b = _embedded("CC[P@](=O)(C)CCC")
        assert species_key(a) != species_key(b)

    def test_phosphine_key_is_invariant_to_atom_order(self, stereo_perception):
        """The key is a property of the configuration, not of the atom numbering.

        ``species_key`` is compared across separately parsed molecules, so the
        sign convention has to be read off the reference RDKit's own CW/CCW are
        defined against -- **bond** order. A neighbor-index-order reading is not
        caught by any other test here: the fixture molecules' P has bond-order
        neighbors ``[1, 3, 4]``, which is already sorted, so the two readings
        are indistinguishable on them. Renumbering is what separates them -- an
        index-order implementation changes the key on 4 of these 9 orderings,
        and on the reversed one recovers the opposite tag from the parsed SMILES.
        """
        from Auto3D.foundation.utils.stereo_check import species_key

        mol = _embedded("CC[P@@](C)CCC")
        count = mol.GetNumAtoms()
        expected = species_key(mol)

        orderings = {
            "reversed": list(reversed(range(count))),
            "rotated_by_one": [*range(1, count), 0],
            "rotated_by_three": [*range(3, count), 0, 1, 2],
        }
        shuffler = random.Random(0)
        for index in range(6):
            permutation = list(range(count))
            shuffler.shuffle(permutation)
            orderings[f"shuffled_{index}"] = permutation

        for name, permutation in orderings.items():
            renumbered = Chem.RenumberAtoms(mol, permutation)
            assert species_key(renumbered) == expected, (
                f"the {name} renumbering changed the phosphine species key, so one "
                f"compound will split across two keys depending on atom numbering"
            )

    def test_aromatic_phosphorus_gets_no_tag_and_a_stable_key(self, stereo_perception):
        """A phosphole's phosphorus is planar, so its signed volume is noise.

        Trivalent P is a stereocenter because its lone pair holds the fourth
        vertex of a pyramid. In a conjugated five-membered ring the lone pair
        joins the aromatic system instead: RDKit perceives the atom as aromatic
        and sp2, the three bonds are very nearly coplanar, and the sign of the
        triple product is then a property of the conformer rather than of the
        compound -- measured over 25 ETKDG seeds of 1,2-dimethylphosphole the
        volume ranges from -1.96 to +2.23 and the sign changes 11 times, which
        splits one compound across two species keys.

        The chemistry agrees with excluding it: a 1-substituted phosphole
        inverts with a barrier near 16 kcal/mol, because aromatic stabilization
        of the planar transition state is exactly what flattens the center, so
        it is not configurationally stable on any relevant timescale.

        Seeds 1 and 2 are chosen because they are the shortest pair whose
        volumes land on opposite sides of zero (-0.94 and +1.25).
        """
        from Auto3D.foundation.utils.stereo_check import (
            _assign_pnictogen_tags,
            species_key,
        )

        first, second = _embedded("Cc1cccp1C", seed=1), _embedded("Cc1cccp1C", seed=2)
        assert _pnictogen(first).GetIsAromatic(), "fixture is not an aromatic phosphorus"

        assert species_key(first) == species_key(second), (
            "an aromatic phosphorus was given a conformer-dependent tag, so one "
            "compound splits across two species keys"
        )
        for mol in (first, second):
            probe = Chem.Mol(mol)
            Chem.AssignStereochemistryFrom3D(probe)
            _assign_pnictogen_tags(probe)
            assert _pnictogen(probe).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED, (
                f"aromatic phosphorus was tagged {_pnictogen(probe).GetChiralTag()}"
            )

    def test_charged_phosphorus_gets_no_tag(self, stereo_perception):
        """The lone-pair-as-fourth-vertex model does not apply to an anion.

        ``CC[P-](C)CCC`` has three bonds plus a lone pair plus an extra
        non-bonding electron (RDKit reads it as one radical electron), so the
        geometry the sign convention assumes is not the geometry present. Guard
        on charge and radical count so the code and the stated model agree.
        """
        from Auto3D.foundation.utils.stereo_check import _assign_pnictogen_tags

        probe = Chem.Mol(_embedded("CC[P-](C)CCC"))
        Chem.AssignStereochemistryFrom3D(probe)
        _assign_pnictogen_tags(probe)
        assert _pnictogen(probe).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED

    def test_arsine_epimers_have_distinct_species_keys(self, stereo_perception):
        """N-M10 is the same defect for arsenic, and the pipeline reaches it.

        ``FindPotentialStereo`` counts a trivalent As center, so
        ``count_unspecified_stereo`` warns about it, and
        ``EnumerateStereoisomers`` builds both epimers of ``CC[As](C)CCC``.
        They then meet in one duplicate-filter group, because
        ``ranking.species_id`` strips both the isomer and the conformer index.
        Without a tag they share a key and the higher-energy epimer is dropped
        with nothing logged -- and arsine inversion barriers are *higher* than
        phosphine's (~40 kcal/mol), so the two are certainly separable compounds.
        """
        from Auto3D.foundation.utils.stereo_check import species_key

        a = _embedded("CC[As@@](C)CCC")
        b = _embedded("CC[As@](C)CCC")
        assert species_key(a) != species_key(b), (
            "the two arsine epimers share a species key, so a duplicate filter "
            "will drop one of two distinct compounds"
        )

    def test_arsine_tag_matches_the_parsed_smiles_tag(self, stereo_perception):
        """Arsenic must use the same sign convention, pinned the same way.

        Nothing guarantees a priori that RDKit's CW/CCW reference for a
        three-coordinate As matches the one measured for P, so it is checked
        rather than assumed. (It does: the volumes are negative for ``[As@@]``
        and positive for ``[As@]`` across seeds 1-5, exactly as for P.)
        """
        _assert_recovered_tag_matches_parsed(("CC[As@@](C)CCC", "CC[As@](C)CCC"))
