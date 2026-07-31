"""Tautomer enumeration preserves specified stereo without inventing any.

The C2 fix disables RDKit's default stereo stripping. That is only correct if
it preserves descriptors the user specified while still dropping descriptors
the tautomerization genuinely destroys, and while assigning none that the
input never had. These tests pin all three, driving Auto3D's real ``rdkit``
tautomer engine through the production factory.
"""
from __future__ import annotations

from Auto3D.isomers.factory import create_tautomer_engine


def _run_rd_taut(job_dir, smiles: str) -> list[str]:
    """Drive TautomerEngine.rd_taut() and return the output SMILES."""
    in_smi = job_dir / "taut_in.smi"
    in_smi.write_text(f"{smiles} probe\n")
    out_smi = job_dir / "taut_out.smi"
    create_tautomer_engine(
        "rdkit", str(in_smi), str(out_smi), pka_norm=False
    ).run()
    return [line.split()[0] for line in out_smi.read_text().splitlines() if line.strip()]


class TestSpecifiedStereoSurvives:
    def test_center_remote_from_the_tautomeric_site_is_kept(self, job_dir):
        """A center the tautomerization cannot reach keeps its configuration."""
        outputs = _run_rd_taut(job_dir, "CC(=O)CCCC[C@H](C)O")
        assert outputs, "tautomer enumeration returned nothing"
        assert all("@" in smi for smi in outputs), (
            f"a remote stereocenter was stripped: {sorted(outputs)}"
        )

    def test_specified_double_bond_geometry_is_kept(self, job_dir):
        """A specified C=C keeps its geometry through enumeration."""
        outputs = _run_rd_taut(job_dir, "C/C=C(\\O)C")
        assert outputs, "tautomer enumeration returned nothing"
        assert any("/" in smi or "\\" in smi for smi in outputs), (
            f"every tautomer lost the specified double-bond geometry: {sorted(outputs)}"
        )

    def test_a_cip_relabeled_center_is_not_mistaken_for_an_inversion(self, job_dir):
        """A tautomer that only changes CIP priority must survive.

        A keto/enol shift can flip an untouched center's CIP label from R to S
        by changing a neighboring branch's substituent priority. The physical
        arrangement is identical, so rejecting it would discard a valid
        tautomer -- the filter must key on constitution, not on the label.
        """
        outputs = _run_rd_taut(job_dir, "OC[C@H](CC(=O)C)C=C")
        assert len(outputs) == 3, (
            f"a configuration-preserving tautomer was rejected: {sorted(outputs)}"
        )
        assert all("@" in smi for smi in outputs), sorted(outputs)


class TestNoStereoIsInvented:
    def test_a_center_destroyed_by_tautomerization_is_still_dropped(self, job_dir):
        """Tautomers whose stereocenter carbon became sp2 carry no descriptor.

        This is the over-correction guard: preserving stereo must not mean
        asserting a configuration on an atom that no longer has one.
        """
        outputs = _run_rd_taut(job_dir, "C[C@H](C(=O)C)N")
        assert outputs, "tautomer enumeration returned nothing"
        # The enamine/imine tautomers flatten the stereocenter's own carbon.
        flattened = [smi for smi in outputs if "C=C(N)" in smi or "C(=N)" in smi]
        assert flattened, f"expected a tautomer that flattens the center: {sorted(outputs)}"
        assert all("@" not in smi for smi in flattened), (
            f"a destroyed stereocenter kept a descriptor: {sorted(flattened)}"
        )

    def test_an_achiral_input_gains_no_stereo(self, job_dir):
        """A keto input must not acquire E/Z on the enol it tautomerizes to."""
        outputs = _run_rd_taut(job_dir, "CCC(C)=O")
        assert outputs, "tautomer enumeration returned nothing"
        assert all("@" not in smi for smi in outputs), (
            f"stereo was invented: {sorted(outputs)}"
        )
        assert all("/" not in smi and "\\" not in smi for smi in outputs), (
            f"double-bond geometry was invented: {sorted(outputs)}"
        )

    def test_a_multi_step_tautomerization_does_not_emit_the_enantiomer(self, job_dir):
        """D-erythrose must not come back as L-erythrose.

        The 2,3-enediol flattens both centers, and RDKit restores a definite
        tag rather than leaving them unspecified -- for one output, the input's
        mirror image. Downstream these survive as two tautomer IDs with equal
        energies and k=1 picks between them by file order.
        """
        from rdkit import Chem

        d_erythrose = "O=C[C@H](O)[C@H](O)CO"
        outputs = _run_rd_taut(job_dir, d_erythrose)
        assert outputs, "tautomer enumeration returned nothing"
        canonical = {Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in outputs}

        assert Chem.MolToSmiles(Chem.MolFromSmiles(d_erythrose)) in canonical, (
            f"the input configuration was lost: {sorted(canonical)}"
        )
        l_erythrose = Chem.MolToSmiles(Chem.MolFromSmiles("O=C[C@@H](O)[C@@H](O)CO"))
        assert l_erythrose not in canonical, (
            f"the enantiomer of the input was emitted as a tautomer: {sorted(canonical)}"
        )
