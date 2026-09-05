"""Tests for Auto3D.foundation.utils.output_names.default_output_path.

The single owner of the "<input_stem>_<engine-or-userNNP>_<tag>.sdf"
convention shared by calc_spe, opt_geometry and calc_thermo (issue 17).
"""

from __future__ import annotations

from Auto3D.foundation.utils.output_names import default_output_path


def test_registry_name_branch_names_the_engine():
    """A model name that is not a real path on disk names the engine itself."""
    out = default_output_path("/data/mols.sdf", "AIMNET", "opt")
    assert out == "/data/mols_AIMNET_opt.sdf"


def test_userNNP_branch_for_a_real_model_file(tmp_path):
    """A model_name that exists on disk (a custom NNP) is tagged 'userNNP',
    never the literal (meaningless, filesystem-specific) path."""
    model_file = tmp_path / "my_custom_model.pt"
    model_file.write_bytes(b"not a real checkpoint, just needs to exist")

    out = default_output_path(str(tmp_path / "mols.sdf"), str(model_file), "G")
    assert out == str(tmp_path / "mols_userNNP_G.sdf")


def test_tag_is_used_verbatim():
    """Each caller's own suffix (E / opt / G) passes straight through."""
    assert default_output_path("mols.sdf", "ANI2x", "E").endswith("_ANI2x_E.sdf")
    assert default_output_path("mols.sdf", "ANI2x", "opt").endswith("_ANI2x_opt.sdf")


def test_dotted_stem_is_preserved():
    """splitext, not a '.' split: 'batch.v2.sdf' keeps 'batch.v2' rather than
    collapsing to 'batch' and risking a collision with an unrelated input."""
    out = default_output_path("batch.v2.sdf", "AIMNET", "opt")
    assert out == "batch.v2_AIMNET_opt.sdf"


def test_relative_input_writes_beside_the_input_not_the_cwd():
    """No directory component in the input -> no directory component in the
    output (os.path.join with an empty dirname), matching the pre-refactor
    behavior at all three call sites."""
    out = default_output_path("mols.sdf", "AIMNET", "opt")
    assert out == "mols_AIMNET_opt.sdf"


def test_nested_directory_is_preserved():
    out = default_output_path("/a/b/c/mols.sdf", "AIMNET", "opt")
    assert out == "/a/b/c/mols_AIMNET_opt.sdf"
