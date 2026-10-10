"""The adapter module is a package; its old import path still serves every public name."""

from __future__ import annotations

import importlib
import pathlib

import pytest

PUBLIC = [
    ("BaseModelAdapter", "Auto3D.engines.models.adapter.base"),
    ("AIMNet2Adapter", "Auto3D.engines.models.adapter.aimnet2"),
    ("ANI2xtAdapter", "Auto3D.engines.models.adapter.ani"),
    ("ANI2xAdapter", "Auto3D.engines.models.adapter.ani"),
    ("CustomModelAdapter", "Auto3D.engines.models.adapter.custom"),
    ("validate_energies", "Auto3D.engines.models.adapter.validation"),
    ("_validate_outputs", "Auto3D.engines.models.adapter.validation"),
    ("_try_compile", "Auto3D.engines.models.adapter.compiling"),
]


def test_adapter_is_a_package_not_a_module():
    import Auto3D.engines.models.adapter as adapter

    assert pathlib.Path(adapter.__file__).name == "__init__.py"


@pytest.mark.parametrize(("name", "home"), PUBLIC)
def test_old_path_re_exports_the_defining_object(name, home):
    import Auto3D.engines.models.adapter as adapter

    assert getattr(adapter, name) is getattr(importlib.import_module(home), name)


def test_the_package_exports_exactly_its_public_names():
    import Auto3D.engines.models.adapter as adapter

    assert set(adapter.__all__) == {name for name, _ in PUBLIC if not name.startswith("_")}
