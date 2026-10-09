"""The parent resolves an aimnet model name and checks a cached model without importing
``aimnet.calculators`` (P-M8): that import loads torch, warp and the CUDA runtime, and
``auto3d run`` paid it once in the parent and again in every worker."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

from Auto3D.engines.models import preflight
from Auto3D.foundation.exceptions import ConfigurationError, ModelLoadError

MODEL_BYTES = b"not a real checkpoint, but the registry only checks its hash\n"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A tiny registry file in aimnet's schema, with the heavy import made impossible."""
    doc = {
        "families": {"test-family": {}},
        "models": {
            "aimnet2-test_0": {
                "family": "test-family",
                "file": "aimnet2_test_0.pt",
                "url": "https://example.invalid/aimnet2_test_0.pt",
                "sha256": hashlib.sha256(MODEL_BYTES).hexdigest(),
            }
        },
        "aliases": {
            "aimnet2": "aimnet2-test_0",
            "aimnet2-test": "aimnet2-test_0",
            # Dangling on purpose: resolves through the alias to a name absent
            # from "models", exercising the one path where _resolve_locally
            # must return None rather than the alias target (see
            # TestResolveWithoutTheHeavyImport.test_a_dangling_alias_is_rejected).
            "aimnet2-orphan": "gone_0",
        },
    }
    path = tmp_path / "model_registry.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(preflight, "_registry_path", lambda: path)
    monkeypatch.setattr(preflight, "require_aimnet", lambda: None)
    # Any attempt to import the heavy package fails loudly instead of silently succeeding.
    monkeypatch.setitem(sys.modules, "aimnet.calculators", None)
    monkeypatch.setitem(sys.modules, "aimnet.calculators.model_registry", None)
    monkeypatch.setitem(sys.modules, "aimnet.models", None)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(cache))
    return {"path": path, "cache": cache, "doc": doc}


class TestResolveWithoutTheHeavyImport:
    def test_the_auto3d_literal_resolves_through_the_alias(self, registry):
        assert preflight.resolve_engine_name("AIMNET") == "aimnet2-test_0"

    def test_an_alias_resolves(self, registry):
        assert preflight.resolve_engine_name("aimnet2-test") == "aimnet2-test_0"

    def test_a_model_name_resolves_to_itself(self, registry):
        assert preflight.resolve_engine_name("aimnet2-test_0") == "aimnet2-test_0"

    def test_a_typo_lists_the_aliases(self, registry):
        with pytest.raises(ConfigurationError) as excinfo:
            preflight.resolve_engine_name("aimnet2-testx")
        message = str(excinfo.value)
        assert "aimnet2-testx" in message and "aimnet2-test" in message

    def test_a_dangling_alias_is_rejected(self, registry):
        """An alias whose target is absent from ``models`` must not pass through.

        ``aimnet2-orphan`` resolves to ``gone_0`` in the fixture's aliases, but
        ``gone_0`` is not a key in ``models`` -- the one case where
        ``_resolve_locally`` deliberately returns None rather than the alias
        target, so the caller raises ``ConfigurationError`` instead of handing
        back a model name nothing can load.
        """
        with pytest.raises(ConfigurationError) as excinfo:
            preflight.resolve_engine_name("aimnet2-orphan")
        assert "aimnet2-orphan" in str(excinfo.value)


class TestPreflightWarmCache:
    def test_a_cached_model_with_the_right_hash_needs_no_aimnet_import(self, registry):
        (registry["cache"] / "aimnet2_test_0.pt").write_bytes(MODEL_BYTES)
        assert preflight.preflight_model("AIMNET") is None

    def test_an_empty_cache_dir_env_var_falls_back_to_aimnet(self, registry, monkeypatch, tmp_path):
        """A set-but-empty AIMNET_CACHE_DIR must not be folded to the home cache.

        ``AIMNET_CACHE_DIR=""`` makes ``_model_cache_dir()`` return ``""``, and
        ``Path("") / file`` names a file in the current directory. Before this
        fix, a correctly hashed file placed there made ``_cached_model_is_valid``
        return True, so pre-flight passed even though aimnet's own
        ``os.makedirs("")`` would fail. ``_cached_model_is_valid`` now treats a
        falsy cache_dir as invalid regardless of what is on disk, so this must
        fall through to the (fake) aimnet import and reach it, not return.
        """
        import types

        monkeypatch.chdir(tmp_path)
        (tmp_path / "aimnet2_test_0.pt").write_bytes(MODEL_BYTES)
        monkeypatch.setenv("AIMNET_CACHE_DIR", "")

        calls = []
        fake = types.ModuleType("aimnet.calculators.model_registry")

        def get_registry_model_path(name):
            calls.append(name)
            raise ConnectionError("Temporary failure in name resolution")

        fake.get_registry_model_path = get_registry_model_path
        monkeypatch.setitem(
            sys.modules, "aimnet.calculators", types.ModuleType("aimnet.calculators")
        )
        monkeypatch.setitem(sys.modules, "aimnet.calculators.model_registry", fake)

        with pytest.raises(ModelLoadError):
            preflight.preflight_model("AIMNET")
        assert calls == ["aimnet2-test_0"], "the fake aimnet module was not reached"

    def test_a_missing_file_falls_back_to_aimnet(self, registry, monkeypatch):
        """Cold cache: the download has to go through aimnet, and here that import is impossible."""
        import types

        calls = []
        fake = types.ModuleType("aimnet.calculators.model_registry")

        def get_registry_model_path(name):
            calls.append(name)
            raise ConnectionError("Temporary failure in name resolution")

        fake.get_registry_model_path = get_registry_model_path
        monkeypatch.setitem(
            sys.modules, "aimnet.calculators", types.ModuleType("aimnet.calculators")
        )
        monkeypatch.setitem(sys.modules, "aimnet.calculators.model_registry", fake)
        with pytest.raises(ModelLoadError) as excinfo:
            preflight.preflight_model("AIMNET")
        assert calls == ["aimnet2-test_0"]
        assert "network" in str(excinfo.value).lower()

    def test_a_corrupt_cached_file_falls_back_to_aimnet_and_names_the_file(
        self, registry, monkeypatch
    ):
        import types

        (registry["cache"] / "aimnet2_test_0.pt").write_bytes(b"truncated")
        fake = types.ModuleType("aimnet.calculators.model_registry")

        def get_registry_model_path(name):
            raise ValueError("Checksum mismatch for aimnet2_test_0.pt: expected ..., got ...")

        fake.get_registry_model_path = get_registry_model_path
        monkeypatch.setitem(
            sys.modules, "aimnet.calculators", types.ModuleType("aimnet.calculators")
        )
        monkeypatch.setitem(sys.modules, "aimnet.calculators.model_registry", fake)
        with pytest.raises(ModelLoadError) as excinfo:
            preflight.preflight_model("AIMNET")
        message = str(excinfo.value).lower()
        assert "checksum" in message and any(
            w in message for w in ("delete", "remove", "aimnet_cache_dir")
        )


def test_a_registry_without_the_expected_keys_falls_back(registry, monkeypatch):
    """A future aimnet that renames the keys must degrade to the slow path, not to a KeyError."""
    import types

    registry["path"].write_text(yaml.safe_dump({"unexpected": {}}))
    fake = types.ModuleType("aimnet.calculators.model_registry")
    fake.load_model_registry = lambda: {
        "aliases": {"aimnet2": "aimnet2-test_0"},
        "models": {"aimnet2-test_0": {}},
    }
    fake.resolve_registry_model_name = lambda n: {"aimnet2": "aimnet2-test_0"}.get(n, n)
    monkeypatch.setitem(sys.modules, "aimnet.calculators", types.ModuleType("aimnet.calculators"))
    monkeypatch.setitem(sys.modules, "aimnet.calculators.model_registry", fake)
    assert preflight.resolve_engine_name("AIMNET") == "aimnet2-test_0"
