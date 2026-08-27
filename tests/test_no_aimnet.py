"""The no-aimnet path: Auto3D without the ``aimnet`` package installed.

The conda-forge package ships without aimnet (its dependency
nvalchemi-toolkit-ops is pip-only, so aimnet cannot be packaged there).
Every aimnet touch point must therefore fail as an actionable
``DependencyError`` (CLI exit 3, "pip install aimnet") rather than a raw
``ModuleNotFoundError`` -- and everything that does not need aimnet
(``import Auto3D``, ANI engines, custom NNP paths) must keep working.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from Auto3D.foundation.exceptions import DependencyError
from tests.helpers_no_aimnet import BrokenTransitiveDepFinder, hide_aimnet

AIMNET_INSTALLED = importlib.util.find_spec("aimnet") is not None


class TestRequireAimnet:
    @pytest.mark.skipif(not AIMNET_INSTALLED, reason="needs aimnet present")
    def test_noop_when_aimnet_installed(self):
        from Auto3D.engines.models.availability import require_aimnet

        require_aimnet()  # must not raise

    def test_raises_dependency_error_when_missing(self, monkeypatch):
        from Auto3D.engines.models.availability import require_aimnet

        hide_aimnet(monkeypatch)

        with pytest.raises(DependencyError) as exc_info:
            require_aimnet()

        assert exc_info.value.dependency_name == "aimnet"
        assert "pip install aimnet" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)

    def test_broken_transitive_dep_reraises_unchanged(self, monkeypatch):
        """aimnet installed but its own dependency broken: NOT our error to
        relabel -- "install aimnet" would be wrong advice."""
        from Auto3D.engines.models.availability import require_aimnet

        for name in [m for m in sys.modules if m == "aimnet" or m.startswith("aimnet.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [BrokenTransitiveDepFinder()] + sys.meta_path)

        with pytest.raises(ModuleNotFoundError) as exc_info:
            require_aimnet()

        assert exc_info.value.name == "warp"
        assert not isinstance(exc_info.value, DependencyError)
