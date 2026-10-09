"""Tests for ``Auto3D.domain.embedding.available_cpu_count``.

``os.cpu_count()`` is the machine's count; the affinity mask is what the
scheduler actually gives this process, and is what ``resolve_embedding_workers``
needs (see ``tests/test_parallel_embed.py`` and ``tests/test_isomers.py`` for
the call sites that depend on this helper, not on ``os.cpu_count`` directly).
"""

import os

import pytest

import Auto3D.domain.embedding as embedding_mod


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="os.sched_getaffinity is Linux-only"
)
def test_available_cpu_count_matches_the_affinity_mask_on_linux():
    """On Linux, ``os.sched_getaffinity`` exists and is the answer."""
    assert embedding_mod.available_cpu_count() == len(os.sched_getaffinity(0))


def test_available_cpu_count_is_at_least_one():
    assert embedding_mod.available_cpu_count() >= 1


def test_available_cpu_count_falls_back_to_cpu_count_when_no_affinity_api(monkeypatch):
    """Platforms without ``os.sched_getaffinity`` (e.g. macOS) fall back to
    ``os.cpu_count() or 1``."""
    monkeypatch.delattr(embedding_mod.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(embedding_mod.os, "cpu_count", lambda: 6)
    assert embedding_mod.available_cpu_count() == 6


def test_available_cpu_count_survives_an_unknown_core_count(monkeypatch):
    """``os.cpu_count()`` returns None when the platform cannot say; the
    helper still returns at least 1 rather than propagating None."""
    monkeypatch.delattr(embedding_mod.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(embedding_mod.os, "cpu_count", lambda: None)
    assert embedding_mod.available_cpu_count() == 1


def test_available_cpu_count_survives_an_oserror_from_sched_getaffinity(monkeypatch):
    """A platform that declares the API but cannot answer for this pid (OSError)
    still falls back to ``os.cpu_count() or 1`` instead of raising."""

    def _raise(_pid):
        raise OSError("no such process")

    monkeypatch.setattr(embedding_mod.os, "sched_getaffinity", _raise, raising=False)
    monkeypatch.setattr(embedding_mod.os, "cpu_count", lambda: 4)
    assert embedding_mod.available_cpu_count() == 4
