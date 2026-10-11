"""Retired lowercase names stay importable for one release and warn."""

from __future__ import annotations

import warnings


def test_the_three_legacy_names_resolve_to_the_new_classes_and_warn():
    import Auto3D.domain.ranking as ranking_mod
    import Auto3D.engines.batch_opt.batchopt as batchopt_mod
    import Auto3D.engines.batch_opt.model_wrapper as wrapper_mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert batchopt_mod.optimizing is batchopt_mod.Optimizer
        assert wrapper_mod.EnForce_ANI is wrapper_mod.BatchedEnergyForce
        assert ranking_mod.ranking is ranking_mod.ConformerRanker
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 3
    assert any("Optimizer" in m for m in messages)
    assert any("BatchedEnergyForce" in m for m in messages)
    assert any("ConformerRanker" in m for m in messages)
