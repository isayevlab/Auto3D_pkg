"""Parallel-embedding reachability and batchsize_atoms scaling parity tests."""

from __future__ import annotations

import pytest


class TestParallelEmbeddingIsReachable:
    """The option must arrive at the isomer engine, not just exist on the config.

    Until 3.0.0 `use_parallel_embedding` was a constructor argument on the isomer
    engine with no route from `Auto3DOptions`, so no `main()` or `smiles2mols` run
    could reach it and the parallel embedder was reachable only from tests --
    which is why an audit listed that module as dead code.

    Asserting `Auto3DOptions(use_parallel_embedding=True).use_parallel_embedding
    is True` would pass with the plumbing still missing: it tests the dataclass,
    not the wiring. These assert what the factory is actually called with.
    """

    def test_the_option_reaches_the_isomer_engine_factory(self, monkeypatch, tmp_path):
        from Auto3D.entry import auto3D as auto3D_mod
        from Auto3D.foundation.config import Auto3DOptions

        seen = {}

        class _StubEngine:
            def run(self):
                raise RuntimeError("stop here: the factory call is what is asserted")

        def _capture(**kwargs):
            seen.update(kwargs)
            return _StubEngine()

        monkeypatch.setattr(auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture))

        smi = tmp_path / "in.smi"
        smi.write_text("CCO ethanol\n")
        options = Auto3DOptions(
            # use_gpu=False: this box and CI are CPU-only, and check_gpu_requested
            # is fatal for a GPU request with no visible device -- it would fire
            # before the factory call under test.
            path=str(smi),
            k=1,
            use_gpu=False,
            use_parallel_embedding=True,
            parallel_workers=3,
            parallel_embedding_threshold=2,
        )

        with pytest.raises(RuntimeError, match="stop here"):
            # `parallel_embedding=True` because `smiles2mols` gates the pool on
            # its own keyword rather than on `args.use_parallel_embedding`; the
            # other two fields still come from the config, which is what this
            # test is about.
            auto3D_mod.smiles2mols(["CCO"], options, parallel_embedding=True)

        assert seen.get("use_parallel_embedding") is True, (
            "parallel embedding never reached the isomer engine: the option "
            f"exists but is not plumbed. Factory got: {sorted(seen)}"
        )
        assert seen.get("parallel_workers") == 3, (
            "parallel_workers stayed at the constructor default, so enabling "
            "parallel embedding could not control its worker count"
        )
        assert seen.get("parallel_embedding_threshold") == 2, (
            "parallel_embedding_threshold stayed at its default, so the batch-size "
            "gate could not be tuned"
        )

    @pytest.mark.parametrize(
        "shape,kwargs,expected",
        [
            # (i) A freshly built default config and no keyword: smiles2mols keeps
            # its documented single-process contract even though the field now
            # defaults to True.
            ("default", {}, False),
            # (ii) The field says True and the keyword is absent. smiles2mols does
            # not read the field at all, so this is still serial -- and the
            # docstrings say so on both sides.
            ("field_true", {}, False),
            # (iii) A `replace()`d config. `replace()` rebuilds the model through
            # its constructor, so every field comes out marked as explicitly set:
            # an opt-in inferred from `model_fields_set` could not tell this from
            # a choice, and silently spawned the pool for a caller who never
            # mentioned the field.
            ("replaced", {}, False),
            # (iv) A `model_dump()` round trip -- the shape `load_yaml_config`
            # produces, so this is also the `parameters.yaml` this repo ships,
            # which names `use_parallel_embedding` explicitly.
            ("round_tripped", {}, False),
            # (v) The keyword is the opt-in, and it is honored.
            ("default", {"parallel_embedding": True}, True),
        ],
    )
    def test_smiles2mols_embeds_in_parallel_only_on_its_own_keyword(
        self, monkeypatch, tmp_path, shape, kwargs, expected
    ):
        """``smiles2mols`` runs in the caller's own process, so it may not take
        the spawn path unless the caller asked for it *at the call*.

        ``main()`` already required the caller's script to be importable without
        side effects -- it has always spawned its workers from an explicit spawn
        context, and every documented ``main()`` example carries an
        ``if __name__ == "__main__":`` guard. ``smiles2mols`` is documented as the
        single-process convenience API and none of its examples have that guard,
        so honoring the flipped default here would have made every such script
        above ``parallel_embedding_threshold`` die with ``BrokenProcessPool`` from
        a re-imported ``__main__``.

        The gate is a keyword-only argument on the function, not an inference from
        ``args.model_fields_set``: that inference read as "did the caller name this
        field", which is not a question a config object can answer. ``replace()``
        and every YAML-built config mark all fields as set, so two documented,
        ordinary ways of building a config defeated it. A keyword cannot be
        defeated by how the config was built.
        """
        from Auto3D.entry import auto3D as auto3D_mod
        from Auto3D.foundation.config import Auto3DOptions

        seen = {}

        class _StubEngine:
            def run(self):
                raise RuntimeError("stop here: the factory call is what is asserted")

        def _capture(**kwargs):
            seen.update(kwargs)
            return _StubEngine()

        monkeypatch.setattr(auto3D_mod.IsomerEngineFactory, "create", staticmethod(_capture))

        smi = tmp_path / "in.smi"
        smi.write_text("CCO ethanol\n")
        if shape == "default":
            options = Auto3DOptions(path=str(smi), k=1, use_gpu=False)
        elif shape == "field_true":
            options = Auto3DOptions(path=str(smi), k=1, use_gpu=False, use_parallel_embedding=True)
        elif shape == "replaced":
            options = Auto3DOptions(path=str(smi), k=1, use_gpu=False).replace(max_confs=5)
        else:
            built = Auto3DOptions(path=str(smi), k=1, use_gpu=False)
            options = Auto3DOptions(**built.model_dump())

        with pytest.raises(RuntimeError, match="stop here"):
            auto3D_mod.smiles2mols(["CCO"], options, **kwargs)

        assert seen.get("use_parallel_embedding") is expected, (
            f"smiles2mols on a {shape!r} config called with {kwargs or 'no keyword'} "
            f"handed the isomer engine use_parallel_embedding="
            f"{seen.get('use_parallel_embedding')!r}, expected {expected!r}"
        )

    def test_parallel_embedding_is_the_default(self):
        """On by default since 3.2.0, with the worker count left unresolved.

        Was ``test_the_default_is_still_serial``. The serial path did not go
        away -- it still runs below ``parallel_embedding_threshold`` and for
        ``--no-parallel-embedding`` -- but a default of 4 workers behind an
        off-by-default switch meant the ordinary run embedded one species at a
        time however many cores the box had (P-C3, D1). ``parallel_workers``
        is ``None`` rather than a number because the useful count depends on
        the machine and on how many species this run actually has, neither of
        which a class default can know.
        """
        from Auto3D.foundation.config import Auto3DOptions

        options = Auto3DOptions(path="in.smi", k=1)
        assert options.use_parallel_embedding is True
        assert options.parallel_workers is None

    @pytest.mark.parametrize("field", ["parallel_workers", "parallel_embedding_threshold"])
    def test_a_count_below_one_is_rejected(self, field):
        """Bounds come from FIELD_BOUNDS, so both entry points share them."""
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.foundation.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match=field):
            Auto3DOptions(path="in.smi", k=1, **{field: 0})

    def test_an_unset_worker_count_is_accepted(self):
        """``parallel_workers=None`` is the default, so the bound must skip it.

        The bound still applies to a concrete int (the sibling test above), so
        ``parallel_workers`` belongs in ``SENTINEL_FIELDS`` rather than out of
        ``FIELD_BOUNDS``: None means "resolve from the machine", 0 means a
        worker count nothing can run.
        """
        from Auto3D.foundation.config import Auto3DOptions

        assert Auto3DOptions(path="in.smi", k=1, parallel_workers=None).parallel_workers is None


def test_smiles2mols_scales_batchsize_atoms_like_main(monkeypatch, tmp_path):
    """``smiles2mols`` must scale ``batchsize_atoms`` by memory exactly as ``main()`` does.

    ``main()`` routes the per-gigabyte field through ``ChunkManager`` and hands
    the optimizer the scaled product; ``smiles2mols`` read the field as an
    absolute count, so the same ``Auto3DOptions`` optimized in batches up to
    16x smaller through this entry point than through the other one -- a
    difference nothing in either signature or docstring announced.

    The stubs are the ones the sibling spy tests in this file use, plus the two
    this one needs to reach the optimizer at all: ``preflight_model`` (the
    default AIMNET engine would otherwise be resolved against the model
    registry/cache) and ``create_model``. ``Optimizer`` is the assertion
    point, so it records the config and stops the run there.
    """
    from Auto3D.entry import auto3D as auto3D_mod
    from Auto3D.foundation.config import Auto3DOptions
    from tests.helpers_adapter import FakeAdapter

    class _StubIsomerEngine:
        def run(self):
            return None

    captured = {}

    def _capture_optimizing(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here: the optimizer config is what is asserted")

    monkeypatch.setattr(
        auto3D_mod.IsomerEngineFactory, "create", staticmethod(lambda **kw: _StubIsomerEngine())
    )
    monkeypatch.setattr(auto3D_mod, "preflight_model", lambda *a, **k: None)
    monkeypatch.setattr(auto3D_mod, "create_model", lambda *a, **k: FakeAdapter())
    monkeypatch.setattr(auto3D_mod, "Optimizer", _capture_optimizing)

    options = Auto3DOptions(
        path=str(tmp_path / "unused.smi"),
        k=1,
        use_gpu=False,
        # memory=4 fixes the multiplier: ChunkManager reads the explicit budget
        # and never asks psutil or nvidia-smi, so this asserts a number rather
        # than whatever this machine happens to have free.
        memory=4,
        batchsize_atoms=1024,
    )

    with pytest.raises(RuntimeError, match="stop here"):
        auto3D_mod.smiles2mols(["CCO"], options)

    assert captured["config"].batchsize_atoms == 1024 * 4, (
        "smiles2mols handed the optimizer an unscaled batchsize_atoms: "
        f"{captured['config'].batchsize_atoms}, expected the per-GB 1024 scaled "
        "by the 4 GB memory budget"
    )
    # The caller's config is untouched -- smiles2mols copies it (M15) and the
    # field stays per-gigabyte everywhere.
    assert options.batchsize_atoms == 1024
