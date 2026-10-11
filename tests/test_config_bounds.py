"""Field-bound, selector, and sentinel-handling tests for Auto3DOptions."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_opt_steps_has_exactly_one_declared_minimum():
    """``opt_steps`` had TWO minimums: ``FIELD_BOUNDS`` declared ``("ge", 1)``
    while ``utils/validation.py`` hand-wrote ``< 10`` twice (in ``check_input``
    and again in ``check_valid_configuration``). So ``opt_steps=5`` was accepted
    by ``Auto3DOptions``/``CLIConfig``, printed a banner, and only then failed
    at run start.

    10 won, not 1 -- see the comment on ``FIELD_BOUNDS["opt_steps"]`` for why
    (the optimizer's own 10-step cadences, and its explicit ``n >= 10`` guard).
    This test asserts the *consolidation*, in both directions: the floor is
    declared once, in ``FIELD_BOUNDS``, and no validator downstream carries a
    second copy of it.
    """
    from Auto3D.engines.models import policy as policy_mod
    from Auto3D.foundation.config import FIELD_BOUNDS, Auto3DOptions, check_field_bounds
    from Auto3D.foundation.exceptions import ConfigurationError
    from Auto3D.orchestration.pipeline import input_checks as input_checks_mod

    kind, bound_min = FIELD_BOUNDS["opt_steps"]
    assert (kind, bound_min) == ("ge", 10)

    # The declared floor is accepted; one below it is refused -- and refused at
    # construction, on the object every entry point builds, not later.
    check_field_bounds({"opt_steps": bound_min})  # must not raise
    Auto3DOptions(path="x.smi", k=1, opt_steps=int(bound_min))
    with pytest.raises(ConfigurationError, match="opt_steps"):
        Auto3DOptions(path="x.smi", k=1, opt_steps=int(bound_min) - 1)

    # And utils/validation.py no longer restates the number anywhere. A source
    # check rather than a behavioral one: a second copy that happens to agree
    # today is exactly how the two drifted apart in the first place, and
    # behavior cannot distinguish "one bound" from "two bounds that match".
    #
    # AST, not a substring scan over lines: the prose explaining why the checks
    # were removed necessarily mentions `opt_steps` and `< 10`, so a text match
    # flags the comment that documents the fix. A `Compare` node is the thing
    # actually forbidden here.
    import ast

    # Both halves of what `utils/validation.py` split into, so the bound cannot
    # reappear in whichever one a future edit happens to touch.
    offenders = []
    for module in (policy_mod, input_checks_mod):
        tree = ast.parse(Path(module.__file__).read_text())
        offenders += [
            f"{Path(module.__file__).name}: {ast.unparse(node)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and "opt_steps" in ast.unparse(node)
            and any(
                isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
                for c in node.comparators
            )
        ]
    assert not offenders, (
        f"an opt_steps bound is hand-written again: {offenders}. "
        "The bound belongs in Auto3D.foundation.config.FIELD_BOUNDS only."
    )


def test_negative_k_rejected():
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=-1)


def test_negative_window_rejected():
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", window=-0.5)


def test_zero_k_rejected():
    """k=0 used to be treated as "not specified" (falsy) and silently
    accepted, but CLIConfig has always rejected it (its k >= 1 bound applies
    to any non-None value, and 0 is not None) -- Auto3DOptions must match,
    per Task 1's "one set of bounds, on every path". Only None/False mean
    "not specified" now; see test_default_and_valid_k_window_accepted for
    those sentinels.
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=0)


def test_default_and_valid_k_window_accepted():
    from Auto3D.foundation.config import Auto3DOptions

    # defaults (None) and valid positives must NOT raise
    Auto3DOptions(path="x.smi")
    Auto3DOptions(path="x.smi", k=5)
    Auto3DOptions(path="x.smi", window=2.0)
    Auto3DOptions(path="x.smi", k=None)  # None is "not specified", allowed


def test_false_is_refused_as_a_sentinel_on_every_field_that_has_one():
    """``False`` no longer means "not specified" on any of the sentinel fields.

    It used to, on ``Auto3DOptions`` only -- ``CLIConfig`` spelled the same idea
    ``None``, and a translation function converted between them on the way
    across. One sentinel means that function is gone, and it means ``False`` has
    to be refused rather than coerced: ``bool`` is an ``int`` subclass, so
    pydantic turns it into ``0`` and the user is told their value is below a
    minimum they never approached.

    The shipped ``docs/legacy-v2/parameters.yaml`` set ``window: False`` and was
    updated with this change, so the in-repo example still loads.
    """
    from Auto3D.foundation.config import SENTINEL_FIELDS, Auto3DOptions
    from Auto3D.foundation.exceptions import ConfigurationError

    for field in sorted(SENTINEL_FIELDS):
        with pytest.raises(ConfigurationError, match="None, not False"):
            Auto3DOptions(path="x.smi", **{field: False})

    # None is accepted on all of them, together and mixed with a real value.
    opts = Auto3DOptions(
        path="x.smi", k=None, window=None, memory=None, max_confs=None, parallel_workers=None
    )
    assert (opts.k, opts.window, opts.memory, opts.max_confs, opts.parallel_workers) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert Auto3DOptions(path="x.smi", k=1, window=None).k == 1


def test_non_numeric_threshold_raises_configuration_error():
    """A non-numeric bound value (e.g. threshold="0.3", a str) must raise
    ConfigurationError, not a bare TypeError.

    `operator.gt`/`operator.ge` (config.py's `_BOUND_OPS`) raise TypeError
    when compared against a str, which used to propagate unhandled from
    `check_field_bounds` -- an untyped exception unlike every range check
    beside it, and one the CLI's `handle_error` shows as an opaque
    "Unexpected Error" (exit 1) instead of a configuration problem with a
    hint (exit 2).
    """
    from Auto3D.foundation.config import Auto3DOptions
    from Auto3D.foundation.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Auto3DOptions(path="x.smi", k=1, threshold="not-a-number")

    # A numeric *string* is a different case and is accepted: YAML hands every
    # scalar over as text, so refusing "0.3" would refuse a valid config file.
    assert Auto3DOptions(path="x.smi", k=1, threshold="0.3").threshold == 0.3


class TestSentinelsAreNotSilentlyReinterpreted:
    """`k=True` passed every gate and meant `k=1`.

    bool is a subclass of int, so `operator.ge(True, 1)` is True and the bounds
    check let it through; `top_k`'s `if k == 1` then matched. The effect was
    harmless, but `k: int | bool = False` advertises a bool where only `False` was
    ever meant as a sentinel, so `True` was a value the type called legal and
    nothing gave a meaning to. A caller who wrote it meant something, and it was
    not "one conformer".
    """

    def test_k_true_is_rejected_rather_than_read_as_one(self):
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.foundation.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="got True"):
            Auto3DOptions(path="in.smi", k=True)

    def test_k_false_is_refused_and_says_to_use_none(self):
        """``False`` was the "use window instead" spelling; ``None`` is now.

        It must not be quietly reinterpreted. ``bool`` is an ``int`` subclass,
        so without an explicit guard pydantic coerces ``False`` to ``0`` and the
        user is told ``k must be >= 1, got 0`` -- a complaint about a value they
        never wrote. The message has to name the replacement instead.
        """
        from Auto3D.foundation.config import Auto3DOptions
        from Auto3D.foundation.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="None, not False"):
            Auto3DOptions(path="in.smi", k=False, window=2.0)

        assert Auto3DOptions(path="in.smi", k=None, window=2.0).window == 2.0

    def test_a_real_k_is_unaffected(self):
        from Auto3D.foundation.config import Auto3DOptions

        assert Auto3DOptions(path="in.smi", k=1).k == 1
