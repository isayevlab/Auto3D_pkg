"""Modern CLI for Auto3D using Typer and Rich.

This package re-exports nothing. ``docs/source/api.rst`` documents no
``Auto3D.presentation.cli`` name, so nothing here is public: import from the module that
defines it (``Auto3D.presentation.cli.app`` for the Typer application, ``Auto3D.presentation.cli.console``
for the output helpers) and expect it to move.

Emptying this barrel also removed a name collision it created. It bound ``app``
(the Typer instance from ``Auto3D.presentation.cli.app``) and ``console`` (the Rich console
from ``Auto3D.presentation.cli.console``) *over the modules of the same names*, so
``from Auto3D.presentation.cli import app`` yielded the Typer object once ``__init__`` had run
and the module otherwise. Both names now resolve to their modules,
unambiguously.
"""
