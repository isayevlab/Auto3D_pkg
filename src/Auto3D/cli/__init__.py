"""Modern CLI for Auto3D using Typer and Rich."""

# Lazy imports to avoid circular dependencies and missing module errors
# during incremental CLI development

__all__ = ["app", "console"]


def __getattr__(name: str):
    """Lazy import for CLI components."""
    if name == "app":
        from Auto3D.cli.app import app
        return app
    if name == "console":
        from Auto3D.cli.console import console
        return console
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
