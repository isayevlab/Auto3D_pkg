"""A named-backend registry, with no knowledge of what is registered.

Auto3D has two families of swappable backend -- neural network potentials and
isomer engines -- and until now each carried its own bespoke lookup: a dict plus
an if-chain for models, a tuple plus an if/elif ladder for isomer engines. Adding
a model backend meant editing five places and an isomer backend six, and one of
those places was the *presentation* layer, where `ENGINE_INFO` kept a parallel
table of display metadata that nothing checked against the set of real backends.
A backend registered without an entry there simply stopped appearing in
`auto3d models info`.

This module is the shared half of both. It deliberately does **not** own
construction: a model adapter is built with ``(device, compile_model)`` and an
isomer engine with eight-odd keyword arguments, so a signature both satisfy would
be a bag of keywords that hides what each backend needs and stops a type checker
helping. Each factory resolves through the registry and then calls its own
constructor.

It is also not a plugin system. Auto3D already accepts a third-party model as a
file path -- ``--engine /path/to/my_nnp.pt``, checked against the ``CustomNNP``
contract at load -- so entry-point discovery would add only *named, installable*
backends, and would mean freezing ``ModelAdapter`` as a public interface while it
is still gaining members. A registry is what such a loader would populate, so
this does not foreclose it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from Auto3D.exceptions import ConfigurationError

T = TypeVar("T")


@dataclass(frozen=True)
class Entry(Generic[T]):
    """One registered backend: the thing itself, plus how to talk about it.

    ``info`` is free-form per-registry metadata -- for models it is what
    ``auto3d models info`` prints. It lives here rather than in a table beside
    the registry because a second table keyed by the same names is exactly the
    drift this module exists to remove.
    """

    name: str
    value: T
    aliases: tuple[str, ...] = ()
    info: Any = None


class Registry(Generic[T]):
    """Names to backends, with one place that explains an unknown name.

    Args:
        kind: What is being registered, used in error messages -- "optimizing
            engine", "isomer engine". Phrased as a noun so the message reads as
            a sentence.
        case_insensitive: Fold names before lookup. Models resolve
            case-insensitively (``--engine ani2x`` works) while isomer engine
            types are exact lowercase; both behaviors are current and tested, so
            the difference is configuration here rather than something this
            module decides.
    """

    def __init__(self, kind: str, *, case_insensitive: bool = False) -> None:
        self._kind = kind
        self._case_insensitive = case_insensitive
        self._entries: dict[str, Entry[T]] = {}
        self._lookup: dict[str, str] = {}

    def _key(self, name: str) -> str:
        return name.upper() if self._case_insensitive else name

    def register(
        self,
        name: str,
        value: T,
        *,
        aliases: tuple[str, ...] = (),
        info: Any = None,
    ) -> None:
        """Add a backend. Registering a name twice is an error, not an overwrite.

        A plain dict silently replaces, which turns a duplicate -- two modules
        registering the same name, or one imported twice under different paths --
        into a backend that works or does not depending on import order.
        """
        for candidate in (name, *aliases):
            key = self._key(candidate)
            if key in self._lookup:
                raise ConfigurationError(
                    f"{self._kind} {candidate!r} is already registered "
                    f"(as {self._lookup[key]!r}); names must be unique."
                )
        self._entries[name] = Entry(name=name, value=value, aliases=aliases, info=info)
        for candidate in (name, *aliases):
            self._lookup[self._key(candidate)] = name

    def entry(self, name: str) -> Entry[T]:
        """The full :class:`Entry`, for a caller that needs ``info`` too."""
        key = self._key(name)
        if key not in self._lookup:
            raise ConfigurationError(
                f"Unknown {self._kind} {name!r}. Available: "
                + ", ".join(repr(n) for n in self.available())
                + "."
            )
        return self._entries[self._lookup[key]]

    def resolve(self, name: str) -> T:
        """The registered value, or a ``ConfigurationError`` naming the alternatives."""
        return self.entry(name).value

    def available(self) -> list[str]:
        """Registered names, canonical spelling only, in registration order."""
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return self._key(name) in self._lookup

    def __len__(self) -> int:
        return len(self._entries)
