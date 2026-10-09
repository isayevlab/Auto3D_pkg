"""Model contracts, adapters and neural network potentials for Auto3D.

This package re-exports nothing. ``docs/source/api.rst`` documents two names out
of it, each at its own module path -- :class:`Auto3D.engines.models.contract.CustomNNP`,
and :class:`Auto3D.engines.models.contract.ModelAdapter`'s consumer half (``forward``,
``energy`` and ``to_species``, public and frozen for callers of ``create_model``,
decision D4, 3.2.0) -- so those are the only public surface here and the only
supported way to import it. Everything else (the adapters, the rest of the
adapter interface, the loaders) is internal: import it from the module that
defines it and expect it to move.

Both contracts live in :mod:`Auto3D.engines.models.contract`:

* :class:`~Auto3D.engines.models.contract.CustomNNP` -- what a user's own NNP must
  satisfy, ``forward(species, coords, charges) -> energies``.
* :class:`~Auto3D.engines.models.contract.ModelAdapter` -- what Auto3D's internals talk
  to, ``forward(coords, species, charges, atom_mask=None) -> (energies, forces)``.

The two take ``species`` and ``coords`` in opposite order, deliberately and
permanently. Read that module's docstring before touching either.

``ModelAdapter`` used to be re-exported here while ``CustomNNP`` was documented
one level deeper, which put the internal interface at a *shallower* path than
the public one and made ``Auto3D.engines.models.ModelAdapter`` look like the surface a
user implements. Both now sit side by side in ``contract``, one import away, and
neither is reachable from this package's namespace.
"""
