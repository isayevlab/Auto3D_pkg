Building the conda package
==========================

This page is for packagers. If you just want to *install* Auto3D, see
:doc:`../installation` -- and note that from **3.1.1**, the conda-forge
package ships without the AIMNet2 engines, for the reason explained below.

.. important::

   **The recipe deliberately excludes** ``aimnet``. conda-forge requires every
   runtime dependency to exist as a conda package, and ``aimnet`` does not:
   its own dependency ``nvalchemi-toolkit-ops`` is pip-only. Rather than block
   the whole package on that, from 3.1.1 ``requirements.run`` is
   ``pyproject.toml``'s ``dependencies`` with ``aimnet`` removed and
   ``torchani`` plus ``ase`` added, so the ANI2x and ANI2xt engines and the
   ASE thermochemistry APIs work with no extra install. AIMNet2 is added
   afterward, in the same environment, with ``pip install aimnet``.

The recipe
----------

The recipe lives in ``conda-recipe/meta.yaml`` in the repository, next to
``pyproject.toml``, so the dependency mapping can be updated in the same commit
as a dependency change. conda-forge itself builds from a separate *feedstock*
repository, not from this directory; the in-tree copy is the source of truth
that a feedstock update is derived from. It is updated as part of each
release -- this page describes its 3.1.1 shape, the one the upcoming
feedstock update is expected to carry.

Building locally
----------------

You need ``conda-build``. Every *run* dependency the recipe declares must be
resolvable in your channels -- ``aimnet`` is not one of them, so it needs no
channel of its own.

.. code:: console

   conda install -n base conda-build
   conda build conda-recipe/ -c conda-forge

The build runs the recipe's test phase at the end: it imports ``Auto3D`` and
runs ``auto3d --help`` and ``auto3d models list``. All three are deliberately
offline -- conda-forge's test phase has no network access, so a test that
downloaded a model would pass locally and fail there.

The test phase does not run ``pip check``. The package's own pip metadata
declares ``aimnet`` a required dependency -- true for the pip install -- but
the conda package deliberately does not install it, so ``pip check`` would
flag a missing requirement that is expected, not a bug. Running it here would
turn an intentional gap into permanent build noise.

To build from your working tree instead of the published sdist, replace the
``source:`` block with:

.. code:: yaml

   source:
     path: ..

and delete the ``sha256`` line. This is the right form while iterating, since it
picks up uncommitted changes.

Installing what you built
-------------------------

.. code:: console

   conda install -c local auto3d
   # or point at the build output directly
   conda install $(conda build conda-recipe/ --output)

Updating the recipe for a new release
-------------------------------------

1. Bump ``version`` in ``meta.yaml``.

2. Replace ``sha256``. **Take it from PyPI, not from a local build** -- a locally
   built sdist is not guaranteed byte-identical to the one the release workflow
   uploaded, and conda-forge verifies the hash against the file it downloads:

   .. code:: console

      curl -s https://pypi.org/pypi/Auto3D/json \
        | python -c "import json,sys; d=json.load(sys.stdin); \
          print(next(f['digests']['sha256'] for f in d['releases'][d['info']['version']] \
          if f['filename'].endswith('.tar.gz')))"

3. Reconcile ``requirements.run`` against ``pyproject.toml``'s
   ``dependencies``. **Two names differ between PyPI and conda-forge**, and
   getting either wrong produces an unsatisfiable recipe rather than an obvious
   error:

   .. list-table::
      :header-rows: 1

      * - PyPI name
        - conda-forge name
      * - ``torch``
        - ``pytorch``
      * - ``Send2Trash``
        - ``send2trash``

   This check is worth scripting rather than eyeballing:

   .. code:: console

      python - <<'PY'
      import re, tomllib, pathlib, yaml
      raw = pathlib.Path("conda-recipe/meta.yaml").read_text()
      r = re.sub(r"{%.*?%}", "", raw)
      for a, b in (("{{ name|lower }}", "auto3d"), ("{{ version }}", "0"),
                   ("{{ name[0] }}", "a"), ("{{ name }}", "auto3d"),
                   ("{{ PYTHON }}", "python")):
          r = r.replace(a, b)
      run = {x.split()[0] for x in yaml.safe_load(r)["requirements"]["run"]} - {"python"}
      pypi = {re.split(r"[><=!]", x)[0].strip().lower()
              for x in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]}
      mapped = {x.replace("pytorch", "torch") for x in run}
      print("in pyproject but not the recipe:", sorted(pypi - mapped) or "none")
      print("in the recipe but not pyproject:", sorted(mapped - pypi) or "none")
      PY

   Expect two mismatches every run, both intentional: ``aimnet`` shows up
   under "in pyproject but not the recipe" (it is deliberately excluded, see
   above), and ``torchani``/``ase`` show up under "in the recipe but not
   pyproject" (they are optional pip extras upstream but unconditional conda
   run dependencies, for the same reason). Anything else in either list is a
   real gap to reconcile.

Why the recipe is shaped the way it is
--------------------------------------

``noarch: python``
   Auto3D is pure Python and its own wheel is ``py3-none-any``, so one build
   serves every platform. ``pytorch`` and ``rdkit`` being compiled does not
   change that: they are resolved at install time, not build time.

The test phase imports only the top-level package
   Since 3.0.0, ``import Auto3D`` loads neither torch nor RDKit (0.03 s, 154
   modules), so this stays cheap. Importing a submodule would pull the whole
   dependency tree into the test phase for no additional signal.

``torchani`` and ``ase`` are run dependencies; ``aimnet`` is not
   Without ``aimnet``, a plain ``conda install`` still needs to leave a user
   with a working optimization engine and thermochemistry, so ``torchani``
   (the ANI2x/ANI2xt engines) and ``ase`` (thermochemistry and geometry
   optimization) -- both already on conda-forge -- are unconditional run
   dependencies of the recipe, even though they are optional pip extras
   upstream. ``aimnet`` is the one dependency left off, for the reason given
   at the top of this page.

Getting AIMNet2 onto conda-forge
---------------------------------

This no longer blocks Auto3D's own feedstock. From 3.1.1, the recipe does not
depend on ``aimnet``, so the existing ``auto3d`` feedstock can be updated from
``conda-recipe/meta.yaml`` the same way as any other release -- no upstream
packaging required.

Upstream packaging is still the path to a fully conda-native AIMNet2 install,
though, and the order matters, because ``aimnet`` depends on the first one:

1. Submit a feedstock for **``nvalchemi-toolkit-ops``**.
2. Submit a feedstock for **``aimnet``**.

Both are upstream projects Auto3D does not own. Submitting a feedstock for
someone else's package is normal on conda-forge, but it means committing to
maintain it -- so it is worth asking the ``aimnet`` maintainers whether they
would rather own it. Until then, ``pip install aimnet`` inside the conda
environment is not a workaround: it is how the recipe is designed to hand off
that one dependency, from 3.1.1 on.

The 2.3.0 gap is separate
-------------------------

conda-forge ships 2.3.0 while PyPI's release before 3.0.0 was 2.3.1. That gap
predates all of the above and is **not** blocked by ``aimnet``, because 2.3.x did
not depend on it. If a conda user needs 2.3.1 specifically, that is a
version-and-hash bump on the existing feedstock and can be done independently.
