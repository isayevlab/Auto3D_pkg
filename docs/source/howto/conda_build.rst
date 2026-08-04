Building the conda package
==========================

This page is for packagers. If you just want to *install* Auto3D, see
:doc:`../installation` -- and note that ``conda install -c conda-forge auto3d``
currently gives you **2.3.0**, not 3.0.0, for the reason explained below.

.. important::

   **Auto3D 3.0.0 is not on conda-forge, and cannot be until two upstream
   packages are added there.** conda-forge requires every runtime dependency to
   exist as a conda package, and two of Auto3D's do not:

   - ``aimnet`` -- a core dependency as of 3.0.0
   - ``nvalchemi-toolkit-ops`` -- required by ``aimnet`` itself

   Until those exist, ``pip install Auto3D`` is the only way to get 3.0.0.

The recipe
----------

The recipe lives in ``conda-recipe/meta.yaml`` in the repository, next to
``pyproject.toml``, so the dependency mapping can be updated in the same commit
as a dependency change. conda-forge itself builds from a separate *feedstock*
repository, not from this directory; the in-tree copy is the source of truth
that a feedstock update is derived from.

Building locally
----------------

You need ``conda-build``, and every runtime dependency must be resolvable in
your channels. Today that means ``aimnet`` has to come from somewhere -- a local
channel, or a channel you control -- because it is not on conda-forge.

.. code:: console

   conda install -n base conda-build
   conda build conda-recipe/ -c conda-forge

The build runs the recipe's test phase at the end: it imports ``Auto3D``, runs
``auto3d --help`` and ``auto3d models list``, and runs ``pip check``. All four
are deliberately offline -- conda-forge's test phase has no network access, so a
test that downloaded a model would pass locally and fail there.

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

Optional extras are not run dependencies
   ``ase`` (thermochemistry and geometry optimization) and ``torchani``
   (the ANI2x engine) are installed alongside the package when wanted. Both
   *are* on conda-forge, so ``conda install ase torchani`` works.

Getting 3.0.0 onto conda-forge
------------------------------

The order matters, because ``aimnet`` depends on the first one:

1. Submit a feedstock for **``nvalchemi-toolkit-ops``**.
2. Submit a feedstock for **``aimnet``**.
3. Update the existing ``auto3d`` feedstock from ``conda-recipe/meta.yaml``.

Both new packages are upstream projects Auto3D does not own. Submitting a
feedstock for someone else's package is normal on conda-forge, but it means
committing to maintain it -- so it is worth asking the ``aimnet`` maintainers
whether they would rather own it.

There is no recipe-level shortcut. conda-forge forbids network access during
builds and forbids ``pip install``\ ing a dependency that is not a conda
package, so the alternatives are all worse than waiting: making ``aimnet``
optional again would contradict the 3.0.0 design in which AIMNet2 is the default
engine, and publishing to a personal channel splits the install story and gives
up conda-forge's dependency resolution.

The 2.3.0 gap is separate
-------------------------

conda-forge ships 2.3.0 while PyPI's release before 3.0.0 was 2.3.1. That gap
predates all of the above and is **not** blocked by ``aimnet``, because 2.3.x did
not depend on it. If a conda user needs 2.3.1 specifically, that is a
version-and-hash bump on the existing feedstock and can be done independently.
