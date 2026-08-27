Building the conda package
==========================

This page is for packagers. If you just want to *install* Auto3D, see
:doc:`../installation` -- and note that from **3.1.1**, the conda-forge
package ships without the AIMNet2 engines, for the reason explained below.

.. important::

   **From 3.1.1, the recipe will deliberately exclude** ``aimnet``.
   conda-forge requires every runtime dependency to exist as a conda package,
   and ``aimnet`` does not: its own dependency ``nvalchemi-toolkit-ops`` is
   pip-only. Rather than block the whole package on that, the updated
   ``requirements.run`` will be ``pyproject.toml``'s ``dependencies`` with
   ``aimnet`` removed and ``torchani`` plus ``ase`` added, so the ANI2x and
   ANI2xt engines and the ASE thermochemistry APIs will work with no extra
   install. AIMNet2 will be added afterward, in the same environment, with
   ``pip install aimnet``.

The recipe
----------

The recipe lives in ``conda-recipe/meta.yaml`` in the repository, next to
``pyproject.toml``, so the dependency mapping can be updated in the same commit
as a dependency change. conda-forge itself builds from a separate *feedstock*
repository, not from this directory; the in-tree copy is the source of truth
that a feedstock update is derived from, and it is updated as part of each
release.

The in-tree recipe has not yet been updated to the 3.1.1 shape described
below -- at this commit it still matches 3.0.0: ``aimnet`` is still a
``requirements.run`` entry, and the test phase still runs ``pip check``. The
rest of this page describes what the 3.1.1 recipe update will carry, applied
as part of that release.

Building locally
----------------

You need ``conda-build``, and every *run* dependency the recipe declares must
be resolvable in your channels. At this commit that still includes
``aimnet``, so it has to come from somewhere -- a local channel, or one you
control -- because it is not on conda-forge. From 3.1.1, the recipe drops
``aimnet`` from ``requirements.run`` and this step goes away.

.. code:: console

   conda install -n base conda-build
   conda build conda-recipe/ -c conda-forge

The build runs the recipe's test phase at the end: it imports ``Auto3D``,
runs ``auto3d --help`` and ``auto3d models list``, and -- at this commit --
runs ``pip check``. All four are deliberately offline -- conda-forge's test
phase has no network access, so a test that downloaded a model would pass
locally and fail there.

From 3.1.1, ``pip check`` drops out of the test phase. The package's own pip
metadata declares ``aimnet`` a required dependency -- true for the pip
install -- but the updated conda package deliberately does not install it, so
``pip check`` would flag a missing requirement that is expected, not a bug.
Leaving it in would turn an intentional gap into permanent build noise.

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

   At this commit the script reports no mismatches -- the recipe still
   mirrors ``pyproject.toml`` exactly. Once the 3.1.1 update above lands,
   expect two mismatches every run, both intentional: ``aimnet`` will show up
   under "in pyproject but not the recipe" (it is deliberately excluded, see
   above), and ``torchani``/``ase`` will show up under "in the recipe but not
   pyproject" (optional pip extras upstream, but unconditional conda run
   dependencies from 3.1.1, for the same reason). Anything else in either
   list is a real gap to reconcile.

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

From 3.1.1, ``torchani`` and ``ase`` become run dependencies; ``aimnet`` does not
   Without ``aimnet``, a plain ``conda install`` still needs to leave a user
   with a working optimization engine and thermochemistry, so ``torchani``
   (the ANI2x/ANI2xt engines) and ``ase`` (thermochemistry and geometry
   optimization) -- both already on conda-forge -- become unconditional run
   dependencies of the recipe, even though they remain optional pip extras
   upstream. ``aimnet`` is the one dependency left off, for the reason given
   at the top of this page.

Getting AIMNet2 onto conda-forge
---------------------------------

From 3.1.1 this no longer blocks Auto3D's own feedstock: the recipe will not
depend on ``aimnet``, so the existing ``auto3d`` feedstock can be updated
from ``conda-recipe/meta.yaml`` the same way as any other release -- no
upstream packaging required.

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
