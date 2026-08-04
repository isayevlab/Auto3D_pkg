# conda recipe for Auto3D

`meta.yaml` is the recipe for Auto3D. It lives in-tree so the dependency mapping
sits next to `pyproject.toml` and can be updated in the same commit as a
dependency change; conda-forge builds from a separate feedstock repository, not
from this directory.

**The documentation for this recipe is
[`docs/source/howto/conda_build.rst`](../docs/source/howto/conda_build.rst)**
(rendered at
[auto3d.readthedocs.io/en/latest/howto/conda_build.html](https://auto3d.readthedocs.io/en/latest/howto/conda_build.html)).
It covers building locally, updating the recipe for a new release, why the recipe
is shaped the way it is, and what has to happen upstream before conda-forge can
ship the current version.

That page is the only copy on purpose. An earlier draft of this file explained
the same things again — the blocked dependencies, where to get `sha256`, the two
PyPI-to-conda name differences — and two hand-maintained explanations of one
subject are exactly what drifts apart. Add to the how-to, not here.

```bash
conda build conda-recipe/ -c conda-forge
```
