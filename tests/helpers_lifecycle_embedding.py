"""Stub embedding-worker body for the embedding-pool lifecycle test.

Its own module, not a nested function in the parent script, so the pool can
pickle it by module-qualified reference and the spawned child can import it
without re-entering the parent's ``main()``.
"""

import os
import time
from pathlib import Path


def slow_embed(smi, name, n_conformers, threshold, np_threads):
    """Record this worker's pid, then stay busy long enough to be orphaned.

    Stands in for ``_embed_single``: what the test observes is whether the pool
    worker notices a dead parent, which has nothing to do with ETKDG, and a real
    embedding would either finish before the parent is killed or make the test's
    wall clock depend on RDKit's.
    """
    Path(os.environ["LIFECYCLE_DIR"], f"pool-{os.getpid()}.pid").write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("LIFECYCLE_EMBED_SECONDS", "60")))
    return []
