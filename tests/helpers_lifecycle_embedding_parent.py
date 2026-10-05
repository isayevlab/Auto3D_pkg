"""Run a real embedding pool with a stub worker body. Launched as a subprocess."""

import os
import sys
from pathlib import Path

import Auto3D.domain.embedding as emb
from tests import helpers_lifecycle_embedding as hle

# Module scope, not inside main(): under the spawn context every pool child
# re-imports this script as __mp_main__, so the substitution has to be in place
# there too. `embed_conformers_parallel` submits the module global by reference,
# which pickles as `tests.helpers_lifecycle_embedding.slow_embed` -- importable
# in the child because `_launch_embedding_pool` puts the repo root on PYTHONPATH.
emb._embed_single = hle.slow_embed


def main() -> int:
    Path(os.environ["LIFECYCLE_DIR"], "parent.pid").write_text(str(os.getpid()))
    # Two species, two workers: every worker gets a task and so writes its pid
    # down before the test signals this process.
    list(
        emb.embed_conformers_parallel(
            [("CCO", "a"), ("CCC", "b")],
            n_conformers=1,
            n_workers=2,
        )
    )
    return 0


if __name__ == "__main__":
    # REQUIRED: under spawn every child re-imports this script, and without the
    # guard each pool worker would start a pool of its own.
    sys.exit(main())
