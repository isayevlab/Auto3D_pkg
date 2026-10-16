# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add src to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Auto3D'
copyright = 'IsayevLab'
author = 'Zhen Liu; Tetiana Zubatiuk; Adrian Roitberg; Olexandr Isayev'

# Dynamic version from package
try:
    from importlib.metadata import version as get_version
    release = get_version('Auto3D')
except Exception:
    release = 'dev'

version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

# External pages that rot. The weekly link check is allowed to be red; a pattern
# goes here with the reason, never a suppression in the workflow.
linkcheck_ignore: list[str] = []

templates_path = ['_templates']

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}

# Sphinx writes one stub per autosummary entry into `source/generated/` on every
# build, and `autosummary_generate_overwrite` (True by default) means it rewrites
# any stub already sitting there. That holds under any Sphinx invocation -- the
# documentation workflow's `make -C docs html` and Read the Docs' own
# `sphinx-build`, which it runs itself against the configuration named in
# `.readthedocs.yaml` -- and both start from a fresh checkout, so the stubs are
# always the ones this configuration just generated and a checked-in copy is
# never read. `source/generated/` is therefore build output and is ignored
# rather than tracked; the stubs were removed from version control on
# 2026-10-16, when seven of the tracked copies were stale and three entries had
# no tracked copy at all.
autosummary_generate = True

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# nbsphinx settings - don't execute notebooks during build
nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'
