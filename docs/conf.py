"""Sphinx configuration for metabo-depthcharge."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


project = "metabo-depthcharge"
author = "Bittremieux Lab"
copyright = "2026, Bittremieux Lab"

try:
    release = _pkg_version("metabo-depthcharge")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
    "member-order": "bysource",
}
autodoc_member_order = "bysource"
autosummary_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{project} {version}"
