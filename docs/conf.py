"""Sphinx configuration for qb-compiler documentation."""

project = "qb-compiler"
copyright = "2026, QubitBoost (https://www.qubitboost.io)"  # noqa: A001
author = "Mike"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
}
