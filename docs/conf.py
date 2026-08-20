"""Sphinx configuration for qb-compiler documentation."""

project = "qb-compiler"
copyright = "2026, QubitBoost (https://www.qubitboost.io)"
author = "Michael William Perry Walshe"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "commercial", "commercial/**"]

html_theme = "qiskit-ecosystem"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
}
