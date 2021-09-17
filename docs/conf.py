"""Sphinx configuration."""
import time

project = "pyffstream"
author = "Gregory Beauregard"
copyright = f"2021-{time.strftime('%Y')}, {author}"
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autoprogram",
]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
always_document_param_types = True
