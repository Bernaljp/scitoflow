"""Sphinx configuration for the scIToFlow documentation.

Heavy runtime dependencies (torch, torch-geometric, torchdiffeq, scanpy, sklearn) are
mocked for the API autodoc so the docs build on a lightweight ReadTheDocs image without a
GPU. Tutorial notebooks are pre-executed and committed; they are not re-run at build time.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -- Project information ------------------------------------------------------------
project = "scIToFlow"
author = "Juan Pablo Bernal-Tamayo"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from importlib.metadata import version as _v
    release = _v("scitoflow")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------------
extensions = [
    "myst_nb",                       # Markdown + executed/rendered notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",           # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",                 # grid cards / tabs on the landing page
    "sphinx_copybutton",             # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary ----------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_mock_imports = [
    "torch", "torchdiffeq", "torch_geometric", "torch_scatter", "torch_sparse",
    "scanpy", "dynamo", "sklearn",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- MyST / notebooks ---------------------------------------------------------------
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3
nb_execution_mode = "off"            # notebooks ship pre-executed; RTD only renders them
nb_merge_streams = True

# -- HTML output (sphinx-book-theme) ------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "scIToFlow"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
        "text": "",
    },
    "repository_url": "https://github.com/Bernaljp/scitoflow",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Bernaljp/scitoflow",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
