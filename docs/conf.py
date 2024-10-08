# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../src/cvanmf'))

# -- Project information -----------------------------------------------------

project = u"cvanmf"
copyright = u"2023, Anthony Duncan"
author = u"Anthony Duncan"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_click"
]
autoapi_dirs = ["../src/cvanmf"]
autoapi_options = ['members', 'undoc-members', 'show-inheritance',
                   'show-module-summary', 'imported-members']
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'both'
autoapi_template_dir = 'autoapi_templates'

# Permit long running cells in example notebooks
nbsphinx_timeout = 360

nb_execution_excludepatterns = ["cell_example*"]

def skip_utils(app, what, name, obj, skip, options):
    if (
            "RE_" in name or
            "CVANMFException" in name or
            'DEF_' in name or
            "cli" in name[:4]
    ):
       skip = True
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_utils)


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Parser extension for myst (notebook parsing), I'm using this to enable math
myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]
