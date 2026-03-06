# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path so Sphinx can find your modules
sys.path.insert(0, os.path.abspath('..'))

project = 'rdmpy'
copyright = '2026, rdmpy Contributors'
author = 'rdmpy Contributors'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that should not be included
# when using the wildcard pattern. We exclude these patterns from being built.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ---
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = 'https://martazarantonello.github.io/rdmpy/'

# -- Extension configuration ---
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
