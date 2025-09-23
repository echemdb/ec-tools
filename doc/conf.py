project = "ec-tools"
copyright = "2023-2025, the ec-tools authors"
author = "the ec-tools authors"

release = "0.0.1"

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "myst_nb",
    "sphinxcontrib.katex",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]

bibtex_default_style = "plain"
bibtex_reference_style = "label"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

templates_path = ["_templates"]

exclude_patterns = [
    "generated",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "news",
    ".ipynb_checkpoints",
    "*.ipynb",
]

todo_include_todos = True

html_theme = "sphinx_rtd_theme"

nb_execution_timeout = 90

html_static_path = []

# Add Edit on GitHub links
html_context = {
    "display_github": True,
    "github_user": "echemdb",
    "github_repo": "ec-tools",
    "github_version": "main/doc/",
}
