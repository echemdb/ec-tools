---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


Welcome to the ec-tools documentation !
========================================

The `ec-tools` library is a collection of tools which can be useful when working with electrochemical data such as cyclic voltammetry. One non-trivial example is the implementation of algorithms to compute the semi integrals which eventually allows for the elegant analysis of diffusion controlled electrochemical process such as electron transfer reactions from cyclic voltammetry data. 

Installation
============

The package can be installed via

```sh .noeval
pip install git+https://github.com/echemdb/ec-tools
```

+++

Read the [installation instructions](installation.md) on further details if you want to contribute to the project.


```{toctree}
:maxdepth: 2
:caption: "Contents:"
:hidden:
installation.md
semiint.md
```
