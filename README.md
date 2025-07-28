# ec-tools

The `ec-tools` library is a collection of tools which can be useful when working with electrochemical data such as cyclic voltammetry. One non-trivial example is the implementation of algorithms to compute the semi integrals which eventually allows for the elegant analysis of diffusion controlled electrochemical process such as electron transfer reactions from cyclic voltammetry data. 

Installation
============

The package can be installed via

```sh
pip install git+https://github.com/echemdb/ec-tools
```
or with a new separate pixi environment
```sh
git clone https://github.com/echemdb/ec-tools
cd ec-tools
pixi shell
```

Overwiew
============
* [installation](installation.md) chapter provides further details about the installation.
* [Semi integration](semiint.md) gives information about the implemented semi integration algorithms, 
including a fundamental description, how-to call the implemented algorithms and a variety of tests.
