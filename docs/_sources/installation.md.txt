Installation
============

The recommended way to install the `ec-tools` package is to use a separate python environment. There are several ways to setup an envirornment. Since we are using `conda`/`mamba` to manage the environments we will limit the explanation to this approach here. After installing mambaforge (see https://mamba.readthedocs.io/en/latest/index.html) a conda environment named `ec-tools` can be created by:

```sh
mamba create -c conda-forge -n ec-tools python=3.11 pip numba pythran
```
The `numba` and `pythran` are installed to allow the speedup of computationally intensive operations, such as semi integration. However, `ec-tools` runs also without them.

`ec-tool` can be installed in the new created environment by:

```sh .noeval
mamba activate ec-tools
pip install git+https://github.com/echemdb/ec-tools
```

This command downloads and installs `ec-tools` (from github) and its dependencies into
your local Python installation.

If the above command fails because you do not have permission to modify your
Python installation, you can install the ec-tools into your user account:

```sh .noeval
pip install --user git+https://github.com/echemdb/ec-tools
```

<!-- You can instead also install the latest unreleased version of the ec-tools
from our [GitHub Repository](https://github.com/echemdb/ec-tools) with

```sh
pip install git+https://github.com/echemdb/ec-tools@main
``` -->



Install for development
--------------------------------

If you want to work on the ec-tools itself, get a copy of the latest
version of the ec-tools:

```sh .noeval
git clone https://github.com/echemdb/ec-tools.git
```

Create the development environment from the provided `environment.yml`:
```sh .noeval
cd ec-tools
mamba create -f environment.yml
```

Install `ec-tools` in editable mode [editable](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs):

```sh
mamba activate ec-tools-dev
pip install -e .
```

Any changes you make to the files in your local copy of the ec-tools should
now be available in your next Python session.

### Testing

```sh
pytest --doctest-modules ec_tools
```

### Documentation

While working on the documentation it is sometimes convenient to have hot reload the documentation in the browser on change, which can be enabled by using `sphinx-autobuild`.

```sh
sphinx-autobuild doc doc/generated/html --watch "ec_tools/*"
```

Open `http://127.0.0.1:8000` in the browser

We would love to see your contribution to ec-tools.
