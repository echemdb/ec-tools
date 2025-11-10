Installation
============

The recommended way to install the `ec-tools` package is to use a separate python environment. There are several ways to setup an envirornment. Since we are using `pixi` to manage the environments we will limit the explanation mostly to this approach here. After installing pixi (see https://pixi.sh/latest) `ec-tools` can be used inside a environment initialized with:

```sh
pixi shell -e opt
```
The `opt` environment includes the `numba` and `pythran` packages to allow the speedup of computationally intensive operations, such as semi integration. However, `ec-tools` runs also without them when `-e opt` is omitted.

`ec-tool` can also be installed by pip:

```sh .noeval
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

If you want to work on the `ec-tools` itself, get a copy of the latest
version of the ec-tools:

```sh .noeval
pixi shell -e dev
```

Any changes you make to the files in your local copy of the ec-tools should
now be available in your next Python session.

### Testing

```sh
pixi run doctest
```

### Documentation

While working on the documentation it is sometimes convenient to have hot reload the documentation in the browser on change, which can be enabled by using `sphinx-autobuild`.

```sh
pixi run doc-watch
```

Open `http://127.0.0.1:8000` in the browser

We would love to see your contribution to ec-tools.
