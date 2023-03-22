Installation
============

The recommended way to install the ec-tools package is to use your package manager,
(e.g., `apt-get` on Debian or Ubuntu, `pacman` on Arch Linux, `brew` on macOS.)

This can be done by:

```sh
pip install ec-tools
```

This command downloads and installs the ec-tools and its dependencies into
your local Python installation.

If the above command fails because you do not have permission to modify your
Python installation, you can install the ec-tools into your user account:

```sh
pip install --user ec-tools
```

You can instead also install the latest unreleased version of the ec-tools
from our [GitHub Repository](https://github.com/echemdb/ec-tools) with

```sh
pip install git+https://github.com/echemdb/ec-tools@main
```


Install with pip for development
--------------------------------

If you want to work on the ec-tools itself, get a copy of the latest
version of the ec-tools:

```sh
git clone https://github.com/echemdb/ec-tools.git
```

Create an [editable](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs) install of the ec-tools:

```sh
pip install -e .
```

Any changes you make to the files in your local copy of the ec-tools should
now be available in your next Python session. 

We would love to see your contribution to the ec-tools.
