"""ec-tools: tools to work with electrochemical data."""

import importlib.util as _importlib_util
import os as _os
from pathlib import Path as _Path

# The pythran backend needs a BLAS to compile np.convolve (used by the Riemann
# semi-integration core). On conda this is provided by a system BLAS (blas-devel)
# and pythran's default config links it. With a pip install of the "pythran" extra
# there is no system BLAS, but scipy-openblas64 ships its own; pythran only links it
# when its config points at it, so in that case we set PYTHRANRC to the bundled
# config. A user-provided PYTHRANRC always takes precedence, and on conda (no
# scipy-openblas64) we leave the default config alone.
if "PYTHRANRC" not in _os.environ and _importlib_util.find_spec("scipy_openblas64") is not None:
    _os.environ["PYTHRANRC"] = str(_Path(__file__).parent / "pythran.cfg")
