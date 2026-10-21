"""phasecast (package `cnmineral`).

Importing this package sets ``KMP_DUPLICATE_LIB_OK=TRUE`` if it is not
already in the environment.  This is the standard workaround for the
libomp collision between
PyTorch and PetThermoTools / MAGEMin on macOS.  Doing it here -- before
any submodule that might import torch loads -- means downstream
scripts and notebooks never have to remember.
"""

import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
