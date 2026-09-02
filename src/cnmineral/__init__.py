"""CrabNet-Mineral v3.

Importing this package sets `KMP_DUPLICATE_LIB_OK=TRUE` if it is not
already in the environment.  Several packages in the env ship their own
OpenMP runtime (torch bundles a libomp, conda's llvm-openmp is another,
xgboost links its own) and loading two in one process aborts on macOS
(`CLAUDE.md` macOS gotcha).  Doing it here -- before any submodule that
might import torch loads -- means downstream scripts and notebooks never
have to remember.
"""

import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
