"""pytest auto-loaded configuration.

pytest auto-loads this file before any test module imports run, which
makes it the right place to set the ``KMP_DUPLICATE_LIB_OK=TRUE`` env
var.  pytest's own collection machinery imports numpy, which in turn
loads libomp; setting the env var inside ``cnmineral/__init__.py`` is
too late because the test collector pulls numpy in before any
``cnmineral`` import.

Background: macOS's PyTorch and PetThermoTools (via MAGEMin) both link
against libomp.  Concurrent loading aborts with "OpenMP already
initialized" unless this env var is set.  See ``docs/MODEL.md`` macOS
gotcha section.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
