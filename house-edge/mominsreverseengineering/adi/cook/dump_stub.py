"""dump.py, with an empty numpy stub so method4_ml.model imports without numpy (v5 tree path never calls numpy)."""
import sys, types, runpy
try:
    import numpy  # noqa
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")
runpy.run_module("adi.cook.dump", run_name="__main__")
