"""FreeNeRF for nerfstudio.

Nothing heavy is imported at package import time: the pure-maths modules (mask, losses, geometry, arms, plan)
work without torch or nerfstudio, so they are unit-tested in CI. The nerfstudio-dependent modules are
encoding.py, model.py, config.py and export.py.
"""
__version__ = "0.1.0"
