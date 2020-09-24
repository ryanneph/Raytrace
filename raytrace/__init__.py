import logging

try:
    from .raytrace_ext import raytrace, beamtrace
except ImportError:
    from .raytracers import raytrace, beamtrace

from .raytracers import siddonraytracer, NonIntersectingRayError, spottrace

def enableDebugOutput():
    """Setup the library logger from a user application"""
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(name)s (%(levelname)s) | %(message)s"))
    module_logger.addHandler(sh)
