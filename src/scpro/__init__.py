"""SCPRO method package.

Public modules
--------------
- :mod:`scpro.hi` for SCPRO-HI horizontal proteomics integration.
- :mod:`scpro.vi` for SCPRO-VI multimodal RNA/protein integration.
"""

from . import hi, vi

__all__ = ["hi", "vi"]
__version__ = "0.1.0"
