"""Lattice layer: IVM shell numbering, fields, dynamics, search, conversions.

Collision note: ``omni_numbering`` and ``lattice_search`` both export
``MAX_SHELL`` (the same object re-exported), and ``ivm_field`` and
``ivm_dynamics`` both export ``ball_sites`` (distinct functions).  Per the
re-export convention, the colliding modules are imported explicitly instead
of star-imported; use the full module paths (e.g.
``from quadmath.lattice.ivm_field import ball_sites``).
"""

from .conversions import *

from . import ivm_dynamics
from . import ivm_field
from . import lattice_search
from . import omni_numbering
