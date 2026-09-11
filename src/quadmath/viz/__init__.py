"""Visualization layer: plotting primitives and the deterministic galleries.

Collision note: ``vis_lattice`` and ``vis_stats`` both export ``gallery``
(distinct functions) and ``GALLERY_FILES`` (distinct tuples).  Per the
re-export convention they are imported explicitly instead of star-imported;
use the full paths (e.g. ``from quadmath.viz.vis_stats import gallery``).
"""

from .visualize import *

from . import vis_lattice
from . import vis_stats
