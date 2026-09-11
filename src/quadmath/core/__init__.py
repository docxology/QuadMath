"""Core layer: quadray coordinates, exact integer linear algebra, volumes,
geometry, metrics, symbolic helpers, and worked examples.

Star re-exports of every core module; no cross-module name collisions exist
in this layer (checked against each module's public names).
"""

from .linalg_utils import *
from .quadray import *
from .cayley_menger import *
from .geometry import *
from .metrics import *
from .symbolic import *
from .examples import *
