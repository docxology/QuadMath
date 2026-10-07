"""Core layer: quadray coordinates, exact integer linear algebra, volumes,
geometry, metrics, symbolic helpers, and worked examples.

Star re-exports of every core module; no cross-module name collisions exist
in this layer (checked against each module's public names).
"""

from .linalg_utils import *  # noqa: F403
from .quadray import *  # noqa: F403
from .cayley_menger import *  # noqa: F403
from .geometry import *  # noqa: F403
from .metrics import *  # noqa: F403
from .symbolic import *  # noqa: F403
from .examples import *  # noqa: F403
