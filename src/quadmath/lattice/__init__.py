"""Lattice layer: IVM shell numbering, fields, dynamics, search, conversions.

Collision note: ``omni_numbering`` and ``lattice_search`` both export
``MAX_SHELL`` (the same object re-exported), and ``ivm_dynamics.ball_sites``
(void-filtered ball) and ``ivm_field.shell_ball_sites`` (union of shells) are
distinct functions under distinct names.  Import the modules explicitly rather
than star-importing them.
"""

from .conversions import *  # noqa: F403

from . import ivm_dynamics as ivm_dynamics
from . import ivm_field as ivm_field
from . import lattice_search as lattice_search
from . import omni_numbering as omni_numbering
