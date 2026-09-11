"""Inference layer: information geometry and Active Inference primitives.

Note: the active-inference functions (``active_inference_step``,
``action_update``, ``expected_free_energy``, ...) live in
:mod:`quadmath.inference.information`; there is no separate
``active_inference`` module in the repository.
"""

from .information import *
