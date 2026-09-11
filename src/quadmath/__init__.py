"""QuadMath: quadray coordinates, optimization on tetrahedral lattices, and
information geometry.

The package layers mirror the repository layout:

- ``quadmath.paths``      — output-directory helpers.
- ``quadmath.core``       — quadray coordinates, exact integer linear algebra,
  Cayley-Menger volumes, special relativity helpers, information metrics,
  symbolic helpers, worked examples.
- ``quadmath.lattice``    — IVM shell numbering, field learning, dynamics,
  nearest-site search, XYZ/IVM conversions.
- ``quadmath.optimize``   — Nelder-Mead on the quadray lattice, discrete
  variational descent.
- ``quadmath.inference``  — information geometry and Active Inference.
- ``quadmath.stats``      — deterministic statistics and benchmarking.
- ``quadmath.learn``      — train/test methodology for the lattice learners.
- ``quadmath.viz``        — plotting primitives and deterministic galleries.
- ``quadmath.pipeline``   — typed composable pipeline layer.
- ``quadmath.tools``      — auto-documentation utilities.

This ``__init__`` re-exports the historical top-level public API so that
``from quadmath import X`` keeps working after the flat ``src/`` modules
moved into the subpackages.  Names that collide across modules (``gallery``
and ``GALLERY_FILES`` in ``vis_lattice`` vs ``vis_stats``; ``ball_sites`` in
``ivm_field`` vs ``ivm_dynamics``) are NOT re-exported here; import those
from their concrete modules (e.g. ``from quadmath.viz.vis_stats import
gallery``).
"""

from quadmath.paths import get_repo_root, get_output_dir, get_data_dir, get_figure_dir

# core
from quadmath.core.quadray import (
    Quadray,
    to_xyz,
    integer_tetra_volume,
    ace_tetravolume_5x5,
    DEFAULT_EMBEDDING,
    magnitude,
    dot,
    distance,
    angle,
    centroid,
    quadray_from_xyz,
    qmul,
    qconjugate,
    qrotate,
    slerp,
    rotate_about_axis,
)
from quadmath.core.linalg_utils import (
    bareiss_determinant_int,
    bareiss_rank,
    integer_adjugate,
)
from quadmath.core.cayley_menger import (
    tetra_volume_cayley_menger,
    ivm_tetra_volume_cayley_menger,
    squared_distances_from_quadrays,
    tetra_circumradius,
    tetra_inradius,
)
from quadmath.core.geometry import (
    minkowski_interval,
    lorentz_factor,
    proper_time,
    spacetime_classify,
)
from quadmath.core.metrics import (
    shannon_entropy,
    information_length,
    fim_eigenspectrum,
    fisher_condition_number,
    fisher_curvature_analysis,
    fisher_quadray_comparison,
    kl_divergence,
    jensen_shannon_divergence,
    fisher_rao_metric,
    angle_error,
    quat_log_euclidean_dispersion,
)
from quadmath.core.symbolic import (
    cayley_menger_volume_symbolic,
    convert_xyz_volume_to_ivm_symbolic,
)
from quadmath.core.examples import (
    example_ivm_neighbors,
    example_volume,
    example_optimize,
    example_cuboctahedron_neighbors,
    example_cuboctahedron_vertices_xyz,
    example_partition_tetra_volume,
)

# lattice
from quadmath.lattice.omni_numbering import (
    NEIGHBOR_MOVES,
    MAX_SHELL,
    shell_count,
    cumulative_count,
    generate_shell,
    sites_through_shell,
    site_index,
    site_at_index,
)
from quadmath.lattice.ivm_field import (
    IVM_NEIGHBOR_STEPS,
    IVMField,
    TetrahedronFit,
    ball_sites,
    fit_geometry,
    is_ivm_site,
    quadray_shell_norm,
    shell_cardinalities,
    shell_sites,
)
from quadmath.lattice.ivm_dynamics import (
    DynamicsParams,
    FitResult,
    IVMLattice,
    Trajectory,
    fit_trajectory,
    heat_step,
    is_nonincreasing,
    make_lattice,
    majority_step,
    neighbor_shifts,
    render_dynamics_demo,
    simulate,
    site_radius_sq,
    step,
    sum_of_squares,
)
from quadmath.lattice.lattice_search import (
    squared_distance,
    nearest,
    within_radius,
)
from quadmath.lattice.conversions import (
    urner_embedding,
    quadray_to_xyz,
    xyz_to_quadray_canonical,
    quadray_roundtrip,
    embedding_basis,
)

# optimize
from quadmath.optimize.nelder_mead_quadray import (
    SimplexState,
    order_simplex,
    centroid_excluding,
    project_to_lattice,
    compute_volume,
    nelder_mead_quadray,
)
from quadmath.optimize.discrete_variational import (
    neighbor_moves_ivm,
    apply_move,
    DiscretePath,
    discrete_ivm_descent,
)

# inference
from quadmath.inference.information import (
    fisher_information_matrix,
    fisher_information_quadray,
    natural_gradient_step,
    free_energy,
    finite_difference_gradient,
    perception_update,
    action_update,
    expected_free_energy,
    active_inference_step,
    information_geometric_distance,
    mutual_information,
    information_gain,
)

# stats
from quadmath.stats.statistics import (
    bootstrap_ci,
    cohens_d,
    p_adjust_bonferroni,
    permutation_test,
    scaling_fit,
    summarize,
    jackknife_ci,
    benjamini_hochberg,
    welch_t_test,
    rotation_stats,
)
from quadmath.stats.benchmarks import (
    BENCH_DEFAULTS,
    BenchRow,
    time_callable,
    bench_conversions,
    bench_shell_enumeration,
    bench_lattice_search,
    bench_field_fit,
    summary_table,
    run_all,
)

# learn
from quadmath.learn.learning_eval import (
    CrossValidationResult,
    LearningCurveResult,
    TrajectorySplitResult,
    cross_validate_field,
    enclosing_radius,
    kfold_site_splits,
    learning_curve,
    trajectory_train_test,
    three_way_split,
    RidgeSiteFit,
    ridge_site_fit,
    GradientDescentTrainer,
)

# viz
from quadmath.viz.visualize import (
    plot_ivm_neighbors,
    animate_simplex,
    plot_simplex_trace,
    plot_partition_tetrahedron,
    animate_discrete_path,
)
from quadmath.viz.vis_lattice import (
    DEFAULT_PLANE,
    dynamics_strip,
    field_slice,
    shell_scatter,
    SiteLike,
)
from quadmath.viz.vis_stats import (
    plot_ci_bars,
    plot_ecdf,
    plot_latency_hist,
    plot_scaling_loglog,
)
from quadmath.viz.animations import (
    Frame,
    GRID_SIZE,
    diffusion_frames,
    frames_to_gif,
    lattice_frames,
    simplex_frames,
)
from quadmath.viz.plots import (
    plot_error_histogram,
    plot_loss_history,
    plot_shell_growth,
    plot_lattice_shell_3d,
)

# validate
from quadmath.validate import (
    ValidationReport,
    check_associativity,
    check_conjugate_inverse,
    check_double_cover,
    check_normalization,
    check_slerp_midpoint,
    run_validation,
)

# pipeline
from quadmath.pipeline import (
    FieldLearner,
    FieldModel,
    Fittable,
    LatticeBall,
    LatticeSource,
    Pipeline,
    Step,
    dynamics_step,
    learn_step,
    sites_step,
)

# tools
from quadmath.tools.glossary_gen import (
    ApiEntry,
    build_api_index,
    generate_markdown_table,
    inject_between_markers,
)

__all__ = [
    # paths
    "get_repo_root",
    "get_output_dir",
    "get_data_dir",
    "get_figure_dir",
    # core.quadray
    "Quadray",
    "to_xyz",
    "integer_tetra_volume",
    "ace_tetravolume_5x5",
    "DEFAULT_EMBEDDING",
    "magnitude",
    "dot",
    "distance",
    "angle",
    "centroid",
    "quadray_from_xyz",
    "qmul",
    "qconjugate",
    "qrotate",
    "slerp",
    "rotate_about_axis",
    # core.linalg_utils
    "bareiss_determinant_int",
    "bareiss_rank",
    "integer_adjugate",
    # core.cayley_menger
    "tetra_volume_cayley_menger",
    "ivm_tetra_volume_cayley_menger",
    "squared_distances_from_quadrays",
    "tetra_circumradius",
    "tetra_inradius",
    # core.geometry
    "minkowski_interval",
    "lorentz_factor",
    "proper_time",
    "spacetime_classify",
    # core.metrics
    "shannon_entropy",
    "information_length",
    "fim_eigenspectrum",
    "fisher_condition_number",
    "fisher_curvature_analysis",
    "fisher_quadray_comparison",
    "kl_divergence",
    "jensen_shannon_divergence",
    "fisher_rao_metric",
    "angle_error",
    "quat_log_euclidean_dispersion",
    # core.symbolic
    "cayley_menger_volume_symbolic",
    "convert_xyz_volume_to_ivm_symbolic",
    # core.examples
    "example_ivm_neighbors",
    "example_volume",
    "example_optimize",
    "example_cuboctahedron_neighbors",
    "example_cuboctahedron_vertices_xyz",
    "example_partition_tetra_volume",
    # lattice.omni_numbering
    "NEIGHBOR_MOVES",
    "MAX_SHELL",
    "shell_count",
    "cumulative_count",
    "generate_shell",
    "sites_through_shell",
    "site_index",
    "site_at_index",
    # lattice.ivm_field
    "IVM_NEIGHBOR_STEPS",
    "IVMField",
    "TetrahedronFit",
    "ball_sites",
    "fit_geometry",
    "is_ivm_site",
    "quadray_shell_norm",
    "shell_cardinalities",
    "shell_sites",
    # lattice.ivm_dynamics
    "DynamicsParams",
    "FitResult",
    "IVMLattice",
    "Trajectory",
    "fit_trajectory",
    "heat_step",
    "is_nonincreasing",
    "make_lattice",
    "majority_step",
    "neighbor_shifts",
    "render_dynamics_demo",
    "simulate",
    "site_radius_sq",
    "step",
    "sum_of_squares",
    # lattice.lattice_search
    "squared_distance",
    "nearest",
    "within_radius",
    # lattice.conversions
    "urner_embedding",
    "quadray_to_xyz",
    "xyz_to_quadray_canonical",
    "quadray_roundtrip",
    "embedding_basis",
    # optimize.nelder_mead_quadray
    "SimplexState",
    "order_simplex",
    "centroid_excluding",
    "project_to_lattice",
    "compute_volume",
    "nelder_mead_quadray",
    # optimize.discrete_variational
    "neighbor_moves_ivm",
    "apply_move",
    "DiscretePath",
    "discrete_ivm_descent",
    # inference.information
    "fisher_information_matrix",
    "fisher_information_quadray",
    "natural_gradient_step",
    "free_energy",
    "finite_difference_gradient",
    "perception_update",
    "action_update",
    "expected_free_energy",
    "active_inference_step",
    "information_geometric_distance",
    "mutual_information",
    "information_gain",
    # stats.statistics
    "bootstrap_ci",
    "cohens_d",
    "p_adjust_bonferroni",
    "permutation_test",
    "scaling_fit",
    "summarize",
    "jackknife_ci",
    "benjamini_hochberg",
    "welch_t_test",
    "rotation_stats",
    # stats.benchmarks
    "BENCH_DEFAULTS",
    "BenchRow",
    "time_callable",
    "bench_conversions",
    "bench_shell_enumeration",
    "bench_lattice_search",
    "bench_field_fit",
    "summary_table",
    "run_all",
    # learn.learning_eval
    "CrossValidationResult",
    "LearningCurveResult",
    "TrajectorySplitResult",
    "cross_validate_field",
    "enclosing_radius",
    "kfold_site_splits",
    "learning_curve",
    "trajectory_train_test",
    "three_way_split",
    "RidgeSiteFit",
    "ridge_site_fit",
    "GradientDescentTrainer",
    # viz.visualize
    "plot_ivm_neighbors",
    "animate_simplex",
    "plot_simplex_trace",
    "plot_partition_tetrahedron",
    "animate_discrete_path",
    # viz.vis_lattice
    "DEFAULT_PLANE",
    "dynamics_strip",
    "field_slice",
    "SiteLike",
    "shell_scatter",
    # viz.vis_stats
    "plot_ci_bars",
    "plot_ecdf",
    "plot_latency_hist",
    "plot_scaling_loglog",
    # viz.animations
    "Frame",
    "GRID_SIZE",
    "diffusion_frames",
    "frames_to_gif",
    "lattice_frames",
    "simplex_frames",
    # viz.plots
    "plot_error_histogram",
    "plot_loss_history",
    "plot_shell_growth",
    "plot_lattice_shell_3d",
    # validate
    "ValidationReport",
    "check_associativity",
    "check_conjugate_inverse",
    "check_double_cover",
    "check_normalization",
    "check_slerp_midpoint",
    "run_validation",
    # pipeline
    "FieldLearner",
    "FieldModel",
    "Fittable",
    "LatticeBall",
    "LatticeSource",
    "Pipeline",
    "Step",
    "dynamics_step",
    "learn_step",
    "sites_step",
    # tools.glossary_gen
    "ApiEntry",
    "build_api_index",
    "generate_markdown_table",
    "inject_between_markers",
]