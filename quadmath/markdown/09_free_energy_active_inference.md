# Appendix: The Free Energy Principle and Active Inference

## Overview

The Free Energy Principle (FEP) posits that biological systems maintain their states by minimizing variational free energy, thereby reducing surprise via prediction and model updating. Active Inference extends this by casting action selection as inference under prior preferences. Background: see the concise overview on the [Free energy principle](https://en.wikipedia.org/wiki/Free_energy_principle) and the monograph [Active Inference (MIT Press)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind).

This appendix emphasizes relationships among: (i) the four-fold partition of Active Inference, (ii) Quadrays (Fuller.4D) as a geometric scaffold for mapping this partition, and (iii) information-geometric flows (Einstein.4D analogy) that underpin perception–action updates. For the naming of 4D namespaces used throughout—Coxeter.4D (Euclidean E4), Einstein.4D (Minkowski spacetime analogy), Fuller.4D (Synergetics/Quadrays)—see `02_4d_namespaces.md`.

## Mathematical Formulation and Equation Callouts (Equations linkage)

- Variational free energy (discrete states) — see Eq. \eqref{eq:supp_free_energy} in the equations appendix, implemented by [`free_energy`](08_equations_appendix.md#code:free_energy):

  \begin{equation}\label{eq:free_energy_appendix_ref}
  \mathcal{F} = -\log P(o\mid s) + \mathrm{KL}\big[ Q(s)\;\|\; P(s) \big]
  \end{equation}

  where \(Q(s)\) is a variational posterior, \(P(s)\) a prior, and \(P(o\mid s)\) the likelihood. Lower \(\mathcal{F}\) is better.

- Fisher Information Matrix (FIM) as metric — see Eq. \eqref{eq:supp_fim} in the equations appendix and [`fisher_information_matrix`](08_equations_appendix.md#code:fisher_information_matrix):

  \begin{equation}\label{eq:fim_definition}
  F_{i,j} = \mathbb{E}\left[ \partial_{\theta_i} \log p(x;\theta)\; \partial_{\theta_j} \log p(x;\theta) \right].
  \end{equation}

- Natural gradient descent under information geometry — see Eq. \eqref{eq:supp_natgrad} in the equations appendix and [`natural_gradient_step`](08_equations_appendix.md#code:natural_gradient_step); overview: [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient):

  \begin{equation}\label{eq:natural_gradient_update}
  \theta \leftarrow \theta - \eta\, F(\theta)^{-1}\, \nabla_{\theta} L(\theta).
  \end{equation}

Figures: See the enhanced Active Inference figures in the optimization section above, which demonstrate the integration of natural gradient descent with Active Inference principles and the 4D framework context.

Discrete variational optimization on the quadray lattice: `discrete_ivm_descent` greedily descends a free-energy-like objective over IVM moves, yielding integer-valued trajectories. See the path animation artifact `discrete_path.mp4` in `quadmath/output/`.

![**Active Inference four-fold partition mapped to a Quadray tetrahedron in Fuller.4D**. This 3D tetrahedral visualization demonstrates the geometric embedding of Active Inference's fundamental four-fold partition within the Quadray coordinate system. **Tetrahedral structure**: The four vertices of the regular tetrahedron represent the four components of the Active Inference framework: perception, action, internal states, and external states. **Partition mapping**: Each face of the tetrahedron corresponds to a specific partition of the four-fold system, with the edges representing the relationships and interactions between different components. **Fuller.4D significance**: This geometric representation leverages the tetrahedral nature of Quadray coordinates to provide an intuitive visualization of the Active Inference framework's structure. The tetrahedron serves as a natural container for the four-fold partition, emphasizing the interconnected nature of perception, action, and state representation in active inference. **Optimization context**: The tetrahedral geometry also suggests natural optimization strategies that respect the four-fold structure, potentially leading to more efficient inference algorithms that leverage the geometric relationships between different components. This visualization demonstrates how the Fuller.4D framework can provide insights into complex systems like Active Inference through geometric intuition.](../output/figures/partition_tetrahedron.png)

![**4D Natural Gradient Trajectory in Active Inference Dynamics**. This comprehensive visualization demonstrates the evolution of Active Inference parameters through natural gradient descent on an information manifold. **Trajectory visualization**: The 3D plot shows the optimization path through perception weight (μ), action weight (a), and internal state (s) dimensions, with color coding indicating progression over time. **Free energy minimization**: The log-scale plot shows the systematic reduction of free energy, demonstrating the Active Inference principle of surprise minimization. **Four-fold partition evolution**: Parameter trajectories show how perception, action, internal, and external components evolve toward optimal values, with horizontal lines indicating true optimal states. **Information geometry context**: The Fisher Information Matrix provides the Riemannian metric for geodesic motion, connecting to Einstein.4D concepts where the Fisher metric replaces spacetime geometry. **4D framework integration**: The visualization demonstrates how Coxeter.4D (Euclidean), Einstein.4D (Minkowski), and Fuller.4D (Synergetics) frameworks work together in Active Inference optimization.](../output/figures/enhanced_figure_13_4d_trajectory.png)

![**Enhanced Free Energy Landscape in 4D Active Inference Framework**. This comprehensive visualization explores the variational free energy surface over perception and action parameters. **3D landscape**: The surface plot shows the free energy as a function of two variational parameters, revealing the complex topology that Active Inference optimization must navigate. **Contour analysis**: 2D contours provide detailed information about parameter sensitivity and optimization paths. **Cross-sectional analysis**: Multiple cross-sections at different parameter values demonstrate how free energy varies with respect to individual parameters, revealing the landscape's structure. **Four-fold partition visualization**: The text panel explains how Active Inference maps to tetrahedral structures in Fuller.4D, with the four components (μ, s, a, ψ) representing internal states, sensory observations, actions, and external causes. **Information geometry metrics**: Local curvature analysis reveals the Fisher information structure, showing how the information manifold's geometry influences optimization dynamics. **Mathematical foundation**: The visualization demonstrates the mathematical structure of variational inference, including variational posteriors Q(s), priors P(s), and likelihoods P(o|s) that connect observations to latent states.](../output/figures/enhanced_figure_14_free_energy_landscape.png)

## Four-Fold Partition and Tetrahedral Mapping (Quadrays; Fuller.4D)

Active Inference partitions the agent–environment system into four coupled states:

- Internal (\(\mu\)) — agent's internal states
- Sensory (\(s\)) — observations
- Active (\(a\)) — actions
- External (\(\psi\)) — latent environmental causes

See, for an overview of this partition and generative process formulations, the [Active Inference review](https://discovery.ucl.ac.uk/id/eprint/10176959/1/1-s2.0-S1571064523001094-main.pdf) and the general entry on [Active inference](https://en.wikipedia.org/wiki/Active_inference).

Tetrahedral mapping via Quadrays (Fuller.4D): assign each state to a vertex of a tetrahedron, using Quadray coordinates `(A,B,C,D)` with non-negative components and at least one zero after normalization. One canonical mapping is `A \leftrightarrow Internal (\mu)`, `B \leftrightarrow Sensory (s)`, `C \leftrightarrow Active (a)`, `D \leftrightarrow External (\psi)`. The edges capture the pairwise couplings (e.g., `\mu\text{--}s` for perceptual inference; `a\text{--}\psi` for control). Integer tetravolume then quantifies the “coupled capacity” region spanned by jointly feasible states in a time slice; see `Quadray` and tetravolume methods in `03_quadray_methods.md`.

Interpretation note: this Quadray-based mapping is a didactic geometric scaffold. It is not standard in the Active Inference literature, which typically develops the four-state partition in probabilistic graphical terms. Our use highlights structural symmetries and discrete volumetric quantities available in Fuller.4D, building on the computational foundations developed in the [4dsolutions ecosystem](https://github.com/4dsolutions) for tetrahedral modeling and volume calculations. See the [Resources](07_resources.md) section for comprehensive details on the computational implementations.

Code linkage (no snippet): see `example_partition_tetra_volume` in `src/examples.py` and the partition tetrahedron figure above.

## How the 4D namespaces relate here

- Fuller.4D (Quadrays): geometric embedding of the four-state partition on a tetrahedron; integer tetravolumes and IVM moves provide discrete combinatorial structure.
- Coxeter.4D (Euclidean E4): exact Euclidean measurements (e.g., Cayley–Menger determinants) for tetrahedra underlying volumetric comparisons and scale relations.
- Einstein.4D (Minkowski analogy): information-geometric flows (natural gradient, metric-aware updates) supply a continuum picture for perception–action dynamics.

The three roles are complementary: Fuller.4D encodes partition structure, Coxeter.4D provides exact metric geometry for static comparisons, and Einstein.4D guides dynamical descent.

## Joint Optimization in the Tetrahedral Framework (Methods linkage)

- Perception: update \(\mu\) to minimize prediction error on \(s\) under the generative model (descending \(\nabla_{\mu} F\)).
- Action: select \(a\) that steers \(\psi\) toward preferred outcomes (descending \(\nabla_{a} F\)).

Continuous-time flows (Einstein.4D analogy for metric/geodesic intuition): see `perception_update` and `action_update` in `src/information.py`. Discrete Quadray moves connect to these flows via greedy descent on a local free-energy-like objective; see `discrete_ivm_descent` in `src/discrete_variational.py` and the path artifacts in `quadmath/output/`.

## Neuroscience and Predictive Coding

Under Active Inference, cortical circuits minimize free energy through recurrent exchanges of descending predictions and ascending prediction errors, aligning with predictive coding accounts. See the neural dynamics framing in [Active Inference neural dynamics (arXiv:2001.08028)](https://arxiv.org/abs/2001.08028).

## Relation to Reinforcement Learning and Control

Active Inference replaces explicit value functions with prior preferences over outcomes and transitions, balancing exploration (epistemic value) and exploitation (pragmatic value) via expected free energy. See [Active Inference and RL (arXiv:2002.12636)](https://arxiv.org/abs/2002.12636). Connections to optimal control arise when minimizing expected free energy plays the role of a control objective; cf. [Optimal control](https://en.wikipedia.org/wiki/Optimal_control).

## Links to Other Theories

- Bayesian Brain hypothesis: [Bayesian brain](https://en.wikipedia.org/wiki/Bayesian_brain)
- Predictive Coding: [Predictive coding](https://en.wikipedia.org/wiki/Predictive_coding)
- Information Geometry: [Fisher information](https://en.wikipedia.org/wiki/Fisher_information), [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient)

## Implications for AI and Robust Computation

FEP/Active Inference provide algorithms that unify perception and action under uncertainty, offering biologically plausible alternatives to standard RL with adaptive exploration and robust decision-making. See [applications in AI (arXiv:1907.03876)](https://arxiv.org/abs/1907.03876).

## Code, Reproducibility, and Cross-References

– Equation references: [Eq. (Free Energy)](08_equations_appendix.md#eq:free_energy), [Eq. (FIM)](08_equations_appendix.md#eq:fim), [Eq. (Natural Gradient)](08_equations_appendix.md#eq:natgrad) in `08_equations_appendix.md`.
– Code anchors (for readers who want to run experiments): [`free_energy`](03_quadray_methods.md#code:free_energy), [`fisher_information_matrix`](03_quadray_methods.md#code:fisher_information_matrix), [`natural_gradient_step`](03_quadray_methods.md#code:natural_gradient_step), `perception_update`, `action_update`, and `discrete_ivm_descent` in `src/information.py` and `src/discrete_variational.py`.

Demo and figures generated by `quadmath/scripts/information_demo.py` and `quadmath/scripts/enhanced_active_inference_figures.py` output to `quadmath/output/`:

- **Active Inference Visualizations**: `enhanced_figure_13_4d_trajectory.png`, `enhanced_figure_14_free_energy_landscape.png` demonstrating 4D framework integration
- **Information Geometry Visualizations**: `fisher_information_matrix.png`, `fisher_information_eigenspectrum.png`, `natural_gradient_path.png`, `free_energy_curve.png`, `partition_tetrahedron.png`
- **Raw data**: `enhanced_figure_13_data.npz`, `enhanced_figure_14_data.npz`, `fisher_information_matrix.csv`, `fisher_information_matrix.npz` (F, grads, X, y, w_true, w_est), `fisher_information_eigenvalues.csv`, `fisher_information_eigensystem.npz`
- **External validation**: Cross-reference with volume calculations and tetrahedral modeling tools from the [4dsolutions ecosystem](https://github.com/4dsolutions). See the [Resources](07_resources.md) section for comprehensive details.
