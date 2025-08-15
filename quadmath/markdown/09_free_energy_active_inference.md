# Appendix: The Free Energy Principle and Active Inference

## Overview

The Free Energy Principle (FEP) posits that biological systems maintain their states by minimizing variational free energy, thereby reducing surprise via prediction and model updating. Active Inference extends this by casting action selection as inference under prior preferences. Background: see the concise overview on the [Free energy principle](https://en.wikipedia.org/wiki/Free_energy_principle) and the monograph [Active Inference (MIT Press)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind).

This appendix emphasizes relationships among: (i) the four-fold partition of Active Inference, (ii) Quadrays (Fuller.4D) as a geometric scaffold for mapping this partition, and (iii) information-geometric flows (Einstein.4D analogy) that underpin perception–action updates. For the naming of 4D namespaces used throughout—Coxeter.4D (Euclidean E4), Einstein.4D (Minkowski spacetime analogy), Fuller.4D (Synergetics/Quadrays)—see `02_4d_namespaces.md`.

## Mathematical Formulation and Equation Callouts (Equations linkage)

- Variational free energy (discrete states) — see Eq. \eqref{eq:supp_free_energy} in the Equations appendix, implemented by [`free_energy`](08_equations_appendix.md#code:free_energy):

  \begin{equation}\label{eq:free_energy_appendix_ref}
  \mathcal{F} = -\log P(o\mid s) + \mathrm{KL}\big[ Q(s)\;\|\; P(s) \big]
  \end{equation}

  where \(Q(s)\) is a variational posterior, \(P(s)\) a prior, and \(P(o\mid s)\) the likelihood. Lower \(\mathcal{F}\) is better.

- Fisher Information Matrix (FIM) as metric — see Eq. \eqref{eq:supp_fim} and [`fisher_information_matrix`](08_equations_appendix.md#code:fisher_information_matrix):

  \begin{equation}\label{eq:fim_definition}
  F_{i,j} = \mathbb{E}\left[ \partial_{\theta_i} \log p(x;\theta)\; \partial_{\theta_j} \log p(x;\theta) \right].
  \end{equation}

- Natural gradient descent under information geometry — see Eq. \eqref{eq:supp_natgrad} and [`natural_gradient_step`](08_equations_appendix.md#code:natural_gradient_step); overview: [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient):

  \begin{equation}\label{eq:natural_gradient_update}
  \theta \leftarrow \theta - \eta\, F(\theta)^{-1}\, \nabla_{\theta} L(\theta).
  \end{equation}

Figures: Figure \ref{fig:fisher_information_matrix}, Figure \ref{fig:fim_eigenspectrum}, Figure \ref{fig:natural_gradient_path}, Figure \ref{fig:free_energy_curve}.

Discrete variational optimization on the quadray lattice: `discrete_ivm_descent` greedily descends a free-energy-like objective over IVM moves, yielding integer-valued trajectories. See the path animation artifact `discrete_path.mp4` in `quadmath/output/`.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/partition_tetrahedron.png}
\caption{Four-fold partition mapped to a Quadray tetrahedron — vertices labeled as Internal ($\mu$), Sensory (s), Active (a), External ($\psi$). Edges depict pairwise couplings: ($\mu$–s) perceptual inference, (a–$\psi$) control, and cross-couplings capturing active perception and sensorimotor contingencies.}
\label{fig:partition_tetrahedron}
\end{figure}

## Four-Fold Partition and Tetrahedral Mapping (Quadrays; Fuller.4D)

Active Inference partitions the agent–environment system into four coupled states:

- Internal (\(\mu\)) — agent's internal states
- Sensory (\(s\)) — observations
- Active (\(a\)) — actions
- External (\(\psi\)) — latent environmental causes

See, for an overview of this partition and generative process formulations, the [Active Inference review](https://discovery.ucl.ac.uk/id/eprint/10176959/1/1-s2.0-S1571064523001094-main.pdf) and the general entry on [Active inference](https://en.wikipedia.org/wiki/Active_inference).

Tetrahedral mapping via Quadrays (Fuller.4D): assign each state to a vertex of a tetrahedron, using Quadray coordinates `(A,B,C,D)` with non-negative components and at least one zero after normalization. One canonical mapping is `A \leftrightarrow Internal (\mu)`, `B \leftrightarrow Sensory (s)`, `C \leftrightarrow Active (a)`, `D \leftrightarrow External (\psi)`. The edges capture the pairwise couplings (e.g., `\mu\text{--}s` for perceptual inference; `a\text{--}\psi` for control). Integer tetravolume then quantifies the “coupled capacity” region spanned by jointly feasible states in a time slice; see `Quadray` and tetravolume methods in `03_quadray_methods.md`.

Interpretation note: this Quadray-based mapping is a didactic geometric scaffold. It is not standard in the Active Inference literature, which typically develops the four-state partition in probabilistic graphical terms. Our use highlights structural symmetries and discrete volumetric quantities available in Fuller.4D, building on the computational foundations developed in the [4dsolutions ecosystem](https://github.com/4dsolutions) for tetrahedral modeling and volume calculations.

Code linkage (no snippet): see `example_partition_tetra_volume` in `src/examples.py` and Figure \ref{fig:partition_tetrahedron}.

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

Demo and figures generated by `quadmath/scripts/information_demo.py` output to `quadmath/output/`:

- **Visualizations**: `fisher_information_matrix.png`, `fisher_information_eigenspectrum.png`, `natural_gradient_path.png`, `free_energy_curve.png`, `partition_tetrahedron.png`.
- **Raw data**: `fisher_information_matrix.csv`, `fisher_information_matrix.npz` (F, grads, X, y, w_true, w_est), `fisher_information_eigenvalues.csv`, `fisher_information_eigensystem.npz`.
- **External validation**: Cross-reference with volume calculations in [`Qvolume.ipynb`](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/Qvolume.ipynb) and tetrahedral modeling tools from the [4dsolutions ecosystem](https://github.com/4dsolutions).
