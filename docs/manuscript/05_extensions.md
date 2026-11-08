# Research Extensions Beyond the Implemented Lattice Methods {#extensions-of-4d-and-quadrays}

The modules in [coordinate methods](03_quadray_methods.md), [optimization](04_optimization_in_4d.md), [field learning](11_ivm_field_learning.md), and [dynamics](12_ivm_dynamics.md) are implemented examples. The directions below are proposals or interpretive context, not additional working APIs, measured speedups, or demonstrated security guarantees.

## Multi-Objective Optimization

A candidate $q$ Pareto-dominates $q'$ when $f_k(q)\leq f_k(q')$ for every objective, with strict inequality for at least one. Computing that relation requires the objectives themselves. A tetrahedral face does not automatically identify a Pareto front, and traversing adjacent tetrahedra need not enumerate nondominated solutions.

Native rational tetravolume could be studied as one spatial diversity diagnostic, but it depends on the embedding and can vanish for widely separated coplanar candidates. No Pareto optimizer is supplied by the discrete single-objective APIs.

## Machine Learning and Robustness

Integer coordinates can encode a structural prior. Whether that prior improves prediction or robustness requires a specified model, held-out data, and a comparator. Discrete projection alone does not establish adversarial robustness: an attacker may change lattice labels or cross a decision boundary. The existing [evaluation utilities](15_learning_evaluation.md) measure site and temporal prediction; they are not an adversarial-defense evaluation.

## Computer Graphics and GPU Acceleration

The current figures use NumPy and matplotlib on the CPU. Batched forward embedding and independent geometric calculations are candidates for GPU work, but this repository does not provide CUDA kernels, compute shaders, GPU QuickHull, or measured GPU speedups. Exact Bareiss arithmetic uses unbounded integers; replacing it with fixed-width GPU arithmetic requires overflow analysis and cannot silently retain the same exactness claim.

For general GPU background, see [CUDA dynamic parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/) and [GPU Gems parallel computing](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing). These references are not evidence of an implemented QuadMath accelerator.

## Active Inference and Free Energy

The implemented discrete-state objective is $\mathcal F=\mathbb E_Q[-\log P(o\mid s)]+\mathrm{KL}(Q\|P)$. The named `expected_free_energy` helper subtracts a supplied scalar log preference; it does not construct policy-conditioned future observations or implement a full active-inference agent. See [the active-inference appendix](09_free_energy_active_inference.md) for the toy-model boundary and callable contracts.

Natural-gradient increments can be composed with suitable models, but the supplied perception/action finite-difference helpers do not automatically use a Fisher metric. No biological or metabolic-efficiency result is established here.

## Complex Systems and Collective Intelligence

Tetrahedral graphs could serve as interaction topologies for consensus or resource-flow models. A graph's shape alone establishes neither consensus nor resistance to manipulated inputs. Such work would need explicit dynamics, adversary assumptions, observables, and comparative experiments. The implemented [conservative graph operator](12_ivm_dynamics.md#conservative-graph-transport) establishes a narrower algebraic mass-preservation property, not a distributed-security protocol.

## Geospatial Intelligence and the World Game

Fuller's World Game motivates whole-system resource questions, but no georeferenced data pipeline, spherical Earth tessellation, global allocation solver, or sensor-authentication system is implemented here. Planar FCC indexing does not itself bound distortion on a globe. Any future geospatial model must declare its projection, geographic units, boundary conditions, data provenance, and optimization objective independently of the Quadray representation.

## Quadrays, Synergetics (Fuller.4D), and William Blake

Blake's fourfold vision and Fuller's geometric pedagogy provide cultural metaphors for multiscale interpretation; they are not mathematical equivalences or empirical evidence for an inference algorithm. Background includes [Blake's visionary geography](https://www.britishartstudies.ac.uk/index/article-index/visionary-sense-of-london/article-category/cover-collaboration), [Active Inference and Blake](https://zenodo.org/records/13711302), and [Fuller and Blake in juxtaposition](https://zenodo.org/records/7519132).

## Pedagogy and Implementations

The [resource bibliography](99_resources.md) links external coordinate implementations and teaching materials. They offer possible comparisons, not an assertion that this checkout has executed every external implementation.

## Higher Dimensions and Decompositions

An $n$-simplex for $n>3$ cannot be decomposed into three-dimensional tetrahedra to recover its $n$-dimensional volume. Its tetrahedral faces have lower-dimensional measure. Genuine higher-dimensional work requires $n$ independent directions and the corresponding $n$-volume formula, such as a generalized Cayley–Menger determinant. The implemented native tetravolume is strictly three-dimensional.

## Limitations and Future Work

Research questions include broader objective families, embedding/scale sensitivity, explicit continuous/discrete projection schedules, and domain-specific robustness tests. None is a guarantee supplied by integer volume quantization. Use [the mathematical specification](../../SPEC.md) for current quotient, metric, and formal scope rather than treating this proposal chapter as an API contract.
