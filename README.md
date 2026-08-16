# Exponential Map Models for Neural Spatial Representations

A first-principles algebraic framework for generating interpretable neural spatial representations using the matrix exponential.

## Overview

This repository contains the official implementation for the paper **"Exponential Map Models as an Interpretable Framework for Generating Neural Spatial Representations."**

Unlike "black-box" deep learning models, this framework constructs spatial maps (such as grid cells and place cells) from a transparent mathematical foundation. By modeling the mapping from physical space to neural activity via the matrix exponential of generator matrices, we derive the exact algebraic conditions required for key navigational properties:

- **Path Integration:** Guaranteed by commuting generator matrices ($[G_x, G_y] = 0$).
- **Translational Invariance:** Achieved via skew-symmetric generators ($G^T = -G$), ensuring orthogonal transformations.
- **Metric Preservation:** Arises when generator eigenvalues form sets of roots of unity, naturally recovering hexagonal grid patterns.

## Installation

```bash
git clone https://github.com/bioAI-Oslo/ExponentialRepresentations.git
cd ExponentialRepresentations
uv sync --all-extras
```

## Quick Start

### High-level API

```python
import expmap
import torch

# Create a hexagonal grid cell model (M=3 symmetry)
model, p0 = expmap.grid_cell_model(M=3, n_copies=50)
positions = torch.rand(1000, 2) * 10 - 5
representations = model(positions, p0)

# Create a place cell model
model, p0 = expmap.place_cell_model(N=300)

# Create a remapping model with context
model, p0 = expmap.remapping_model(M=3, n_copies=50)
representations = model(positions, p0, context=0.0)
```

### Low-level API

```python
from expmap.generators import roots_of_unity_eigenvalues, block_diag_skew, build_generators
from expmap.linalg import random_orthogonal_matrix, starting_state
from expmap import ExponentialMap
import numpy as np

# Construct generators from scratch
phi = roots_of_unity_eigenvalues(M=3, n_copies=50)
sigma_x = block_diag_skew(np.cos(phi))
sigma_y = block_diag_skew(np.sin(phi))
R = random_orthogonal_matrix(len(sigma_x), seed=42)
Gx, Gy = build_generators(sigma_x, sigma_y, R)
p0 = starting_state(R)

model = ExponentialMap(Gx, Gy)
```

### Navigation via bundling and binding

```python
from expmap.navigation import bundle, query

# Bundle reward locations into a composite memory
reward_positions = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
reward_reps = model(reward_positions, p0)
memory = bundle(reward_reps)

# Query current location against memory
current_rep = model(torch.tensor([[0.0, 0.0]]), p0)
similarity = query(current_rep[0], memory)
```

## Notebooks

Notebooks reproduce the figures from the manuscript:

| Notebook                                         | Description                                                | Figure          |
| ------------------------------------------------ | ---------------------------------------------------------- | --------------- |
| `01_metric_preserving_representations.ipynb`     | Grid cell-like representations from roots of unity         | Fig. 2          |
| `02_similarity_preserving_representations.ipynb` | Place field-like representations from Gaussian eigenvalues | Fig. 3          |
| `03_context_dependent_remapping.ipynb`           | Context-dependent remapping                                | Fig. 3          |
| `04_expmap_cann.ipynb`                           | CANN on-manifold dynamics                                  | Fig. 4          |
| `05_binding_and_bundling.ipynb`                  | Multi-map navigation                                       | Fig. 5          |
| `06_bessel_zeros.ipynb`                          | Bessel function zeros and grid scale ratios                | Fig. 6          |
| `07_roots_of_unity.ipynb`                        | Roots of unity visualizations                              | Fig. 2 (detail) |
| `08_experimental_fitting.ipynb`                  | Fitting to experimental grid cell data                     | Fig. 9, 10      |
| `09_learning_rule.ipynb`                         | Biologically plausible learning rule                       | Appendix J.4    |

## Package Structure

- `src/expmap/` — Core library
  - `model.py` — `ExponentialMap` and `RemappingModel` (`nn.Module`)
  - `generators.py` — Generator matrix construction (eigenvalues → block-diagonal → full matrix)
  - `linalg.py` — Orthogonal matrices, starting states, basis construction
  - `analysis.py` — Ratemaps, batch inference, similarity computation
  - `navigation.py` — Hyperdimensional bundling, binding, and querying
  - `simulation.py` — Trajectory and position generation
  - `plotting.py` — Visualization utilities
- `notebooks/` — Reproduce manuscript figures
- `data/` — Experimental data and pre-computed parameters
- `tests/` — Unit tests

## Citation

If you use this code or framework in your research, please cite:

```
@inproceedings{pettersen2026exponential,
  title={Exponential Map Models as an Interpretable Framework for Generating Neural Spatial Representations},
  author={Pettersen, Markus and Haug, Nicolai and Bergli, Joakim and Surowiec, Thomas M. and Lepperød, Mikkel},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
