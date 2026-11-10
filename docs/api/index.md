# API reference

The package is organized along the mathematical structure of the framework. High-level constructors build complete models in a single call, while individual modules expose the underlying components.

## High-level constructors

```{eval-rst}
.. automodule:: expmap
   :no-members:

.. autofunction:: expmap.grid_cell_model

.. autofunction:: expmap.place_cell_model

.. autofunction:: expmap.remapping_model
```

## Module reference

- [`expmap.model`](model.md): `ExponentialMap` and `RemappingModel`, the mapping from position and context to the population vector (Eq. 1, Sec. 2.5).
- [`expmap.generators`](generators.md): the construction pipeline from an eigenvalue specification to the generator matrices (Sec. 2.3-2.4, App. G).
- [`expmap.linalg`](linalg.md): orthogonal mixing matrices, the canonical origin state and the block rotations of App. C.
- [`expmap.analysis`](analysis.md): rate maps, batch inference, the similarity function, autocorrelograms and the grid score.
- [`expmap.navigation`](navigation.md): bundling, binding, querying and the closed-form similarity gradient (Sec. 3.3, App. L).
- [`expmap.attractor`](attractor.md): the continuous attractor network whose on-manifold dynamics is the exponential map (Sec. 3.1, App. J.1).
- [`expmap.fitting`](fitting.md): fitting generators to recorded rate maps (Sec. 3.2, App. K).
- [`expmap.plasticity`](plasticity.md): the local learning rule for skew-symmetric connectivity (App. J.4).
- [`expmap.simulation`](simulation.md): uniform position samples.
- [`expmap.plotting`](plotting.md): rate map collages.
