# Exponential Map Models for Neural Spatial Representations

[![Paper](https://img.shields.io/badge/paper-OpenReview-8c1b13.svg)](https://openreview.net/forum?id=3ZO4NGunEr)
[![Documentation](https://img.shields.io/badge/docs-Jupyter--Book-blue.svg)](https://bioai-oslo.github.io/ExponentialRepresentations/)
[![Code Quality](https://github.com/bioAI-Oslo/ExponentialRepresentations/actions/workflows/code_quality.yml/badge.svg)](https://github.com/bioAI-Oslo/ExponentialRepresentations/actions/workflows/code_quality.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Supplementary code for

> Pettersen, M., Haug, N., Bergli, J., Surowiec, T. M. & Lepperød, M. (2026). Exponential Map Models as an Interpretable Framework for Generating Neural Spatial Representations. _The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026)_. https://openreview.net/forum?id=3ZO4NGunEr

The paper constructs neural spatial representations algebraically rather than by training a network. A position maps to a population vector through the matrix exponential of two generator matrices, $\mathbf{p}(x, y) = e^{x G_x + y G_y} \mathbf{p}_0$, and the navigational properties of the representation follow from algebraic conditions on the generators. Commuting generators, $[G_x, G_y] = 0$, give exact and path-independent path integration. Skew-symmetric generators, $G^\top = -G$, give equinorm representations whose similarity depends only on the displacement between two positions. Eigenvalues on sets of roots of unity preserve the metric of flat space and produce hexagonal grid fields, Gaussian eigenvalues produce place-like fields, and a third generator for a context signal models context-dependent remapping.

This repository holds the `expmap` package that implements the construction, the notebooks that reproduce the figures of the paper, and the fits of the model to recorded grid and place cells (Sec. 3.2, App. K). Nothing in the construction is learned. The fits are the one place where gradient descent is used.

<details>
<summary>Abstract</summary>

A foundational challenge in neuroscience and AI is understanding how physical space is mapped into neural representations. While artificial neural networks can learn brain-like spatial representations, their "black-box" nature obscures whether these representations reflect general principles or artifacts of specific architectures, objectives or optimization choices, offering no guarantee that solutions for tasks like path integration generalize beyond training data. Here, we use a first-principles framework based on the matrix exponential, in which generator matrices map physical locations to neural population vectors without gradient-based optimization. Within this formalism, navigational properties follow directly from algebraic constraints on the generators: commuting generators enable trajectory-invariant, exact path integration, while skew-symmetry yields equinorm representations with translation-invariant similarities. Preserving the flat Euclidean metric further constrains the generator spectra to balanced wavevector configurations, such as the roots-of-unity constellations that generate periodic grid fields. Similarity preservation further produces localized place-like fields, and extending the generators to a context signal models context-dependent remapping. Beyond its algebraic relation to continuous attractor and oscillatory interference models, the framework natively supports goal-oriented multi-map navigation through hyperdimensional bundling and binding, fits recorded grid and place cell data, extrapolates their rate maps far beyond the fitted region, and makes testable predictions for entorhinal circuits. By grounding spatial representations in exact algebraic conditions, this work offers a transparent alternative to empirical representation learning, revealing the conditions under which a coherent neural map of space arises.

</details>

## Getting started

Clone the repository and let [uv](https://docs.astral.sh/uv/) create the environment:

```bash
git clone https://github.com/bioAI-Oslo/ExponentialRepresentations.git
cd ExponentialRepresentations
uv sync --all-extras
```

Everything runs through `uv run`, for instance `uv run jupyter lab` to open the notebooks or `uv run pytest` for the tests. The [documentation](https://bioai-oslo.github.io/ExponentialRepresentations/) covers the rest:

- [Basic usage](https://bioai-oslo.github.io/ExponentialRepresentations/docs/basic_usage.html) builds the three models of the paper and shows the pipeline from positions to rate maps and similarity.
- [Reproducing the results](https://bioai-oslo.github.io/ExponentialRepresentations/notebooks/README.html) lists the notebooks, the figures they produce and what they need to run.
- [Data](https://bioai-oslo.github.io/ExponentialRepresentations/data/README.html) describes the recorded rate maps and the saved fits, with their sources and licenses.
- [Developer notes](https://bioai-oslo.github.io/ExponentialRepresentations/docs/dev_guide.html) cover the environment, the checks and the documentation build.
- The [API reference](https://bioai-oslo.github.io/ExponentialRepresentations/docs/api/index.html) documents the package module by module.

## Citation

```bibtex
@inproceedings{pettersen2026exponential,
  title     = {Exponential Map Models as an Interpretable Framework for Generating Neural Spatial Representations},
  author    = {Pettersen, Markus and Haug, Nicolai and Bergli, Joakim and Surowiec, Thomas M. and Lepper{\o}d, Mikkel},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=3ZO4NGunEr}
}
```

## License

The code is released under the [MIT License](LICENSE).

The files in `data/` are derived from published recordings and carry the licenses of their sources. The CA1 fit is distributed under CC BY-SA 4.0. See [data/README.md](data/README.md).
