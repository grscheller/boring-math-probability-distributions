# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`boring-math-probability-distributions` is a PyPI package (`boring_math.probability_distributions`) for generating and visualizing probability distributions. It is part of the broader [boring math](https://grscheller.github.io/boring-math) project family.

Built with Python 3.13+, uses `flit` for packaging, and depends on `pythonic_fp.fptools` (for `MayBe` monad) and `matplotlib`.

## Commands

```fish
# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/distributions/test_beta_distribution.py

# Run a single test class or method
python -m pytest tests/distributions/test_beta_distribution.py::Test_Beta::test_beta_2_2

# Type checking
python -m mypy src/

# Build the package
python -m flit build
```

Pytest is configured via `[tool.pytest]` in `pyproject.toml` (not `[tool.pytest.ini_options]`), with `addopts = ["-ra"]`, `consider_namespace_packages = true`, and `testpaths = ["tests/"]`. Test optional-dependencies (`pytest`, `numpy`, `scipy`) are declared under `[project.optional-dependencies] test`; numpy and scipy are used only in comparison/exploration scripts, not the main test suite.

## Architecture

### Package layout

The package is installed as `boring_math.probability_distributions` (a namespace package under `boring_math`). Source lives in `src/boring_math/probability_distributions/`.

```
distribution.py        # Abstract base classes: ContDist, DiscreteDist
datasets.py            # DataSet (sorted float data + stats), DataSets (container)
distributions/
    beta.py            # Beta(α, β)   — continuous, numerically integrated CDF
    normal.py          # Normal(μ, σ) — continuous, closed-form CDF via erf
    uniform.py         # Uniform      — continuous
    binomial.py        # Binomial     — discrete
    poisson.py         # Poisson(λ)   — discrete
```

### Key design patterns

- **`MayBe` monad** (`pythonic_fp.fptools.maybe.MayBe`) is used pervasively instead of `Optional`. Properties like `mean`, `stdev`, `median` raise `Never` if absent; callers should check `has_mean()` / `has_stdev()` etc. before accessing.
- **Abstract base classes**: `ContDist` and `DiscreteDist` (in `distribution.py`) require subclasses to implement `pdf`, `cdf`, and `__add__`. `ContDist` also stores a numerically integrated `_cdf: MayBe[tuple[float, ...]]`.
- **Beta CDF numerical integration**: `Beta.__init__` pre-computes a 2048-step CDF array. For `α < 1` (singularity at 0), a finer 512-sub-step initial integration seeds the accumulation. `cdf(x)` looks up `floor(x * steps)` in this array.
- **`@final` on all concrete classes** — subclassing distributions is not intended.
- **Greek letter attributes**: Distribution parameters use Unicode names directly (e.g., `self.α`, `self.β`, `self.μ`, `self.σ`, `self.λ`).
- **Matplotlib imports are commented out** in `normal.py` and `beta.py`; plotting is only active in `binomial.py` and `poisson.py`.

### External dependency: `boring_math.special_functions`

`Beta` imports `beta_real` from `boring_math.special_functions.gamma_family.beta`, and `Poisson` imports `exp` from `boring_math.special_functions.exponential.exp`. These come from a separate `boring-math-special-functions` PyPI package in the same `boring_math` namespace.

### `explore/` directory

Contains scratch scripts for visual/manual testing (excluded from the sdist). Not part of the package — run directly with `python explore/plot_beta.py`.

## Coding conventions

- Single-quoted strings in production code (enforced by ruff); double-quoted docstrings.
- Variable name `E741` (ambiguous names like `l`, `O`, `I`) is globally ignored via ruff config — Greek letters are preferred.
- Tests are organized under `tests/distributions/` mirroring the source layout.
- Tolerance levels in tests are named `toleranceNN` where NN is the negative exponent (e.g., `tolerance07 = 1e-7`).
