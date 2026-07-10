# Installation

scIToFlow requires Python 3.10 or newer.

## From PyPI-style install (GitHub)

```bash
pip install git+https://github.com/Bernaljp/scitoflow.git
```

The core install pulls PyTorch, [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq),
and [`torch-geometric`](https://pytorch-geometric.readthedocs.io/), plus the standard
scientific stack (NumPy, SciPy, pandas, scikit-learn, AnnData, matplotlib).

## Optional extras

```bash
# moments + gene-activity preprocessing (scanpy, dynamo-release)
pip install "scitoflow[preprocess] @ git+https://github.com/Bernaljp/scitoflow.git"

# development (pytest, ruff)
pip install "scitoflow[dev] @ git+https://github.com/Bernaljp/scitoflow.git"

# building these docs (sphinx, myst-nb, sphinx-book-theme)
pip install "scitoflow[docs] @ git+https://github.com/Bernaljp/scitoflow.git"
```

| Extra | Adds | For |
|-------|------|-----|
| `preprocess` | `scanpy`, `dynamo-release` | Building a model-ready AnnData from raw matrices |
| `dev` | `pytest`, `ruff` | Running the test suite and linter |
| `docs` | `sphinx`, `myst-nb`, `sphinx-book-theme` | Building this documentation |

## From source

```bash
git clone https://github.com/Bernaljp/scitoflow.git
cd scitoflow
pip install -e ".[dev]"
pytest -q          # 6 tests, runs on CPU in a few seconds
```

## GPU vs CPU

```{admonition} You do not need a GPU to try scIToFlow
:class: tip
`scitoflow.train_vae` resolves the device automatically: it uses CUDA when a GPU is
available and otherwise falls back to CPU. The tutorials in this documentation run
end to end on CPU using the built-in {func}`scitoflow.simulate_dataset`. For real
tissue-scale datasets (thousands of spots, tens of epochs) a CUDA GPU is strongly
recommended.
```

The model runs in double precision. Set the default dtype once at the top of your script or
notebook so the data and the model parameters match:

```python
import torch
torch.set_default_dtype(torch.float64)
```

## Verify the install

```python
import scitoflow
print(scitoflow.__version__)

from scitoflow import simulate_dataset, VAE, get_topology
adata = simulate_dataset(n_genes=30, grid=12, seed=0)
print(adata)                       # 144 spots x 30 genes, with M_c/M_u/M_s layers
print(get_topology("full").states) # ('c', 'u', 's')
```
