# brian2-symbode

**Symbolic ODE extraction for Brian2 networks.**

[![CI](https://github.com/ARC-IIT/brian2-symbode/actions/workflows/ci.yml/badge.svg)](https://github.com/ARC-IIT/brian2-symbode/actions)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

`brian2-symbode` extracts the full symbolic ODE system from a [Brian2](https://brian2.readthedocs.io/) network, resolving summed synaptic variables, subexpressions, and custom functions into clean SymPy expressions.

## Note
This is an early prototype focused on core functionality. The API and features are likely to evolve based on user feedback and use cases. Please open an issue or reach out if you have suggestions or encounter any problems. I'm new to handling repositories and packaging, so any help or feedback is greatly appreciated!

**Keep in mind the repo was set up by an AI assistant, so there may be some rough edges in terms of structure and documentation.**

The examples are mostly based on work I have been doing. Nothing ensures that the library is robust in general, but I hope it can be useful for others working with Brian2 and symbolic analysis.

Important: the project and tests currently assume dimensionless variables (no physical units). The code and test fixtures were validated for this dimensionless mode; using physical units (e.g. `Hz`, `second`) may raise Brian2 unit-mismatch errors and has not been tested.

## Why?

Brian2 distributes network dynamics across NeuronGroups, Synapses, and implicit summed variables. This makes it hard to:

- Compute **Jacobians** for stability analysis
- Build **differentiable simulations** in PyTorch / JAX
- Generate **analytical** expressions for publication

`brian2-symbode` reconstructs the complete ODE from these distributed pieces.

## Features

- **Recursive resolution** of summed variables, subexpressions, and pre/post synaptic references
- **Structured decomposition** separating local dynamics from matrix-vector coupling terms
- **Custom function mapping** (replace Brian2 C++/CUDA functions with SymPy equivalents)
- **Dead variable detection** (summed targets that are structurally zero for a given group)
- **Parameter metadata** extraction (shared/per-neuron flags, group sizes, equation types)
- **Symbolic Jacobian** computation
- **Discretisation** (Euler, RK2, Exponential Euler)

## Installation

This project is not published on PyPI. Install from source for now:

```bash
git clone https://github.com/ARC-IIT/brian2-symbode.git
cd brian2-symbode
python -m pip install -e ".[dev]"
```

## Quick Start

```python
from brian2 import *
from brian2_symbode import SymbolicGraphExtractor, get_derived_ode, compute_full_jacobian

# 1. Build your Brian2 network as usual
start_scope()
eqs = '''
dx/dt = (-x + I_syn) / tau : 1
I_syn : 1
tau : second (shared)
g : 1
r = g * x : 1
'''
G = NeuronGroup(10, eqs, method='euler')
G.tau = 200 * ms
S = Synapses(G, G,
    model='w : 1\nI_syn_post = w * r_pre : 1 (summed)',
    method='euler')
S.connect(p=0.5)
S.w = 'rand()'
net = Network(G, S)
net.store()

# 2. Extract the symbolic ODE
ext = SymbolicGraphExtractor(net)
ode = get_derived_ode(ext, G, 'x')
print("ODE:", ode)

# 3. Compute the Jacobian
import sympy as sp
x = sp.Symbol(f'x_{G.name}')
J = compute_full_jacobian([ode], [x])
print("Jacobian:", J)

# 4. Get structured decomposition (for differentiable simulators)
structured = ext.get_structured_ode(G, 'x')
print("Local dynamics:", structured['local_expr'])
print("Coupling terms:", len(structured['coupling_terms']))
print("Parameters:", [str(p) for p in structured['free_params']])
```

## Custom Function Mapping

Brian2 supports custom C++/CUDA functions that have no SymPy equivalent. Use `function_map` to provide symbolic replacements:

```python
import sympy as sp

def my_transfer_function(x, g, r_0, r_max):
    """Piecewise saturating transfer function."""
    span = r_max - r_0
    return sp.Piecewise(
        (r_0, x < 0),
        (r_0 + span * sp.tanh((g * x) / span), True),
    )

ext = SymbolicGraphExtractor(net, function_map={'f': my_transfer_function})
```

## Structured ODE Output

`get_structured_ode()` returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `local_expr` | `sympy.Expr` | Element-wise local dynamics |
| `coupling_terms` | `list[dict]` | One per incoming synapse (weight, source, activation) |
| `dead_symbols` | `list[Symbol]` | Summed variables that are structurally zero |
| `state_symbol` | `Symbol` | The state variable symbol |
| `free_params` | `list[Symbol]` | All non-state parameter symbols |
| `param_metadata` | `dict` | Per-parameter Brian2 metadata (flags, sizing, etc.) |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=brian2_symbode
```

## Citation

If you use `brian2-symbode` in your research, please cite:

```bibtex
@software{brian2_symbode,
  title = {brian2-symbode: Symbolic ODE Extraction for Brian2 Networks},
  url = {https://github.com/ARC-IIT/brian2-symbode},
  license = {GPL-3.0-or-later},
}
```

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
