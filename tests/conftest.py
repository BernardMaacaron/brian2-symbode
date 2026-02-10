"""
Shared test fixtures for brian2-symbode.

Provides:
- ``soc_equations``: A realistic rate-model equation string with summed
  synaptic currents and a custom transfer function.
- ``f_symbolic``: A SymPy-compatible piecewise transfer function that
  mirrors the C++/CUDA implementations used in SOC networks.
- ``soc_network``: A fully wired 4-population (E↔I × 2) Brian2 network
  with identity weight matrices.
"""

import numpy as np
import pytest
import sympy as sp
from brian2 import (
    NeuronGroup,
    Synapses,
    Network,
    ms,
    start_scope,
    defaultclock,
)


# =====================================================================
# Equation string — a realistic rate-model ODE
# =====================================================================

SOC_EQ_RATE = """
dx/dt = (-x + I_EE + I_EI + I_IE + I_II) / tau : 1
I_EE : 1
I_EI : 1
I_IE : 1
I_II : 1
tau : second (shared)
g : 1
r_0 : 1 (shared, constant)
r_max : 1 (shared, constant)
r = f(x, g, r_0, r_max) : 1
"""


# =====================================================================
# Symbolic transfer function (SymPy-compatible)
# =====================================================================

def f_symbolic(x, g, r_0=20.0, r_max=100.0):
    """Piecewise saturating transfer function.

    For x < 0 → r_0; for x >= 0 → r_0 + span * tanh(g*x / scale).
    """
    span = r_max - r_0
    scale = span if not isinstance(span, sp.Basic) else span
    return sp.Piecewise(
        (r_0, x < 0),
        (r_0 + scale * sp.tanh((g * x) / scale), True),
    )


# =====================================================================
# Network builder (4 populations, E↔I × 2)
# =====================================================================

@pytest.fixture
def soc_network():
    """Build a minimal 4-population SOC network for testing.

    Returns (network, exc, inh) where ``exc`` and ``inh`` each have
    N_E=3 and N_I=2 neurons with identity coupling weights.
    """
    start_scope()

    N_E, N_I = 3, 2
    tau_val = 200 * ms

    # --- Neuron groups ---
    exc = NeuronGroup(N_E, SOC_EQ_RATE, method="euler")
    inh = NeuronGroup(N_I, SOC_EQ_RATE, method="euler")

    exc.tau = tau_val
    inh.tau = tau_val
    exc.r_0 = 20.0
    inh.r_0 = 20.0
    exc.r_max = 100.0
    inh.r_max = 100.0
    exc.g = 1.0
    inh.g = 1.0

    # --- E→E synapse ---
    syn_EE = Synapses(
        exc,
        exc,
        model="we : 1\nI_EE_post = we * r_pre : 1 (summed)",
        method="euler",
    )
    syn_EE.connect()
    W_ee = np.eye(N_E)
    syn_EE.we = W_ee.flatten()

    # --- I→E synapse ---
    syn_IE = Synapses(
        inh,
        exc,
        model="wi : 1\nI_IE_post = wi * r_pre : 1 (summed)",
        method="euler",
    )
    syn_IE.connect()
    W_ie = np.ones((N_E, N_I))
    syn_IE.wi = W_ie.flatten()

    # --- E→I synapse ---
    syn_EI = Synapses(
        exc,
        inh,
        model="we : 1\nI_EI_post = we * r_pre : 1 (summed)",
        method="euler",
    )
    syn_EI.connect()
    W_ei = np.ones((N_I, N_E))
    syn_EI.we = W_ei.flatten()

    # --- I→I synapse ---
    syn_II = Synapses(
        inh,
        inh,
        model="wi : 1\nI_II_post = wi * r_pre : 1 (summed)",
        method="euler",
    )
    syn_II.connect()
    W_ii = np.eye(N_I)
    syn_II.wi = W_ii.flatten()

    net = Network(exc, inh, syn_EE, syn_IE, syn_EI, syn_II)
    net.store()

    return net, exc, inh
