"""
brian2-symbode — Symbolic ODE extraction from Brian2 networks.

Extract, decompose, and analyse the full symbolic ODE system that Brian2
implicitly constructs from NeuronGroups, Synapses, and summed variables.

Quick start::

    from brian2_symbode import SymbolicGraphExtractor, get_derived_ode

    extractor = SymbolicGraphExtractor(net, function_map={'f': my_f_symbolic})
    ode = get_derived_ode(extractor, neurons, 'v')

"""

from brian2_symbode.extractor import (
    SymbolicGraphExtractor,
    get_derived_ode,
    compute_full_jacobian,
)

__version__ = "0.1.0"

__all__ = [
    "SymbolicGraphExtractor",
    "get_derived_ode",
    "compute_full_jacobian",
]
