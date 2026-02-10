# Changelog

## [0.1.0] - 2026-02-10

### Added
- `SymbolicGraphExtractor` class for extracting symbolic ODEs from Brian2 networks
- `get_derived_ode()` for flat symbolic ODE extraction
- `get_structured_ode()` for decomposed local + coupling extraction
- `compute_full_jacobian()` for symbolic Jacobian computation
- `discretise()` supporting Euler, RK2, and Exponential Euler methods
- Dead variable detection for summed targets
- Parameter metadata extraction (shared/per-neuron, flags, sizing)
- Custom function mapping (`function_map`) for Brian2 C++/CUDA functions
