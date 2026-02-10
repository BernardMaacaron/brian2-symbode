"""
Tests for brian2_symbode.extractor.

Uses a 4-population rate-model network (E↔I × 2) defined in conftest.py.
All fixtures are self-contained — no external project dependencies.
"""

import sympy as sp
import pytest

from brian2_symbode import SymbolicGraphExtractor, get_derived_ode, compute_full_jacobian
from tests.conftest import f_symbolic


# ======================================================================
# 1. Flat ODE extraction: get_derived_ode
# ======================================================================

class TestFlatODE:
    """Tests for the monolithic (flat) ODE extraction path."""

    def test_returns_sympy_expression(self, soc_network):
        """get_derived_ode() should return a SymPy expression, not None."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        ode = get_derived_ode(ext, exc, "x")
        assert isinstance(ode, sp.Basic)

    def test_contains_state_variable(self, soc_network):
        """The ODE should reference the group's own state variable."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        ode = get_derived_ode(ext, exc, "x")
        sym_names = {str(s) for s in ode.free_symbols}
        assert f"x_{exc.name}" in sym_names

    def test_contains_tau(self, soc_network):
        """The ODE should reference the time constant τ."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        ode = get_derived_ode(ext, exc, "x")
        sym_names = {str(s) for s in ode.free_symbols}
        assert f"tau_{exc.name}" in sym_names


# ======================================================================
# 2. Structured ODE extraction: get_structured_ode
# ======================================================================

class TestStructuredODE:
    """Tests for the decomposed (structured) ODE extraction path."""

    def test_returns_required_keys(self, soc_network):
        """Structured dict must contain all documented keys."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        expected_keys = {
            "local_expr", "coupling_terms", "dead_symbols",
            "state_symbol", "free_params", "param_metadata",
        }
        assert expected_keys == set(result.keys())

    def test_coupling_count_for_exc(self, soc_network):
        """Exc group receives from E→E and I→E → exactly 2 coupling terms."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        assert len(result["coupling_terms"]) == 2

    def test_coupling_count_for_inh(self, soc_network):
        """Inh group receives from E→I and I→I → exactly 2 coupling terms."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(inh, "x")
        assert len(result["coupling_terms"]) == 2

    def test_dead_symbols_for_exc(self, soc_network):
        """Exc group should have 2 dead symbols (I_EI and I_II target inh, not exc)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        dead_names = {str(s) for s in result["dead_symbols"]}
        assert len(dead_names) == 2
        # These summed vars exist in the network but not for this group
        assert any("I_EI" in n for n in dead_names)
        assert any("I_II" in n for n in dead_names)

    def test_state_symbol_name(self, soc_network):
        """State symbol must match 'x_<groupname>'."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        assert str(result["state_symbol"]) == f"x_{exc.name}"

    def test_local_expr_is_sympy(self, soc_network):
        """local_expr must be a SymPy expression."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        assert isinstance(result["local_expr"], sp.Basic)


# ======================================================================
# 3. Parameter metadata
# ======================================================================

class TestParamMetadata:
    """Tests for the param_metadata field of structured ODE output."""

    def test_metadata_populated(self, soc_network):
        """param_metadata should not be empty for a group with parameters."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        assert len(result["param_metadata"]) > 0

    def test_tau_is_shared(self, soc_network):
        """τ should be flagged as 'shared' (not per-neuron)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        tau_key = f"tau_{exc.name}"
        assert tau_key in result["param_metadata"]
        meta = result["param_metadata"][tau_key]
        assert "shared" in meta["flags"]
        assert not meta["is_per_neuron"]

    def test_g_is_per_neuron(self, soc_network):
        """g should be per-neuron (no 'shared' flag)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        g_key = f"g_{exc.name}"
        assert g_key in result["param_metadata"]
        meta = result["param_metadata"][g_key]
        assert meta["is_per_neuron"]

    def test_weight_metadata_present(self, soc_network):
        """Weight symbols from coupling terms should appear in param_metadata."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        result = ext.get_structured_ode(exc, "x")
        weight_names = {str(ct["weight_symbol"]) for ct in result["coupling_terms"]}
        for wn in weight_names:
            assert wn in result["param_metadata"], f"Missing metadata for weight {wn}"


# ======================================================================
# 4. Jacobian
# ======================================================================

class TestJacobian:
    """Tests for compute_full_jacobian."""

    def test_jacobian_shape(self, soc_network):
        """Jacobian should be (n_groups, n_groups)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        ode_exc = get_derived_ode(ext, exc, "x")
        ode_inh = get_derived_ode(ext, inh, "x")

        x_e = sp.Symbol(f"x_{exc.name}")
        x_i = sp.Symbol(f"x_{inh.name}")
        J = compute_full_jacobian([ode_exc, ode_inh], [x_e, x_i])
        assert J.shape == (2, 2)

    def test_jacobian_diagonal_nonzero(self, soc_network):
        """Diagonal entries (self-coupling + leak) should be non-trivial."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic})
        ode_exc = get_derived_ode(ext, exc, "x")
        ode_inh = get_derived_ode(ext, inh, "x")

        x_e = sp.Symbol(f"x_{exc.name}")
        x_i = sp.Symbol(f"x_{inh.name}")
        J = compute_full_jacobian([ode_exc, ode_inh], [x_e, x_i])
        # Diagonal terms should not be identically zero (leak term is -x/τ)
        assert J[0, 0] != 0
        assert J[1, 1] != 0


# ======================================================================
# 5. Discretisation
# ======================================================================

class TestDiscretise:
    """Tests for the discretise() method."""

    def test_euler_adds_dt_step(self, soc_network):
        """Euler discretisation: x_new = x + dt * f(x)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net)
        x = sp.Symbol("x")
        dt = sp.Symbol("dt")
        ode = -x / sp.Symbol("tau")
        result = ext.discretise(ode, x, "euler", dt)
        # Should contain x minus something
        assert result.has(x)
        assert result.has(dt)

    def test_exponential_euler_decay(self, soc_network):
        """Exponential Euler for dx/dt = -x/τ should produce exp(-dt/τ)."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net)
        x = sp.Symbol("x")
        tau = sp.Symbol("tau")
        dt = sp.Symbol("dt")
        ode = -x / tau
        result = ext.discretise(ode, x, "exponential_euler", dt)
        assert result.has(sp.exp)

    def test_unknown_method_raises(self, soc_network):
        """Unknown integration method should raise ValueError."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net)
        x = sp.Symbol("x")
        with pytest.raises(ValueError, match="Unknown integration method"):
            ext.discretise(-x, x, "magical_method", sp.Symbol("dt"))


# ======================================================================
# 6. Verbose / logging
# ======================================================================

class TestVerbose:
    """Ensure verbose flag does not break extraction."""

    def test_verbose_mode_runs(self, soc_network, capsys):
        """Extraction with verbose=True should not raise."""
        net, exc, inh = soc_network
        ext = SymbolicGraphExtractor(net, function_map={"f": f_symbolic}, verbose=True)
        ode = get_derived_ode(ext, exc, "x")
        assert isinstance(ode, sp.Basic)
        captured = capsys.readouterr()
        assert "[brian2-symbode]" in captured.out
