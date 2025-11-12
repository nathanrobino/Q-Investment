"""
Unit tests for the Qmod class.

This test suite covers:
- Parameter validation
- Model solution and steady state computation
- Simulation functionality
- Error handling

Run with: python -m pytest tests/test_qmod.py
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Qmod import Qmod


class TestQmodInitialization:
    """Test model initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test that model initializes with default parameters."""
        model = Qmod()
        assert model.beta == 0.98
        assert model.tau == 0.05
        assert model.alpha == 0.33
        assert model.omega == 1
        assert model.zeta == 0
        assert model.delta == 0.1
        assert model.psi == 1
        
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = Qmod(beta=0.95, alpha=0.3, tau=0.1)
        assert model.beta == 0.95
        assert model.alpha == 0.3
        assert model.tau == 0.1
        
    def test_invalid_beta(self):
        """Test that invalid beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be in"):
            Qmod(beta=-0.1)
        with pytest.raises(ValueError, match="beta must be in"):
            Qmod(beta=1.1)
            
    def test_invalid_tau(self):
        """Test that invalid tau raises ValueError."""
        with pytest.raises(ValueError, match="tau must be in"):
            Qmod(tau=-0.1)
        with pytest.raises(ValueError, match="tau must be in"):
            Qmod(tau=1.0)
            
    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            Qmod(alpha=0)
        with pytest.raises(ValueError, match="alpha must be in"):
            Qmod(alpha=1)
            
    def test_invalid_omega(self):
        """Test that negative omega raises ValueError."""
        with pytest.raises(ValueError, match="omega must be non-negative"):
            Qmod(omega=-1)
            
    def test_invalid_delta(self):
        """Test that invalid delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be in"):
            Qmod(delta=0)
        with pytest.raises(ValueError, match="delta must be in"):
            Qmod(delta=1.1)
            
    def test_invalid_psi(self):
        """Test that non-positive psi raises ValueError."""
        with pytest.raises(ValueError, match="psi must be positive"):
            Qmod(psi=0)
        with pytest.raises(ValueError, match="psi must be positive"):
            Qmod(psi=-1)


class TestEconomicFunctions:
    """Test economic functions of the model."""
    
    @pytest.fixture
    def model(self):
        """Create a standard model instance for testing."""
        return Qmod(beta=0.98, alpha=0.33, tau=0.05)
    
    def test_production_function(self, model):
        """Test Cobb-Douglas production function."""
        k = 1.0
        output = model.f(k)
        expected = model.psi * k**model.alpha
        assert np.isclose(output, expected)
        
    def test_marginal_product(self, model):
        """Test marginal product of capital."""
        k = 1.0
        mpk = model.f_k(k)
        expected = model.psi * model.alpha * k**(model.alpha - 1)
        assert np.isclose(mpk, expected)
        
    def test_after_tax_revenue(self, model):
        """Test after-tax revenue calculation."""
        k = 1.0
        revenue = model.pi(k)
        expected = (1 - model.tau) * model.f(k)
        assert np.isclose(revenue, expected)
        
    def test_adjustment_cost(self, model):
        """Test adjustment cost function."""
        k = 1.0
        i = 0.15  # Some level of investment
        cost = model.j(i, k)
        assert cost >= 0  # Adjustment costs should be non-negative


class TestModelSolution:
    """Test model solution methods."""
    
    @pytest.fixture
    def model(self):
        """Create and solve a standard model."""
        m = Qmod(beta=0.98, alpha=0.33, tau=0.05)
        return m
    
    def test_solve_computes_steady_state(self, model):
        """Test that solve() computes steady state."""
        model.solve(n_points=20)
        assert model.kss is not None
        assert model.kss > 0
        
    def test_solve_computes_policy_function(self, model):
        """Test that solve() computes policy function."""
        model.solve(n_points=20)
        assert model.k1Func is not None
        
    def test_solve_computes_P(self, model):
        """Test that solve() computes price P."""
        model.solve(n_points=20)
        assert model.P is not None
        assert model.P == (1 - model.zeta)
        
    def test_policy_function_at_steady_state(self, model):
        """Test that policy function returns steady state at steady state."""
        model.solve(n_points=20)
        k_next = model.k1Func(model.kss)
        # Should be close to steady state (within numerical tolerance)
        assert np.isclose(k_next, model.kss, rtol=0.01)
        
    def test_steady_state_is_positive(self, model):
        """Test that steady state capital is positive."""
        model.solve(n_points=20)
        assert model.kss > 0


class TestSimulation:
    """Test simulation functionality."""
    
    @pytest.fixture
    def solved_model(self):
        """Create and solve a model for simulation tests."""
        m = Qmod(beta=0.98, alpha=0.33, tau=0.05)
        m.solve(n_points=30)
        return m
    
    def test_simulate_without_solve_raises_error(self):
        """Test that simulate() raises error if model not solved."""
        model = Qmod()
        with pytest.raises(RuntimeError, match="Must call solve"):
            model.simulate(k0=1.0, t=10)
            
    def test_simulate_returns_correct_shape(self, solved_model):
        """Test that simulate() returns array of correct length."""
        t = 50
        path = solved_model.simulate(k0=1.5, t=t)
        assert len(path) == t
        
    def test_simulate_starts_at_k0(self, solved_model):
        """Test that simulation starts at initial capital."""
        k0 = 1.5
        path = solved_model.simulate(k0=k0, t=50)
        assert np.isclose(path[0], k0)
        
    def test_simulate_converges_to_steady_state(self, solved_model):
        """Test that simulation converges to steady state."""
        k0 = 1.5
        path = solved_model.simulate(k0=k0, t=200)
        # Final value should be close to steady state
        assert np.isclose(path[-1], solved_model.kss, rtol=0.05)
        
    def test_simulate_monotonic_convergence(self, solved_model):
        """Test that capital converges monotonically from above."""
        k0 = 2.0  # Start above steady state
        path = solved_model.simulate(k0=k0, t=100)
        # Path should be decreasing
        assert np.all(np.diff(path) <= 1e-6)  # Allow small numerical errors


class TestShootingAlgorithm:
    """Test the shooting algorithm and related methods."""
    
    @pytest.fixture
    def model(self):
        """Create a solved model for shooting tests."""
        m = Qmod(beta=0.98, alpha=0.33, tau=0.05)
        m.solve(n_points=20)
        return m
    
    def test_shoot_returns_correct_length(self, model):
        """Test that shoot() returns array of correct length."""
        k0 = 1.0
        k1 = 1.05
        t = 50
        path = model.shoot(k0, k1, t)
        assert len(path) == t
        
    def test_shoot_starts_at_k0_k1(self, model):
        """Test that shoot() starts at given initial values."""
        k0 = 1.0
        k1 = 1.05
        path = model.shoot(k0, k1, 50)
        assert np.isclose(path[0], k0)
        assert np.isclose(path[1], k1)
        
    def test_find_k1_returns_scalar(self, model):
        """Test that find_k1() returns a scalar value."""
        k0 = 1.5
        k1 = model.find_k1(k0)
        assert isinstance(k1, (float, np.floating))


class TestPhaseDiagram:
    """Test phase diagram related functions."""
    
    @pytest.fixture
    def model(self):
        """Create a solved model for phase diagram tests."""
        m = Qmod(beta=0.98, alpha=0.33, tau=0.05)
        m.solve(n_points=20)
        return m
    
    def test_iota_returns_scalar(self, model):
        """Test that iota() returns a scalar."""
        lam_1 = 0.95
        result = model.iota(lam_1)
        assert isinstance(result, (float, np.floating))
        
    def test_jkl_returns_scalar(self, model):
        """Test that jkl() returns a scalar."""
        lam_1 = 0.95
        result = model.jkl(lam_1)
        assert isinstance(result, (float, np.floating))
        
    def test_lambda0locus_at_steady_state(self, model):
        """Test lambda0locus at steady state."""
        # At steady state, lambda should equal P
        lam = model.lambda0locus(model.kss)
        # Should be close to P (within numerical tolerance)
        assert not np.isnan(lam)
        
    def test_findLambda_returns_scalar(self, model):
        """Test that findLambda() returns a scalar."""
        k0 = 1.0
        k1 = 1.05
        lam = model.findLambda(k0, k1)
        assert isinstance(lam, (float, np.floating))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_adjustment_costs(self):
        """Test model with zero adjustment costs."""
        model = Qmod(omega=0)
        model.solve(n_points=10)
        assert model.kss > 0
        
    def test_high_depreciation(self):
        """Test model with high depreciation rate."""
        model = Qmod(delta=0.9)
        model.solve(n_points=10)
        assert model.kss > 0
        
    def test_high_discount_factor(self):
        """Test model with high discount factor."""
        model = Qmod(beta=0.99)
        model.solve(n_points=10)
        assert model.kss > 0
        
    def test_with_investment_tax_credit(self):
        """Test model with investment tax credit."""
        model = Qmod(zeta=0.1)
        model.solve(n_points=10)
        assert model.P == (1 - 0.1)
        assert model.kss > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

