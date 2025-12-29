"""
Test Script for PDE Solver Tesseracts

Tests the spatial_derivatives and burgers_loss tesseracts with:
1. Sine wave test (known analytical derivatives)
2. Gaussian pulse test
3. Combined pipeline test (derivatives → loss)
"""

import jax.numpy as jnp
from tesseract_core import Tesseract


def test_spatial_derivatives():
    """Test spatial derivatives with sine wave (known analytical solution)."""
    print("\n" + "=" * 70)
    print("  TEST 1: Spatial Derivatives with Sine Wave")
    print("=" * 70)

    # Create test data: u = sin(x), du/dx = cos(x), d²u/dx² = -sin(x)
    n_points = 100
    x = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    u = jnp.sin(x)
    dx = x[1] - x[0]

    # Analytical solutions
    du_dx_exact = jnp.cos(x)
    d2u_dx2_exact = -jnp.sin(x)

    print(f"\nGrid: {n_points} points, dx = {dx:.6f}")
    print("Domain: [0, 2π] with periodic BC")

    # Test with spatial derivatives tesseract
    spatial_deriv = Tesseract.from_image("spatial_derivatives")

    with spatial_deriv:
        # Test 2nd order
        result_2nd = spatial_deriv.apply(
            {"u": u, "dx": dx, "bc": "periodic", "order": 2, "compute_second": True}
        )

        # Compute errors
        error_du_2nd = jnp.sqrt(jnp.mean((result_2nd["du_dx"] - du_dx_exact) ** 2))
        error_d2u_2nd = jnp.sqrt(jnp.mean((result_2nd["d2u_dx2"] - d2u_dx2_exact) ** 2))

        print("\n2nd Order Accuracy:")
        print(f"  du/dx RMSE:   {error_du_2nd:.6e}")
        print(f"  d²u/dx² RMSE: {error_d2u_2nd:.6e}")

        # Test 4th order
        result_4th = spatial_deriv.apply(
            {"u": u, "dx": dx, "bc": "periodic", "order": 4, "compute_second": True}
        )

        error_du_4th = jnp.sqrt(jnp.mean((result_4th["du_dx"] - du_dx_exact) ** 2))
        error_d2u_4th = jnp.sqrt(jnp.mean((result_4th["d2u_dx2"] - d2u_dx2_exact) ** 2))

        print("\n4th Order Accuracy:")
        print(f"  du/dx RMSE:   {error_du_4th:.6e}")
        print(f"  d²u/dx² RMSE: {error_d2u_4th:.6e}")

        print(f"\nAccuracy Improvement: {error_du_2nd / error_du_4th:.2f}x better")

    return result_2nd, result_4th


def test_burgers_loss():
    """Test Burgers loss computation with known solution."""
    print("\n" + "=" * 70)
    print("  TEST 2: Burgers Loss Computation")
    print("=" * 70)

    # Create test solution: traveling wave
    n_points = 100
    x = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    dx = x[1] - x[0]

    # Simple wave solution
    u = 0.5 + 0.3 * jnp.sin(x)

    print("\nTest solution: u = 0.5 + 0.3*sin(x)")
    print(f"Grid: {n_points} points")
    print("Viscosity: ν = 0.01")

    # Compute derivatives
    spatial_deriv = Tesseract.from_image("spatial_derivatives")
    burgers = Tesseract.from_image("burgers_loss")

    with spatial_deriv, burgers:
        # Get derivatives
        derivs = spatial_deriv.apply(
            {"u": u, "dx": dx, "bc": "periodic", "order": 2, "compute_second": True}
        )

        print("\nDerivatives computed:")
        print(
            f"  du/dx range: [{derivs['du_dx'].min():.4f}, {derivs['du_dx'].max():.4f}]"
        )
        print(
            f"  d²u/dx² range: [{derivs['d2u_dx2'].min():.4f}, {derivs['d2u_dx2'].max():.4f}]"
        )

        # Compute Burgers loss
        loss_result = burgers.apply(
            {
                "u": u,
                "du_dx": derivs["du_dx"],
                "d2u_dx2": derivs["d2u_dx2"],
                "viscosity": jnp.array(0.01),
            }
        )

        print("\nBurgers Loss Results:")
        print(f"  Physics loss: {loss_result['physics_loss']:.6e}")
        print(f"  Data loss:    {loss_result['data_loss']:.6e}")
        print(f"  Total loss:   {loss_result['total_loss']:.6e}")
        print(
            f"  Residual range: [{loss_result['residual'].min():.6f}, {loss_result['residual'].max():.6f}]"
        )

    return loss_result


def test_combined_pipeline():
    """Test the complete pipeline: solution → derivatives → loss."""
    print("\n" + "=" * 70)
    print("  TEST 3: Complete Pipeline Test")
    print("=" * 70)

    # Create a Gaussian pulse
    n_points = 128
    x = jnp.linspace(0, 10, n_points, endpoint=False)
    dx = x[1] - x[0]

    # Gaussian pulse centered at x=5
    u = jnp.exp(-((x - 5) ** 2) / 0.5)

    print("\nTest: Gaussian pulse")
    print(f"Grid: {n_points} points, dx = {dx:.6f}")
    print("Domain: [0, 10] with periodic BC")

    spatial_deriv = Tesseract.from_image("spatial_derivatives")
    burgers = Tesseract.from_image("burgers_loss")

    with spatial_deriv, burgers:
        # Different viscosity values
        viscosities = [0.001, 0.01, 0.1]

        print("\nTesting different viscosities:")
        for nu in viscosities:
            # Compute derivatives
            derivs = spatial_deriv.apply(
                {"u": u, "dx": dx, "bc": "periodic", "order": 4, "compute_second": True}
            )

            # Compute loss
            loss_result = burgers.apply(
                {
                    "u": u,
                    "du_dx": derivs["du_dx"],
                    "d2u_dx2": derivs["d2u_dx2"],
                    "viscosity": jnp.array(nu),
                }
            )

            print(f"  ν = {nu:.3f}: physics_loss = {loss_result['physics_loss']:.6e}")

    print("\n✓ Pipeline test completed successfully!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  TESSERACT PDE SOLVER - Test Suite")
    print("=" * 70)
    print("\nTesting spatial_derivatives and burgers_loss tesseracts...")

    try:
        # Run individual tests
        test_spatial_derivatives()
        test_burgers_loss()
        test_combined_pipeline()

        print("\n" + "=" * 70)
        print("  ✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Implement neural_operator tesseract")
        print("  2. Create training pipeline with gradient descent")
        print("  3. Generate Burgers equation datasets")
        print("\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
