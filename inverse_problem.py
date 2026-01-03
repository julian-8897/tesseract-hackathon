"""
Inverse Problem Demo: JAX PINN vs PyTorch PINN

Demonstrates Tesseract's key capability: cross-framework autodiff.
- Same inverse problem code works with either pinn_jax or pinn_pytorch
- jax.grad flows through PyTorch PINN via Tesseract's VJP endpoint
- Compare training speed between frameworks

Problem: Given observed solution data, infer the unknown viscosity ν
in Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
"""

import sys
import time

import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def get_initial_params(backend="jax"):
    """Get initial parameters for the specified backend."""
    if backend == "jax":
        sys.path.insert(0, "tesseracts/pinn_jax")
        from tesseract_api import PINNNet, flatten_params

        model = PINNNet(jax.random.PRNGKey(42))
        params = flatten_params(model)
        sys.path.pop(0)
        # Clear the imported module to avoid conflicts
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        return jnp.array(params)
    else:  # pytorch
        # For PyTorch, initialize from actual model for proper initialization
        sys.path.insert(0, "tesseracts/pinn_pytorch")
        # Use seed=42 for consistent initialization (matches JAX)
        import torch
        from tesseract_api import PINNNet, flatten_params

        torch.manual_seed(42)
        model = PINNNet(hidden_sizes=[64, 64, 64], n_fourier_features=32, seed=42)
        params = flatten_params(model)
        sys.path.pop(0)
        # Clear the imported module to avoid conflicts
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        return jnp.array(params)


def generate_observations(n_points, true_viscosity, domain, key):
    """
    Generate synthetic observations from analytical solution.

    For Burgers with small viscosity and sinusoidal IC,
    we use heat equation decay as approximation:
    u(x,t) ≈ sin(2πx) * exp(-ν * (2π)² * t)
    """
    keys = jax.random.split(key, 3)

    x = jax.random.uniform(
        keys[0], (n_points,), minval=domain["x"][0], maxval=domain["x"][1]
    )
    t = jax.random.uniform(keys[1], (n_points,), minval=0.05, maxval=domain["t"][1])

    # Analytical solution (heat equation decay)
    decay = jnp.exp(-true_viscosity * (2 * jnp.pi) ** 2 * t)
    u_observed = jnp.sin(2 * jnp.pi * x) * decay

    # Add small noise
    noise = jax.random.normal(keys[2], (n_points,)) * 0.02
    u_observed = u_observed + noise

    return x, t, u_observed


def compute_loss(
    viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
):
    """
    Total PINN loss for inverse problem.

    Components:
    1. Data loss: fit observations
    2. Physics loss: satisfy PDE residual using finite differences
    3. Initial condition loss: u(x, 0) = sin(2πx)
    4. Boundary condition loss: periodic BCs u(0,t) = u(1,t)

    All differentiable w.r.t. viscosity via Tesseract autodiff!
    """
    # ========== Data Loss ==========
    result_obs = apply_tesseract(
        pinn,
        {
            "x": x_obs,
            "t": t_obs,
            "params_flat": params_flat,
        },
    )
    u_pred = result_obs["u_pred"]
    data_loss = jnp.mean((u_pred - u_obs) ** 2)

    # ========== Physics Loss (PDE Residual) ==========
    # Burgers equation: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
    # Get u AND derivatives in ONE call - computed via autodiff inside tesseract!

    result_col = apply_tesseract(
        pinn, {"x": x_col, "t": t_col, "params_flat": params_flat}
    )

    # Extract solution and derivatives (computed via framework's autodiff)
    u_col = result_col["u_pred"]  # Solution values
    u_x = result_col["u_x"]  # ∂u/∂x (exact from autodiff!)
    u_t = result_col["u_t"]  # ∂u/∂t (exact from autodiff!)
    u_xx = result_col["u_xx"]  # ∂²u/∂x² (exact from autodiff!)

    # Burgers residual - now using exact derivatives
    residual = u_t + u_col * u_x - viscosity * u_xx
    physics_loss = jnp.mean(residual**2)

    # ========== Initial Condition Loss ==========
    # u(x, 0) = sin(2πx)
    t_ic = jnp.zeros_like(x_ic)
    result_ic = apply_tesseract(
        pinn,
        {
            "x": x_ic,
            "t": t_ic,
            "params_flat": params_flat,
        },
    )
    u_ic = result_ic["u_pred"]
    u_ic_true = jnp.sin(2 * jnp.pi * x_ic)
    ic_loss = jnp.mean((u_ic - u_ic_true) ** 2)

    # ========== Boundary Condition Loss ==========
    # Periodic BC: u(0, t) = u(1, t)
    x_left = jnp.zeros_like(t_bc)
    x_right = jnp.ones_like(t_bc)

    result_left = apply_tesseract(
        pinn,
        {
            "x": x_left,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    result_right = apply_tesseract(
        pinn,
        {
            "x": x_right,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    u_left = result_left["u_pred"]
    u_right = result_right["u_pred"]
    bc_loss = jnp.mean((u_left - u_right) ** 2)

    # ========== Total Loss ==========
    # Weight the losses: data is primary, physics constrains, IC and BC anchor
    total_loss = data_loss + 0.1 * physics_loss + 0.5 * ic_loss + 0.5 * bc_loss

    return total_loss


def run_inverse_problem(
    backend="jax",
    true_viscosity=0.05,
    initial_viscosity=0.01,
    n_obs=100,
    n_epochs=200,
    learning_rate=0.001,
):
    """
    Run inverse problem to infer viscosity.

    Args:
        backend: "jax" or "pytorch" - which PINN tesseract to use
    """
    domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}

    print(f"\n{'=' * 60}")
    print(f"  Inverse Problem: {backend.upper()} PINN")
    print(f"{'=' * 60}")
    print(f"True viscosity:    ν = {true_viscosity}")
    print(f"Initial guess:     ν = {initial_viscosity}")

    # Generate observations
    key = jax.random.PRNGKey(123)
    x_obs, t_obs, u_obs = generate_observations(n_obs, true_viscosity, domain, key)

    # Generate collocation points for physics loss
    key_col, key_ic, key_bc = jax.random.split(key, 3)
    n_col = 200  # Physics residual points
    x_col = jax.random.uniform(
        key_col, (n_col,), minval=domain["x"][0], maxval=domain["x"][1]
    )
    t_col = jax.random.uniform(key_col, (n_col,), minval=0.05, maxval=domain["t"][1])

    # Initial condition points
    n_ic = 50
    x_ic = jax.random.uniform(
        key_ic, (n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
    )

    # Boundary condition points
    n_bc = 50
    t_bc = jax.random.uniform(key_bc, (n_bc,), minval=0.05, maxval=domain["t"][1])

    # Initialize tesseract
    image_name = "pinn_jax" if backend == "jax" else "pinn_pytorch"
    pinn = Tesseract.from_image(image_name)

    # Get initial parameters
    params_flat = get_initial_params(backend)
    print(f"Model parameters: {params_flat.size}")

    # Initialize viscosity as learnable parameter
    viscosity = jnp.array(initial_viscosity)

    # Optimizers
    visc_optimizer = optax.adam(learning_rate)
    visc_opt_state = visc_optimizer.init(viscosity)

    param_optimizer = optax.adam(1e-3)
    param_opt_state = param_optimizer.init(params_flat)

    # Gradient functions - THIS IS WHERE TESSERACT AUTODIFF SHINES!
    # jax.grad will compute gradients that flow THROUGH the tesseract
    # For PyTorch PINN, this means JAX gradients flow through PyTorch!
    grad_visc = jax.grad(compute_loss, argnums=0)
    grad_params = jax.grad(compute_loss, argnums=1)

    with pinn:
        print(f"\n✓ {backend.upper()} PINN tesseract ready")
        print("\nOptimizing...")
        print("-" * 60)

        times = []
        viscosity_history = [float(viscosity)]

        for epoch in range(n_epochs):
            start_time = time.time()

            # Compute gradients
            # For pinn_pytorch, this calls the VJP endpoint we implemented!
            v_grad = grad_visc(
                viscosity,
                params_flat,
                x_obs,
                t_obs,
                u_obs,
                x_col,
                t_col,
                x_ic,
                t_bc,
                pinn,
            )
            p_grad = grad_params(
                viscosity,
                params_flat,
                x_obs,
                t_obs,
                u_obs,
                x_col,
                t_col,
                x_ic,
                t_bc,
                pinn,
            )

            # Update viscosity
            visc_updates, visc_opt_state = visc_optimizer.update(v_grad, visc_opt_state)
            viscosity = optax.apply_updates(viscosity, visc_updates)
            viscosity = jnp.maximum(viscosity, 1e-6)  # Keep positive

            # Update PINN params
            param_updates, param_opt_state = param_optimizer.update(
                p_grad, param_opt_state
            )
            params_flat = optax.apply_updates(params_flat, param_updates)

            epoch_time = time.time() - start_time
            times.append(epoch_time)
            viscosity_history.append(float(viscosity))

            if epoch % 20 == 0 or epoch == n_epochs - 1:
                loss = compute_loss(
                    viscosity,
                    params_flat,
                    x_obs,
                    t_obs,
                    u_obs,
                    x_col,
                    t_col,
                    x_ic,
                    t_bc,
                    pinn,
                )
                error = abs(float(viscosity) - true_viscosity)
                print(
                    f"Epoch {epoch:4d} | Loss: {float(loss):.6f} | "
                    f"ν: {float(viscosity):.6f} | Error: {error:.6f} | "
                    f"Time: {epoch_time * 1000:.1f}ms"
                )

        print("-" * 60)

        # Results
        final_viscosity = float(viscosity)
        relative_error = abs(final_viscosity - true_viscosity) / true_viscosity * 100
        avg_time = sum(times) / len(times) * 1000

        print("\nResults:")
        print(f"  Inferred ν:     {final_viscosity:.6f}")
        print(f"  True ν:         {true_viscosity:.6f}")
        print(f"  Relative error: {relative_error:.2f}%")
        print(f"  Avg time/epoch: {avg_time:.1f}ms")

    return {
        "backend": backend,
        "final_viscosity": final_viscosity,
        "true_viscosity": true_viscosity,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
    }


def compare_backends(n_epochs=50, n_obs=80):
    """Run inverse problem with both backends and compare."""

    print("\n" + "=" * 70)
    print("  CROSS-FRAMEWORK AUTODIFF DEMO")
    print("  Tesseract enables jax.grad through PyTorch!")
    print("=" * 70)

    results = {}

    # Run JAX PINN
    results["jax"] = run_inverse_problem(
        backend="jax",
        true_viscosity=0.05,
        initial_viscosity=0.01,
        n_obs=n_obs,
        n_epochs=n_epochs,
    )

    # Run PyTorch PINN
    results["pytorch"] = run_inverse_problem(
        backend="pytorch",
        true_viscosity=0.05,
        initial_viscosity=0.01,
        n_obs=n_obs,
        n_epochs=n_epochs,
    )

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON: JAX vs PyTorch PINN")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'JAX':>15} {'PyTorch':>15}")
    print("-" * 55)
    print(
        f"{'Inferred viscosity':<25} {results['jax']['final_viscosity']:>15.6f} {results['pytorch']['final_viscosity']:>15.6f}"
    )
    print(
        f"{'Relative error (%)':<25} {results['jax']['relative_error']:>15.2f} {results['pytorch']['relative_error']:>15.2f}"
    )
    print(
        f"{'Avg time/epoch (ms)':<25} {results['jax']['avg_time_ms']:>15.1f} {results['pytorch']['avg_time_ms']:>15.1f}"
    )

    speedup = results["pytorch"]["avg_time_ms"] / results["jax"]["avg_time_ms"]
    if speedup > 1:
        print(f"\n→ JAX is {speedup:.1f}x faster than PyTorch")
    else:
        print(f"\n→ PyTorch is {1 / speedup:.1f}x faster than JAX")

    print("\n" + "=" * 70)
    print("  KEY TAKEAWAY")
    print("=" * 70)
    print("""
  ✓ Same inverse problem code works with BOTH backends
  ✓ jax.grad computes gradients through PyTorch via Tesseract VJP
  ✓ Swap models with one line: Tesseract.from_image("pinn_jax" or "pinn_pytorch")
  ✓ This is impossible without Tesseract's cross-framework autodiff!
""")

    return results


def run_single_backend(backend="jax", n_epochs=50):
    """Run inverse problem with a single backend (for testing)."""
    return run_inverse_problem(
        backend=backend,
        true_viscosity=0.05,
        initial_viscosity=0.01,
        n_obs=80,
        n_epochs=n_epochs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inverse Problem Demo")
    parser.add_argument(
        "--backend",
        choices=["jax", "pytorch", "both"],
        default="both",
        help="Which backend to use",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    if args.backend == "both":
        results = compare_backends(n_epochs=args.epochs)
    else:
        results = run_single_backend(backend=args.backend, n_epochs=args.epochs)
