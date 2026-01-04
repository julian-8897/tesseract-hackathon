"""
Tesseract Inverse Problem Demo using Streamlit.

"""

import time
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import streamlit as st
from tesseract_jax import apply_tesseract

from inverse_problem import (
    Tesseract,
    compute_loss,
    get_initial_params,
)

st.set_page_config(
    page_title="Tesseract Inverse Problem Demo",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)


@dataclass
class GradientFlowMetrics:
    """Track Tesseract gradient flow metrics."""

    epoch: int
    vjp_calls: int
    apply_calls: int
    visc_grad_norm: float
    param_grad_norm: float
    loss_value: float
    shapes: Dict[str, tuple]


def initialize_session_state():
    """Initialize all session state variables."""
    if "training" not in st.session_state:
        st.session_state.training = False
    if "trained_viscosity" not in st.session_state:
        st.session_state.trained_viscosity = {}
    if "viscosity_history" not in st.session_state:
        st.session_state.viscosity_history = {}
    if "loss_history" not in st.session_state:
        st.session_state.loss_history = {}
    if "params_flat" not in st.session_state:
        st.session_state.params_flat = {}
    if "epoch_times" not in st.session_state:
        st.session_state.epoch_times = {}
    if "gradient_metrics" not in st.session_state:
        st.session_state.gradient_metrics = []
    if "show_gradient_inspector" not in st.session_state:
        st.session_state.show_gradient_inspector = False


def generate_solution_grid(viscosity, params_flat, pinn, nx=100, nt=50):
    """Generate solution on a grid for visualization."""
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    # Flatten for tesseract evaluation
    x_flat = jnp.array(X.flatten())
    t_flat = jnp.array(T.flatten())

    result = apply_tesseract(
        pinn, {"x": x_flat, "t": t_flat, "params_flat": params_flat}
    )
    # PINN solution evaluation
    u_pred = np.array(result["u_pred"]).reshape(nt, nx)

    # Analytical Solution
    decay = np.exp(-viscosity * (2 * np.pi) ** 2 * T)
    u_analytical = np.sin(2 * np.pi * X) * decay

    return X, T, u_pred, u_analytical


def render_gradient_flow_inspector(backend, gradient_metrics):
    """Render the gradient flow inspector UI."""
    st.markdown("---")
    with st.expander(
        "🔍 **Gradient Flow Inspector** (Tesseract Internals)", expanded=True
    ):
        st.markdown(f"""
        ### Cross-Framework Autodiff Pipeline

        This shows how **Tesseract enables JAX gradients to flow through {backend.upper()}**:
        """)

        # Flow diagram
        if backend == "pytorch":
            st.markdown("""
            ```
            JAX Optimizer (optax)
                    ↓
            jax.grad(compute_loss)
                    ↓
            Tesseract VJP Endpoint  ← Cross-framework boundary!
                    ↓
            PyTorch Autograd (torch.autograd.grad)
                    ↓
            PyTorch PINN forward pass
                    ↓
            Gradients flow back through VJP
                    ↓
            JAX receives gradients  ← Back to JAX!
            ```
            """)
        else:
            st.markdown("""
            ```
            JAX Optimizer (optax)
                    ↓
            jax.grad(compute_loss)
                    ↓
            Tesseract Apply Endpoint
                    ↓
            JAX Autograd (jax.grad)
                    ↓
            JAX PINN forward pass
                    ↓
            Gradients computed natively
            ```
            """)

        if not gradient_metrics:
            st.info("Run training to see gradient flow metrics...")
            return

        # Metrics tabs
        tab1, tab2, tab3 = st.tabs(
            ["📊 Call Statistics", "📈 Gradient Norms", "🔬 Tensor Shapes"]
        )

        with tab1:
            st.subheader("Tesseract API Call Count")

            col1, col2, col3 = st.columns(3)
            latest = gradient_metrics[-1]

            col1.metric(
                "apply() calls per epoch",
                latest.apply_calls,
                help="Forward pass evaluations",
            )
            col2.metric(
                "VJP calls per epoch",
                latest.vjp_calls,
                help="Backward pass (gradient) evaluations",
            )
            col3.metric("Total AD operations", latest.apply_calls + latest.vjp_calls)

            st.info(f"""
            **Key Insight**: Each epoch requires:
            - **{latest.apply_calls} forward passes** (apply): Evaluate u, u_x, u_t, u_xx at collocation points
            - **{latest.vjp_calls} backward passes** (VJP): Compute ∂L/∂ν and ∂L/∂params

            {"**Cross-framework magic**: VJP calls route through PyTorch!" if backend == "pytorch" else "**Native JAX**: All operations stay in JAX"}
            """)

        with tab2:
            st.subheader("Gradient Magnitude Evolution")

            epochs = [m.epoch for m in gradient_metrics]
            visc_grads = [m.visc_grad_norm for m in gradient_metrics]
            param_grads = [m.param_grad_norm for m in gradient_metrics]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Viscosity gradient
            ax1.semilogy(
                epochs, visc_grads, "o-", color="#2ecc71", linewidth=2, markersize=4
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("||∂L/∂ν||")
            ax1.set_title("Viscosity Gradient Norm")
            ax1.grid(alpha=0.3)

            # Parameter gradient
            ax2.semilogy(
                epochs, param_grads, "o-", color="#e74c3c", linewidth=2, markersize=4
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("||∂L/∂params||")
            ax2.set_title("Network Parameter Gradient Norm")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            col1, col2 = st.columns(2)
            col1.metric("Latest ||∂L/∂ν||", f"{visc_grads[-1]:.2e}")
            col2.metric("Latest ||∂L/∂params||", f"{param_grads[-1]:.2e}")

            st.info("""
            **Gradient norms** show the sensitivity of loss to parameters:
            - High norm → steep loss landscape, large updates
            - Decreasing norm → approaching optimum
            - These are computed via Tesseract's VJP (Vector-Jacobian Product)
            """)

        with tab3:
            st.subheader("Tensor Shapes Through Pipeline")

            if latest.shapes:
                st.json(latest.shapes)
            else:
                st.info("No shape information available")

            st.markdown(f"""
            **Data flow through Tesseract**:
            - **Inputs**: x, t (collocation points), params_flat (network weights)
            - **Outputs**: u_pred, u_x, u_t, u_xx (solution + derivatives)
            - All computed via **{backend.upper()} autodiff**, exposed through Tesseract API
            """)


def train_step(
    backend,
    viscosity,
    params_flat,
    visc_opt_state,
    param_opt_state,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    visc_optimizer,
    param_optimizer,
    epoch=0,
    track_gradients=False,
):
    """Single training step with optional gradient flow tracking."""
    import jax

    # Always use the original compute_loss - simplified approach
    grad_visc = jax.grad(compute_loss, argnums=0)
    grad_params = jax.grad(compute_loss, argnums=1)

    # Compute gradients - This triggers VJP calls!
    v_grad = grad_visc(
        viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
    )
    p_grad = grad_params(
        viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
    )

    # Compute gradient norms
    visc_grad_norm = float(jnp.linalg.norm(v_grad))
    param_grad_norm = float(jnp.linalg.norm(p_grad))

    # Update viscosity
    visc_updates, visc_opt_state = visc_optimizer.update(v_grad, visc_opt_state)
    viscosity = optax.apply_updates(viscosity, visc_updates)
    viscosity = jnp.maximum(viscosity, 1e-6)

    # Update params
    param_updates, param_opt_state = param_optimizer.update(p_grad, param_opt_state)
    params_flat = optax.apply_updates(params_flat, param_updates)

    # Compute loss
    loss = compute_loss(
        viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
    )

    metrics = None
    if track_gradients:
        metrics = GradientFlowMetrics(
            epoch=epoch,
            vjp_calls=2,  # One for viscosity grad, one for params grad
            apply_calls=5,  # Data, physics, IC, BC_left, BC_right in compute_loss
            visc_grad_norm=visc_grad_norm,
            param_grad_norm=param_grad_norm,
            loss_value=float(loss),
            shapes={
                "x_obs": tuple(x_obs.shape),
                "t_obs": tuple(t_obs.shape),
                "params_flat": tuple(params_flat.shape),
            },
        )

    return viscosity, params_flat, visc_opt_state, param_opt_state, float(loss), metrics


def main():
    initialize_session_state()

    st.title("Tesseract: Cross-Framework Autodiff Demo")
    st.markdown(
        """
    ### Inverse Problem: Inferring Viscosity from Burgers Equation
    
    This demo showcases **Tesseract's capabilities**: enabling JAX gradients to flow through PyTorch models
    
    **Problem**: Given observed solution data, infer the unknown viscosity $\\nu$ in:
    $$\\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} = \\nu \\frac{\\partial^2 u}{\\partial x^2}$$
    """
    )

    st.sidebar.header("⛭ Configuration")

    backend = st.sidebar.selectbox(
        "Backend",
        ["jax", "pytorch"],
        help="Run the same inverse Burgers pipeline with either a JAX or PyTorch PINN backend.",
    )

    true_viscosity = st.sidebar.slider(
        "rTrue viscosity $\nu$",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="The true viscosity we're trying to infer",
    )

    initial_viscosity = st.sidebar.slider(
        "Initial Guess",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Starting point for optimization",
    )

    n_obs = st.sidebar.slider(
        "Observations",
        min_value=20,
        max_value=200,
        value=100,
        step=20,
        help="Number of observed data points",
    )

    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01,
        help="Standard deviation of observation noise",
    )

    n_epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of training epochs for PINN",
    )

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Optimizer learning rate",
    )

    # Gradient Flow Inspection
    st.sidebar.markdown("---")
    st.sidebar.subheader("⛮ Additional Features")
    show_gradient_inspector = st.sidebar.checkbox(
        "Enable Gradient Flow Inspector",
        value=False,
        help="Track and visualize Tesseract's autodiff operations (slight performance overhead)",
    )

    if st.sidebar.button("Train Model", type="primary"):
        st.session_state.training = True
        st.session_state.gradient_metrics = []

    # Main content
    if st.session_state.training:
        # Setup
        domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        key = jax.random.PRNGKey(123)

        keys = jax.random.split(key, 3)
        x_obs = jax.random.uniform(
            keys[0], (n_obs,), minval=domain["x"][0], maxval=domain["x"][1]
        )
        t_obs = jax.random.uniform(
            keys[1], (n_obs,), minval=0.05, maxval=domain["t"][1]
        )
        decay = jnp.exp(-true_viscosity * (2 * jnp.pi) ** 2 * t_obs)
        u_observed = jnp.sin(2 * jnp.pi * x_obs) * decay
        noise = jax.random.normal(keys[2], (n_obs,)) * noise_level
        u_obs = u_observed + noise

        # Make collocation points
        key_col, key_ic, key_bc = jax.random.split(key, 3)
        n_col = 200
        x_col = jax.random.uniform(
            key_col, (n_col,), minval=domain["x"][0], maxval=domain["x"][1]
        )
        t_col = jax.random.uniform(
            key_col, (n_col,), minval=0.05, maxval=domain["t"][1]
        )

        n_ic = 50
        x_ic = jax.random.uniform(
            key_ic, (n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
        )

        n_bc = 50
        t_bc = jax.random.uniform(key_bc, (n_bc,), minval=0.05, maxval=domain["t"][1])

        # Initialize tesseract
        image_name = "pinn_jax" if backend == "jax" else "pinn_pytorch"
        pinn = Tesseract.from_image(image_name)
        params_flat = get_initial_params(backend)

        # Initialize viscosity
        viscosity = jnp.array(initial_viscosity)

        # Optimizers
        visc_optimizer = optax.adam(learning_rate)
        visc_opt_state = visc_optimizer.init(viscosity)
        param_optimizer = optax.adam(1e-3)
        param_opt_state = param_optimizer.init(params_flat)

        # Training display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Backend", backend.upper())
        with col2:
            st.metric("True Viscosity", f"{true_viscosity:.4f}")
        with col3:
            st.metric("Initial Guess", f"{initial_viscosity:.4f}")

        st.markdown("---")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Metrics display
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_visc = metric_col1.empty()
        metric_error = metric_col2.empty()
        metric_loss = metric_col3.empty()
        metric_time = metric_col4.empty()

        # Plots for tracking performance
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.subheader("Viscosity Convergence")
            visc_chart = st.empty()
        with plot_col2:
            st.subheader("Training Loss")
            loss_chart = st.empty()

        # Initialize histories
        visc_history = [float(initial_viscosity)]
        loss_history = []
        time_history = []

        # Training loop
        with pinn:
            for epoch in range(n_epochs):
                start_time = time.time()

                # Track gradients every 5 epochs (or first 10) if inspector enabled
                track_this_epoch = show_gradient_inspector and (
                    epoch % 5 == 0 or epoch < 10
                )

                (
                    viscosity,
                    params_flat,
                    visc_opt_state,
                    param_opt_state,
                    loss,
                    metrics,
                ) = train_step(
                    backend,
                    viscosity,
                    params_flat,
                    visc_opt_state,
                    param_opt_state,
                    x_obs,
                    t_obs,
                    u_obs,
                    x_col,
                    t_col,
                    x_ic,
                    t_bc,
                    pinn,
                    visc_optimizer,
                    param_optimizer,
                    epoch=epoch,
                    track_gradients=track_this_epoch,
                )

                if metrics and show_gradient_inspector:
                    st.session_state.gradient_metrics.append(metrics)

                epoch_time = time.time() - start_time
                time_history.append(epoch_time)

                visc_val = float(viscosity)
                visc_history.append(visc_val)
                loss_history.append(loss)

                # Update every 5 epochs
                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    error = abs(visc_val - true_viscosity)
                    rel_error = error / true_viscosity * 100

                    # Update progress
                    progress = (epoch + 1) / n_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{n_epochs}")

                    # Update metrics
                    metric_visc.metric(
                        "Current ν",
                        f"{visc_val:.6f}",
                        delta=f"{visc_val - true_viscosity:.6f}",
                    )
                    metric_error.metric("Relative Error", f"{rel_error:.2f}%")
                    metric_loss.metric("Loss", f"{loss:.6f}")
                    metric_time.metric("Epoch Time", f"{epoch_time * 1000:.1f}ms")

                    # Update plots
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    ax1.plot(
                        visc_history, label="Inferred ν", color="#1f77b4", linewidth=2
                    )
                    ax1.axhline(
                        true_viscosity,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"True ν = {true_viscosity}",
                    )
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Viscosity")
                    ax1.legend()
                    ax1.grid(alpha=0.3)
                    visc_chart.pyplot(fig1)
                    plt.close(fig1)

                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.semilogy(loss_history, color="#ff7f0e", linewidth=2)
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Loss (log scale)")
                    ax2.grid(alpha=0.3)
                    loss_chart.pyplot(fig2)
                    plt.close(fig2)

            st.markdown("---")
            st.success("✅ Training Complete!")

            final_visc = float(viscosity)
            final_error = abs(final_visc - true_viscosity) / true_viscosity * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Viscosity", f"{final_visc:.6f}")
            col2.metric("True Viscosity", f"{true_viscosity:.6f}")
            col3.metric("Relative Error", f"{final_error:.2f}%")
            col4.metric("Avg Time/Epoch", f"{np.mean(time_history) * 1000:.1f}ms")

            # Solution visualization
            st.markdown("---")
            st.subheader("🎨 Solution Visualization")

            with st.spinner("Generating solution visualization..."):
                X, T, u_pred, u_analytical = generate_solution_grid(
                    final_visc, params_flat, pinn
                )

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # PINN solution
            im0 = axes[0].contourf(X, T, u_pred, levels=20, cmap="RdBu_r")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("t")
            axes[0].set_title(f"PINN Solution (ν={final_visc:.4f})")
            plt.colorbar(im0, ax=axes[0])

            # Analytical solution
            im1 = axes[1].contourf(X, T, u_analytical, levels=20, cmap="RdBu_r")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("t")
            axes[1].set_title(f"Analytical Solution (ν={true_viscosity:.4f})")
            plt.colorbar(im1, ax=axes[1])

            # Error
            error_map = np.abs(u_pred - u_analytical)
            im2 = axes[2].contourf(X, T, error_map, levels=20, cmap="hot")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("t")
            axes[2].set_title(f"Absolute Error (Max: {error_map.max():.4f})")
            plt.colorbar(im2, ax=axes[2])

            # Add observation points
            axes[0].scatter(
                x_obs, t_obs, c="lime", s=10, alpha=0.5, label="Observations"
            )
            axes[0].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Store results in session state
            st.session_state.trained_viscosity[backend] = final_visc
            st.session_state.viscosity_history[backend] = visc_history
            st.session_state.loss_history[backend] = loss_history
            st.session_state.params_flat[backend] = params_flat
            st.session_state.epoch_times[backend] = time_history

            # Gradient Flow Inspector (if enabled)
            if show_gradient_inspector and st.session_state.gradient_metrics:
                render_gradient_flow_inspector(
                    backend, st.session_state.gradient_metrics
                )

            st.markdown("---")
            st.info(
                """
            **🔑 Key takeaway**

            This demo showcases a **Tesseract-powered inverse Burgers solver**.

            - The same Tesseract pipeline runs with either a JAX or PyTorch PINN backend
            - When using the PyTorch backend, JAX-based gradients are still available via the Tesseract VJP interface
            - Backend choice is just a configuration detail: swap implementations without changing the optimization code
            - Try toggling the backend and see that the Tesseract component behaves identically
            """
            )

        st.session_state.training = False

    else:
        # Initial state - show info
        st.info(
            """
        ◀ Configure parameters in the sidebar and click **"Train Model"** to begin demonstration.
        
        **What this demo does:**
        - Generates synthetic observation data from Burgers equation
        - Uses a Physics-Informed Neural Network (PINN) to infer the unknown viscosity
        - Shows real-time convergence and training metrics
        - Visualizes the learned solution vs analytical solution
        """
        )

        # Show comparison if both backends have been trained
        if (
            "jax" in st.session_state.trained_viscosity
            and "pytorch" in st.session_state.trained_viscosity
        ):
            st.markdown("---")
            st.subheader("📊 Backend Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "JAX PINN", f"{st.session_state.trained_viscosity['jax']:.6f}"
                )
                st.metric(
                    "Avg Time/Epoch",
                    f"{np.mean(st.session_state.epoch_times['jax']) * 1000:.1f}ms",
                )

            with col2:
                st.metric(
                    "PyTorch PINN",
                    f"{st.session_state.trained_viscosity['pytorch']:.6f}",
                )
                st.metric(
                    "Avg Time/Epoch",
                    f"{np.mean(st.session_state.epoch_times['pytorch']) * 1000:.1f}ms",
                )

            # Side-by-side convergence plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                st.session_state.viscosity_history["jax"],
                label="JAX",
                linewidth=2,
                color="#1f77b4",
            )
            ax.plot(
                st.session_state.viscosity_history["pytorch"],
                label="PyTorch",
                linewidth=2,
                color="#ff7f0e",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Inferred Viscosity")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
