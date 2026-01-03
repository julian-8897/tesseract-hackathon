"""
Tesseract Inverse Problem Demo - Interactive Streamlit App

Showcases cross-framework autodiff: JAX gradients flowing through PyTorch PINN!
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import streamlit as st

from inverse_problem import (
    Tesseract,
    compute_loss,
    get_initial_params,
)

st.set_page_config(
    page_title="Tesseract Inverse Problem Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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


def generate_solution_grid(viscosity, params_flat, pinn, nx=100, nt=50):
    """Generate solution on a grid for visualization."""
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    # Flatten for tesseract evaluation
    x_flat = jnp.array(X.flatten())
    t_flat = jnp.array(T.flatten())

    # Evaluate PINN
    from tesseract_jax import apply_tesseract

    result = apply_tesseract(
        pinn, {"x": x_flat, "t": t_flat, "params_flat": params_flat}
    )
    u_pred = np.array(result["u_pred"]).reshape(nt, nx)

    # Also compute analytical solution for comparison
    decay = np.exp(-viscosity * (2 * np.pi) ** 2 * T)
    u_analytical = np.sin(2 * np.pi * X) * decay

    return X, T, u_pred, u_analytical


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
):
    """Single training step."""
    import jax

    # Define gradient functions
    grad_visc = jax.grad(compute_loss, argnums=0)
    grad_params = jax.grad(compute_loss, argnums=1)

    # Compute gradients
    v_grad = grad_visc(
        viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
    )
    p_grad = grad_params(
        viscosity, params_flat, x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc, pinn
    )

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

    return viscosity, params_flat, visc_opt_state, param_opt_state, float(loss)


def main():
    initialize_session_state()

    st.title("🔮 Tesseract: Cross-Framework Autodiff Demo")
    st.markdown(
        """
    ### Inverse Problem: Inferring Viscosity from Burgers Equation
    
    This demo showcases **Tesseract's superpower**: enabling JAX gradients to flow through PyTorch models!
    
    **Problem**: Given observed solution data, infer the unknown viscosity $\\nu$ in:
    $$\\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} = \\nu \\frac{\\partial^2 u}{\\partial x^2}$$
    """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    backend = st.sidebar.selectbox(
        "Backend",
        ["jax", "pytorch"],
        help="Choose which PINN tesseract to use. Same code works with both!",
    )

    true_viscosity = st.sidebar.slider(
        "True Viscosity (ν)",
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
        help="Number of optimization iterations",
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

    # Training button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        st.session_state.training = True

    # Main content
    if st.session_state.training:
        # Setup
        domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        key = jax.random.PRNGKey(123)

        # Generate observations
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

        # Generate collocation points
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

        # Plots
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

                viscosity, params_flat, visc_opt_state, param_opt_state, loss = (
                    train_step(
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
                    )
                )

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

            # Final results
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

            # Key takeaway
            st.markdown("---")
            st.info(
                f"""
            **🔑 Key Takeaway**: 
            
            This demo used the **{backend.upper()}** PINN tesseract. The exact same optimization code works with either `pinn_jax` or `pinn_pytorch`!
            
            - For `pinn_pytorch`: JAX's `jax.grad` computes gradients **through PyTorch** via Tesseract's VJP endpoint
            - This enables seamless model swapping with **zero code changes**
            - Try switching backends and see identical results!
            """
            )

        st.session_state.training = False

    else:
        # Initial state - show info
        st.info(
            """
        👈 Configure parameters in the sidebar and click **"Train Model"** to start!
        
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
