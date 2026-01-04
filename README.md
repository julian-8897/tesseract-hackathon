# 🧊 Tesseract Cross-Framework Autodiff Demo

[![Tesseract Version](https://img.shields.io/badge/tesseract--core-v1.2.0-blue)](https://github.com/pasteurlabs/tesseract-core)
[![tesseract-jax](https://img.shields.io/badge/tesseract--jax-v0.2.3-green)](https://github.com/pasteurlabs/tesseract-jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

> **Inverse Problem: Inferring Viscosity from Burgers Equation using Physics-Informed Neural Networks**

This project demonstrates Tesseract's breakthrough capability: **cross-framework automatic differentiation**. Train inverse problems using JAX optimizers while the neural network runs in PyTorch — gradients flow seamlessly across framework boundaries via Tesseract's VJP endpoint.

---

## 🎯 Tesseract Features Demonstrated

| Feature | Description |
|---------|-------------|
| **Cross-Framework Autodiff** | JAX `jax.grad` computes gradients through PyTorch models |
| **VJP Endpoint** | Custom `vector_jacobian_product` for reverse-mode AD |
| **JVP Endpoint** | Custom `jacobian_vector_product` for forward-mode AD |
| **Differentiable Annotations** | Schema-based autodiff with `Differentiable[Array[...]]` |
| **Backend Swapping** | Same optimization code works with JAX or PyTorch |
| **tesseract-jax Integration** | Seamless `apply_tesseract` with JAX transforms |
| **Higher-Order Derivatives** | Gradients through second-order derivatives (u_xx) |

---

## 📋 Table of Contents

- [Tesseract Features Demonstrated](#-tesseract-features-demonstrated)
- [Quick Start](#-quick-start)
- [Motivation](#-motivation)
- [The Problem We Solve](#-the-problem-we-solve)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Demo](#-running-the-demo)
- [Understanding the Code](#-understanding-the-code)
- [Key Tesseract Concepts](#-key-tesseract-concepts)
- [Results](#-results)
- [Compatibility](#-compatibility)
- [Resources](#-resources)
- [License](#-license)
- [Contact & Contributing](#-contact--contributing)

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/julian-8897/tesseract-hackathon.git
cd tesseract-hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Build Tesseracts (requires Docker running)
./buildall.sh

# 3. Run the demo
python inverse_problem.py --backend both --epochs 50

# 4. Or launch interactive app
streamlit run app.py
```

---

## 🎯 Motivation

### The Cross-Framework Problem

In scientific machine learning, researchers often face a dilemma:

- **JAX** excels at functional transformations, JIT compilation, and composable autodiff (`jax.grad`, `jax.vmap`)
- **PyTorch** has the largest ecosystem of pre-trained models, debugging tools, and community support

But what if you want to:
- Use a PyTorch model inside a JAX optimization loop?
- Compute `jax.grad` through a PyTorch neural network?
- Swap backends without rewriting your optimization code?

**This is impossible with standard tools** — until Tesseract.

### What Tesseract Enables

Tesseract wraps models as **containerized, differentiable microservices** with standardized autodiff endpoints:

| Endpoint | Purpose |
|----------|---------|
| `apply` | Forward pass: inputs → outputs |
| `VJP` | Vector-Jacobian Product (reverse-mode AD) |
| `JVP` | Jacobian-Vector Product (forward-mode AD) |

This means:
- ✅ JAX can call `jax.grad` on a function that internally uses PyTorch
- ✅ Gradients flow through the Tesseract container via the VJP endpoint
- ✅ Same optimization code works with any backend

---

## 🔬 The Problem We Solve

### Inverse Problem: Parameter Inference in PDEs

Given **noisy observations** of a physical system, infer the **unknown parameters** governing its dynamics.

**Specific Problem**: Infer the viscosity $\nu$ in the 1D Burgers equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

Where:
- $u(x, t)$ is the velocity field
- $\nu$ is the kinematic viscosity (unknown parameter to infer)
- Initial condition: $u(x, 0) = \sin(2\pi x)$
- Boundary conditions: periodic on $[0, 1]$

### Why This Problem?

1. **Requires derivatives**: The PDE residual needs $\partial u/\partial t$, $\partial u/\partial x$, $\partial^2 u/\partial x^2$
2. **Requires gradient-based optimization**: We minimize a loss w.r.t. both network weights and $\nu$
3. **Showcases higher-order AD**: Gradients flow through derivative computations
4. **Real-world relevance**: Parameter inference is fundamental in physics, biology, finance

---

## ⚙️ How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      JAX Optimization Loop                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ optax.adam  │───▶│  jax.grad   │───▶│   compute_loss()    │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                    │             │
│                                          ┌─────────▼─────────┐  │
│                                          │  apply_tesseract  │  │
│                                          └─────────┬─────────┘  │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
                        ┌────────────────────────────▼────────────────────────────┐
                        │                   Tesseract Container                    │
                        │  ┌──────────────────────────────────────────────────┐   │
                        │  │  PINN (JAX/Equinox OR PyTorch)                   │   │
                        │  │  • Forward: (x, t, params) → u, u_x, u_t, u_xx   │   │
                        │  │  • VJP: cotangents → gradients                   │   │
                        │  └──────────────────────────────────────────────────┘   │
                        └─────────────────────────────────────────────────────────┘
```

### Gradient Flow (Cross-Framework Magic)

When using the **PyTorch backend** with JAX optimization:

```
jax.grad(compute_loss)
        ↓
Tesseract VJP endpoint called
        ↓
PyTorch autograd computes gradients
        ↓
Gradients returned to JAX
        ↓
JAX optimizer updates parameters
```

This is **impossible without Tesseract** — normally JAX cannot differentiate through PyTorch.

### Loss Function Components

The PINN loss has four terms:

| Component | Formula | Purpose |
|-----------|---------|---------|
| Data Loss | $\|u_{pred} - u_{obs}\|^2$ | Fit observations |
| Physics Loss | $\|u_t + u \cdot u_x - \nu \cdot u_{xx}\|^2$ | Satisfy PDE |
| IC Loss | $\|u(x, 0) - \sin(2\pi x)\|^2$ | Match initial condition |
| BC Loss | $\|u(0, t) - u(1, t)\|^2$ | Enforce periodic BCs |

**Total Loss**: $\mathcal{L} = \mathcal{L}_{data} + 0.1 \cdot \mathcal{L}_{physics} + 0.5 \cdot \mathcal{L}_{IC} + 0.5 \cdot \mathcal{L}_{BC}$

---

## 📁 Project Structure

```
tesseract-hackathon/
├── app.py                      # Streamlit interactive demo
├── inverse_problem.py          # Core inverse problem implementation
├── buildall.sh                 # Build all Tesseract containers
├── pyproject.toml              # Python dependencies
├── requirements.txt            # Minimal requirements
│
└── tesseracts/                 # Tesseract definitions
    ├── pinn_jax/
    │   ├── tesseract_api.py        # JAX/Equinox PINN implementation
    │   ├── tesseract_config.yaml   # Tesseract metadata
    │   └── tesseract_requirements.txt
    │
    └── pinn_pytorch/
        ├── tesseract_api.py        # PyTorch PINN implementation
        ├── tesseract_config.yaml   # Tesseract metadata
        └── tesseract_requirements.txt
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `inverse_problem.py` | CLI demo: runs inverse problem with JAX/PyTorch backends |
| `app.py` | Streamlit app with live training visualization & gradient flow inspector |
| `tesseracts/pinn_jax/tesseract_api.py` | PINN in JAX/Equinox with Fourier features |
| `tesseracts/pinn_pytorch/tesseract_api.py` | Same PINN architecture in PyTorch |

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | 3.13 recommended |
| Docker | Latest | [Docker Desktop](https://docs.docker.com/desktop/) |
| uv / pip | Latest | Package manager |

### Step 1: Clone the Repository

```bash
git clone https://github.com/julian-8897/tesseract-hackathon.git
cd tesseract-hackathon
```

### Step 2: Create Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate

# Or using standard venv
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

### Step 4: Build Tesseract Containers

```bash
# Make sure Docker is running!
./buildall.sh
```

This builds two Docker images:
- `pinn_jax` — JAX/Equinox PINN
- `pinn_pytorch` — PyTorch PINN

**Expected output:**
```
=========================================
Building Tesseract Hackathon Template
=========================================

Building tesseracts/pinn_jax/
✓ tesseracts/pinn_jax/ built successfully

Building tesseracts/pinn_pytorch/
✓ tesseracts/pinn_pytorch/ built successfully

=========================================
✓ All tesseracts built successfully!
=========================================
```

### Step 5: Verify Installation

```bash
# List built tesseracts
docker images | grep pinn

# Expected:
# pinn_jax       latest    ...
# pinn_pytorch   latest    ...
```

---

## 🏃 Running the Demo

### Option 1: Command-Line Demo

```bash
# Run with both backends (comparison mode)
python inverse_problem.py --backend both --epochs 100

# Run with JAX only
python inverse_problem.py --backend jax --epochs 50

# Run with PyTorch only
python inverse_problem.py --backend pytorch --epochs 50
```

### Option 2: Streamlit Interactive App

```bash
streamlit run app.py
```

**Features:**
- 🎛️ Adjust viscosity, noise, learning rate via sliders
- 📊 Real-time loss and viscosity convergence plots
- 🔍 Gradient Flow Inspector showing Tesseract internals
- 🎨 Solution visualization (PINN vs analytical)

---

## 📖 Understanding the Code

### 1. Tesseract API Structure (`tesseract_api.py`)

Each Tesseract must define:

```python
# Input/Output schemas (Pydantic models)
class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]]      # Spatial coords
    t: Differentiable[Array[(None,), Float32]]      # Time coords
    params_flat: Differentiable[Array[(None,), Float32]]  # Network params

class OutputSchema(BaseModel):
    u_pred: Differentiable[Array[(None,), Float32]]  # Solution
    u_x: Differentiable[Array[(None,), Float32]]     # ∂u/∂x
    u_t: Differentiable[Array[(None,), Float32]]     # ∂u/∂t
    u_xx: Differentiable[Array[(None,), Float32]]    # ∂²u/∂x²

# Required endpoints
def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass"""
    ...

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    """Backward pass for reverse-mode AD"""
    ...

def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    """Forward-mode AD"""
    ...
```

### 2. PINN Architecture

Both backends use identical architecture:

```
Input: (x, t) ∈ ℝ²
    ↓
Fourier Feature Encoding (reduces spectral bias)
    → [x, t, sin(x·B_x), cos(x·B_x), sin(t·B_t), cos(t·B_t)]
    → ℝ^(4·n_fourier + 2) = ℝ^130 (default)
    ↓
MLP: 130 → 64 → 64 → 64 → 1
    (tanh activations)
    ↓
Output: u(x, t) ∈ ℝ
```

### 3. Calling Tesseract from JAX

```python
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

# Load tesseract (starts Docker container)
pinn = Tesseract.from_image("pinn_jax")  # or "pinn_pytorch"

with pinn:
    # Forward pass
    result = apply_tesseract(pinn, {
        "x": x_points,
        "t": t_points,
        "params_flat": params
    })
    
    # result contains: u_pred, u_x, u_t, u_xx
    
    # Compute gradients (calls VJP automatically!)
    grad_fn = jax.grad(loss_function)
    grads = grad_fn(viscosity, params, ...)
```

### 4. The Magic: Cross-Framework Gradients

In `inverse_problem.py`:

```python
# This works even when pinn is PyTorch!
grad_visc = jax.grad(compute_loss, argnums=0)
grad_params = jax.grad(compute_loss, argnums=1)

# compute_loss calls apply_tesseract internally
# When backend="pytorch", jax.grad triggers Tesseract's VJP endpoint
# which calls PyTorch's autograd
v_grad = grad_visc(viscosity, params_flat, ..., pinn)
```

---

## 🔑 Key Tesseract Concepts

### Differentiable Annotations

```python
x: Differentiable[Array[(None,), Float32]]
```
- Tells Tesseract this field participates in autodiff
- `None` in shape means dynamic batch dimension

### VJP (Vector-Jacobian Product)

- Implements **reverse-mode autodiff** (backpropagation)
- Given cotangents (∂L/∂outputs), computes ∂L/∂inputs
- Essential for gradient-based optimization

```python
def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    # vjp_inputs: which inputs need gradients ("x", "t", "params_flat")
    # vjp_outputs: which outputs have cotangents ("u_pred", "u_x", ...)
    # cotangent_vector: ∂L/∂output for each output
    
    # Returns: ∂L/∂input for each requested input
```

### JVP (Jacobian-Vector Product)

- Implements **forward-mode autodiff**
- Given tangents (perturbations in inputs), computes output perturbations
- Useful for sensitivity analysis, Hessian-vector products

### Higher-Order Derivatives

The PINN computes u_x, u_t, u_xx using autodiff inside the Tesseract:

```python
# In apply():
u_x = jax.grad(u_fn, argnums=0)(x, t)  # ∂u/∂x
u_xx = jax.grad(u_x_fn, argnums=0)(x, t)  # ∂²u/∂x²
```

When we call `jax.grad(compute_loss)`, gradients flow **through** these derivative computations — this is higher-order AD in action!

---

## 📊 Results

### Convergence Comparison

| Metric | JAX Backend | PyTorch Backend |
|--------|-------------|-----------------|
| True viscosity | 0.05 | 0.05 |
| Inferred viscosity | ~0.0489 | ~0.0492 |
| Relative error | <3% | <3% |
| Avg time/epoch | ~150ms | ~180ms |

### Key Takeaways

1. ✅ **Same code, swappable backends**: Change one line to switch JAX ↔ PyTorch
2. ✅ **Exact gradients**: Tesseract uses native autodiff, not finite differences
3. ✅ **Cross-framework AD works**: JAX optimizer + PyTorch model = no problem
4. ✅ **Containerized & portable**: Deploy anywhere Docker runs

---

## Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| tesseract-core | 1.2.0 | Tesseract runtime and CLI |
| tesseract-jax | 0.2.3 | JAX integration layer |
| Python | ≥3.10 | Required for type hints |
| Docker | Latest | Required for container execution |
| JAX | 0.8.2 | With CPU backend |
| PyTorch | 2.9.1 | For PyTorch PINN backend |
| Equinox | 0.13.2 | For JAX PINN (Eqx modules) |

**Tested on**: macOS (Apple Silicon)

---

## Additional Resources

### Tesseract Documentation
- [Tesseract Core GitHub](https://github.com/pasteurlabs/tesseract-core)
- [Tesseract-JAX GitHub](https://github.com/pasteurlabs/tesseract-jax)
- [Creating Tesseracts](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html)
- [Differentiable Programming Guide](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html)
- [Tesseract Endpoints Reference](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html)

### Community
- [Tesseract Forums](https://si-tesseract.discourse.group/)
- [Showcase Gallery](https://si-tesseract.discourse.group/c/showcase/11)

### Background Reading
- [Physics-Informed Neural Networks (Raissi et al.)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Fourier Features for PINNs](https://arxiv.org/abs/2006.10739)
- [Simulation Intelligence (SI) Whitepaper](https://arxiv.org/abs/2112.03235)

---

## License

Licensed under **[Apache License 2.0](LICENSE)**.

---

## Acknowledgments

Built for the [Tesseract Hackathon](https://pasteurlabs.ai/tesseract-hackathon-2025/#overview) by Pasteur Labs.

---

