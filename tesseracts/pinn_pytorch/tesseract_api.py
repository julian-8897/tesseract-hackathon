# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PINN (Physics-Informed Neural Network) Tesseract API.
Same architecture as PINN JAX version but in PyTorch.
Maps (x, t) → u(x, t) for solving PDEs.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    """Input schema for PINN."""

    x: Differentiable[Array[(None,), Float32]] = Field(
        description="Spatial coordinates (collocation points)"
    )
    t: Differentiable[Array[(None,), Float32]] = Field(
        description="Time coordinates (same shape as x)"
    )
    params_flat: Differentiable[Array[(None,), Float32]] = Field(
        description="Flattened neural network parameters"
    )


class OutputSchema(BaseModel):
    """Output schema for PINN."""

    u_pred: Differentiable[Array[(None,), Float32]] = Field(
        description="Predicted solution u(x, t)"
    )
    u_x: Differentiable[Array[(None,), Float32]] = Field(
        description="Spatial derivative ∂u/∂x computed via autodiff"
    )
    u_t: Differentiable[Array[(None,), Float32]] = Field(
        description="Time derivative ∂u/∂t computed via autodiff"
    )
    u_xx: Differentiable[Array[(None,), Float32]] = Field(
        description="Second spatial derivative ∂²u/∂x² computed via autodiff"
    )


class PINNNet(nn.Module):
    """Simple MLP for PINN with Fourier feature encoding (PyTorch)."""

    def __init__(self, hidden_sizes=[64, 64, 64], n_fourier_features=32, seed=0):
        """Initialize network with Fourier features for better convergence."""
        super().__init__()

        torch.manual_seed(seed)

        self.register_buffer("B_x", torch.randn(n_fourier_features) * 2.0)
        self.register_buffer("B_t", torch.randn(n_fourier_features) * 2.0)

        # Input: 2*n_fourier_features (sin + cos for x and t) + 2 (raw x, t)
        input_dim = 4 * n_fourier_features + 2
        layer_sizes = [input_dim] + hidden_sizes + [1]

        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t):
        """Forward pass for batch of points."""
        # x, t: (batch_size,)

        # Fourier feature encoding
        x_proj = x.unsqueeze(-1) * self.B_x  # (batch, n_fourier)
        t_proj = t.unsqueeze(-1) * self.B_t  # (batch, n_fourier)

        # Concatenate: [x, t, sin(x*B), cos(x*B), sin(t*B), cos(t*B)]
        features = torch.cat(
            [
                x.unsqueeze(-1),
                t.unsqueeze(-1),
                torch.sin(x_proj),
                torch.cos(x_proj),
                torch.sin(t_proj),
                torch.cos(t_proj),
            ],
            dim=-1,
        )  # (batch, input_dim)

        h = features
        for layer in self.layers[:-1]:
            h = layer(h)
            h = torch.tanh(h)

        return self.layers[-1](h).squeeze(-1)


def flatten_params(model):
    """Flatten model parameters to a single numpy array."""
    params = []
    for p in model.parameters():
        params.append(p.detach().cpu().numpy().flatten())
    # Include buffers (B_x, B_t)
    for b in model.buffers():
        params.append(b.detach().cpu().numpy().flatten())
    return np.concatenate(params).astype(np.float32)


def unflatten_params(params_flat, reference_model=None):
    """Unflatten parameters back to model, returning new model."""
    if reference_model is None:
        reference_model = PINNNet()

    # Create new model with same architecture
    model = PINNNet(hidden_sizes=[64, 64, 64], n_fourier_features=32, seed=0)

    params_flat = np.array(params_flat)
    start = 0

    # Load parameters
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        size = param.numel()
        state_dict[name] = torch.tensor(
            params_flat[start : start + size].reshape(param.shape), dtype=torch.float32
        )
        start += size

    # Load buffers (B_x, B_t)
    for name, buf in model.named_buffers():
        size = buf.numel()
        state_dict[name] = torch.tensor(
            params_flat[start : start + size].reshape(buf.shape), dtype=torch.float32
        )
        start += size

    model.load_state_dict(state_dict)
    return model


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply PINN - PyTorch forward pass with derivatives via autodiff."""
    x = np.array(inputs.x)
    t = np.array(inputs.t)
    params_flat = np.array(inputs.params_flat)

    # Unflatten parameters to model
    model = unflatten_params(params_flat)
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True)

    u_pred = model(x_tensor, t_tensor)

    # Compute spatial/temporal derivatives via PyTorch autodiff
    u_x = torch.autograd.grad(
        outputs=u_pred,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_t = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(u_x),
        create_graph=False,
        retain_graph=False,
    )[0]

    return {
        "u_pred": u_pred.detach().numpy(),
        "u_x": u_x.detach().numpy(),
        "u_t": u_t.detach().numpy(),
        "u_xx": u_xx.detach().numpy(),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """
    Compute VJP (backward pass) for Tesseract autodiff.

    Enables jax.grad to flow through PyTorch models.
    """
    x = np.array(inputs.x)
    t = np.array(inputs.t)
    params_flat = np.array(inputs.params_flat)

    # Unflatten parameters to model
    model = unflatten_params(params_flat)
    model.train()  # Enable gradients

    # Convert to tensors with gradient tracking
    # Always need requires_grad=True for forward pass to compute derivatives
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True)

    # For params, we need gradients
    if "params_flat" in vjp_inputs:
        for p in model.parameters():
            p.requires_grad_(True)

    # Forward pass - compute ALL outputs like in apply()
    u_pred = model(x_tensor, t_tensor)

    # Compute derivatives if needed (they're part of the outputs)
    u_x = torch.autograd.grad(
        outputs=u_pred,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_t = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Backward pass with cotangent vectors for ALL outputs
    # Collect all cotangents and outputs
    outputs_dict = {"u_pred": u_pred, "u_x": u_x, "u_t": u_t, "u_xx": u_xx}

    # Only backward through outputs that are requested
    for output_name in vjp_outputs:
        if output_name in outputs_dict and output_name in cotangent_vector:
            cotangent = torch.tensor(
                np.array(cotangent_vector[output_name]), dtype=torch.float32
            )
            # Accumulate gradients from each output
            outputs_dict[output_name].backward(cotangent, retain_graph=True)

    # Collect gradients
    result = {}

    if "x" in vjp_inputs:
        result["x"] = (
            x_tensor.grad.numpy() if x_tensor.grad is not None else np.zeros_like(x)
        )

    if "t" in vjp_inputs:
        result["t"] = (
            t_tensor.grad.numpy() if t_tensor.grad is not None else np.zeros_like(t)
        )

    if "params_flat" in vjp_inputs:
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().cpu().numpy().flatten())
            else:
                grads.append(np.zeros(p.numel()))
        # Buffers don't have gradients
        for b in model.buffers():
            grads.append(np.zeros(b.numel()))
        result["params_flat"] = np.concatenate(grads).astype(np.float32)

    return result


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """
    Compute JVP (forward-mode AD) for Tesseract autodiff.

    Uses torch.autograd.functional.jvp
    """
    x = np.array(inputs.x)
    t = np.array(inputs.t)
    params_flat = np.array(inputs.params_flat)

    model = unflatten_params(params_flat)
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    def forward_fn(x_in, t_in):
        return model(x_in, t_in)

    # Prepare tangent vectors
    tangent_x = torch.tensor(
        np.array(tangent_vector.get("x", np.zeros_like(x))), dtype=torch.float32
    )
    tangent_t = torch.tensor(
        np.array(tangent_vector.get("t", np.zeros_like(t))), dtype=torch.float32
    )

    # Compute JVP
    _, jvp_out = torch.autograd.functional.jvp(
        forward_fn, (x_tensor, t_tensor), (tangent_x, tangent_t)
    )

    return {"u_pred": jvp_out.numpy()}


def abstract_eval(abstract_inputs):
    """Calculate output shape from input shape."""
    x_shape = abstract_inputs.x
    if hasattr(x_shape, "shape"):
        shape = x_shape.shape
    elif isinstance(x_shape, dict):
        shape = x_shape.get("shape", (None,))
    else:
        shape = (None,)

    # All outputs have the same shape as the input
    output_spec = {"shape": shape, "dtype": "float32"}
    return {
        "u_pred": output_spec,
        "u_x": output_spec,
        "u_t": output_spec,
        "u_xx": output_spec,
    }
