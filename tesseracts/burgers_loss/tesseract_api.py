# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Burgers Loss Tesseract

Computes the PDE residual for the 1D Burgers equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

Residual form: R = u·∂u/∂x - ν·∂²u/∂x²
Minimizing ||R||² enforces the physics constraint during neural operator training.
"""

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(BaseModel):
    """Input schema for Burgers equation residual computation."""

    u: Differentiable[Array[(None,), Float32]] = Field(
        description="Solution field at spatial points (length N)"
    )
    du_dx: Differentiable[Array[(None,), Float32]] = Field(
        description="First spatial derivative ∂u/∂x (from spatial_derivatives tesseract)"
    )
    d2u_dx2: Differentiable[Array[(None,), Float32]] = Field(
        description="Second spatial derivative ∂²u/∂x² (from spatial_derivatives tesseract)"
    )
    viscosity: Float32 = Field(
        description="Kinematic viscosity coefficient ν (typically 0.001 - 0.1)",
        default=0.01,
    )
    # Optional: reference solution for data loss
    u_true: Optional[Differentiable[Array[(None,), Float32]]] = Field(
        default=None, description="Ground truth solution (optional, for data loss)"
    )
    data_loss_weight: Float32 = Field(
        default=1.0, description="Weight for data loss term (if u_true provided)"
    )


class OutputSchema(BaseModel):
    """Output schema for Burgers loss tesseract."""

    residual: Differentiable[Array[(None,), Float32]] = Field(
        description="Pointwise PDE residual R = u·∂u/∂x - ν·∂²u/∂x²"
    )
    physics_loss: Differentiable[Float32] = Field(
        description="Physics-informed loss: mean squared residual ||R||²"
    )
    data_loss: Differentiable[Float32] = Field(
        description="Data loss: MSE between u and u_true (0 if u_true not provided)"
    )
    total_loss: Differentiable[Float32] = Field(
        description="Combined loss: physics_loss + data_loss_weight * data_loss"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """
    Compute Burgers equation residual and losses.

    Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    Residual: R = u·∂u/∂x - ν·∂²u/∂x²

    For steady-state or single time-step predictions, minimizing ||R||²
    enforces the PDE constraint.
    """
    u = jnp.array(inputs["u"])
    du_dx = jnp.array(inputs["du_dx"])
    d2u_dx2 = jnp.array(inputs["d2u_dx2"])
    viscosity = inputs["viscosity"]
    u_true = inputs.get("u_true", None)
    data_loss_weight = inputs.get("data_loss_weight", 1.0)

    # Compute PDE residual: u·∂u/∂x - ν·∂²u/∂x²
    # Note: For Burgers, the nonlinear term is u·∂u/∂x (convection)
    convection_term = u * du_dx
    diffusion_term = viscosity * d2u_dx2
    residual = convection_term - diffusion_term

    # Physics-informed loss: mean squared residual
    physics_loss = jnp.mean(residual**2)

    # Data loss: if ground truth provided, compute MSE
    if u_true is not None:
        u_true_array = jnp.array(u_true)
        data_loss = jnp.mean((u - u_true_array) ** 2)
    else:
        data_loss = jnp.array(0.0)

    # Total loss: weighted combination
    total_loss = physics_loss + data_loss_weight * data_loss

    return {
        "residual": residual,
        "physics_loss": physics_loss,
        "data_loss": data_loss,
        "total_loss": total_loss,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply function that calls the JIT-compiled implementation."""
    out = apply_jit(inputs.model_dump())
    return out


#
# JAX-handled AD endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Helper functions
#


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
