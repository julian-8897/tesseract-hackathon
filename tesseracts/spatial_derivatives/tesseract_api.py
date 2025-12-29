from typing import Any, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(BaseModel):
    u: Differentiable[Array[(None,), Float32]] = Field(
        description="1D Field Values (of length N)"
    )

    dx: Float32 = Field(description="Uniform grid spacing (>0)")

    bc: Literal["periodic", "dirichlet", "neumann"] = Field(
        default="dirichlet", description="Boundary Conditions"
    )
    # Finite-difference accuracy order
    order: Literal[2, 4] = Field(
        default=2, description="Finite-difference order (2 or 4)"
    )
    # Whether to return the second derivative as well
    compute_second: bool = Field(
        default=True, description="Return second derivative (d2u/dx2) if True"
    )


class OutputSchema(BaseModel):
    # First derivative (same shape as u)
    du_dx: Differentiable[Array[(None,), Float32]] = Field(
        description="First spatial derivative d(u)/dx"
    )
    # Optional second derivative (same shape as u)
    d2u_dx2: Optional[Differentiable[Array[(None,), Float32]]] = Field(
        default=None, description="Second spatial derivative d²(u)/dx² (if requested)"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """Compute spatial derivatives using finite differences with JAX autodiff."""
    u = jnp.array(inputs["u"])
    dx = inputs["dx"]
    bc = inputs["bc"]
    order = inputs["order"]
    compute_second = inputs["compute_second"]

    n = len(u)

    # First derivative computation
    if bc == "periodic":
        if order == 2:
            # 2nd order central: (u[i+1] - u[i-1]) / (2*dx)
            du_dx = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * dx)
        else:  # order == 4
            # 4th order central: (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / (12*dx)
            du_dx = (
                -jnp.roll(u, -2)
                + 8 * jnp.roll(u, -1)
                - 8 * jnp.roll(u, 1)
                + jnp.roll(u, -2)
            ) / (12 * dx)

    elif bc == "dirichlet":
        # Assume u=0 at boundaries
        if order == 2:
            du_dx = jnp.zeros_like(u)
            # Interior points: central difference
            du_dx = du_dx.at[1:-1].set((u[2:] - u[:-2]) / (2 * dx))
            # Boundary points: one-sided difference
            du_dx = du_dx.at[0].set((-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx))
            du_dx = du_dx.at[-1].set((u[-3] - 4 * u[-2] + 3 * u[-1]) / (2 * dx))
        else:  # order == 4
            du_dx = jnp.zeros_like(u)
            # Interior points (need at least 2 points on each side)
            du_dx = du_dx.at[2:-2].set(
                (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * dx)
            )
            # Near-boundary points: fall back to 2nd order
            du_dx = du_dx.at[1].set((u[2] - u[0]) / (2 * dx))
            du_dx = du_dx.at[-2].set((u[-1] - u[-3]) / (2 * dx))
            # Boundary points: one-sided
            du_dx = du_dx.at[0].set((-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx))
            du_dx = du_dx.at[-1].set((u[-3] - 4 * u[-2] + 3 * u[-1]) / (2 * dx))

    else:  # neumann: du/dx = 0 at boundaries
        if order == 2:
            du_dx = jnp.zeros_like(u)
            # Interior points
            du_dx = du_dx.at[1:-1].set((u[2:] - u[:-2]) / (2 * dx))
            # Boundary points: enforce zero gradient by extrapolation
            du_dx = du_dx.at[0].set(0.0)
            du_dx = du_dx.at[-1].set(0.0)
        else:  # order == 4
            du_dx = jnp.zeros_like(u)
            # Interior points
            du_dx = du_dx.at[2:-2].set(
                (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * dx)
            )
            # Near-boundary: 2nd order
            du_dx = du_dx.at[1].set((u[2] - u[0]) / (2 * dx))
            du_dx = du_dx.at[-2].set((u[-1] - u[-3]) / (2 * dx))
            # Boundary: zero gradient
            du_dx = du_dx.at[0].set(0.0)
            du_dx = du_dx.at[-1].set(0.0)

    # Second derivative computation (if requested)
    d2u_dx2 = None
    if compute_second:
        if bc == "periodic":
            if order == 2:
                # 2nd order central: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
                d2u_dx2 = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / (dx**2)
            else:  # order == 4
                # 4th order central: (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx^2)
                d2u_dx2 = (
                    -jnp.roll(u, -2)
                    + 16 * jnp.roll(u, -1)
                    - 30 * u
                    + 16 * jnp.roll(u, 1)
                    - jnp.roll(u, 2)
                ) / (12 * dx**2)

        elif bc == "dirichlet":
            if order == 2:
                d2u_dx2 = jnp.zeros_like(u)
                # Interior points
                d2u_dx2 = d2u_dx2.at[1:-1].set((u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2))
                # Boundary points: use one-sided stencils
                d2u_dx2 = d2u_dx2.at[0].set(
                    (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / (dx**2)
                )
                d2u_dx2 = d2u_dx2.at[-1].set(
                    (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4]) / (dx**2)
                )
            else:  # order == 4
                d2u_dx2 = jnp.zeros_like(u)
                # Interior points
                d2u_dx2 = d2u_dx2.at[2:-2].set(
                    (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4])
                    / (12 * dx**2)
                )
                # Near-boundary: 2nd order
                d2u_dx2 = d2u_dx2.at[1].set((u[2] - 2 * u[1] + u[0]) / (dx**2))
                d2u_dx2 = d2u_dx2.at[-2].set((u[-1] - 2 * u[-2] + u[-3]) / (dx**2))
                # Boundary: one-sided
                d2u_dx2 = d2u_dx2.at[0].set(
                    (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / (dx**2)
                )
                d2u_dx2 = d2u_dx2.at[-1].set(
                    (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4]) / (dx**2)
                )

        else:  # neumann
            if order == 2:
                d2u_dx2 = jnp.zeros_like(u)
                # Interior points
                d2u_dx2 = d2u_dx2.at[1:-1].set((u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2))
                # Boundary: use Neumann BC (extrapolate for ghost points)
                d2u_dx2 = d2u_dx2.at[0].set(2 * (u[1] - u[0]) / (dx**2))
                d2u_dx2 = d2u_dx2.at[-1].set(2 * (u[-2] - u[-1]) / (dx**2))
            else:  # order == 4
                d2u_dx2 = jnp.zeros_like(u)
                # Interior points
                d2u_dx2 = d2u_dx2.at[2:-2].set(
                    (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4])
                    / (12 * dx**2)
                )
                # Near-boundary: 2nd order
                d2u_dx2 = d2u_dx2.at[1].set((u[2] - 2 * u[1] + u[0]) / (dx**2))
                d2u_dx2 = d2u_dx2.at[-2].set((u[-1] - 2 * u[-2] + u[-3]) / (dx**2))
                # Boundary: Neumann
                d2u_dx2 = d2u_dx2.at[0].set(2 * (u[1] - u[0]) / (dx**2))
                d2u_dx2 = d2u_dx2.at[-1].set(2 * (u[-2] - u[-1]) / (dx**2))

    return {"du_dx": du_dx, "d2u_dx2": d2u_dx2}


def abstract_eval(abstract_inputs):
    """Calculate output shape from input shape without executing computation."""
    u_shapedtype = abstract_inputs.u

    return {
        "du_dx": ShapeDType(shape=u_shapedtype.shape, dtype=u_shapedtype.dtype),
        "d2u_dx2": ShapeDType(shape=u_shapedtype.shape, dtype=u_shapedtype.dtype),
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
