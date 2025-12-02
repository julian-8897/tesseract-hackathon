"""
Scaler Tesseract

This tesseract takes an input vector and a scale factor, returning the scaled vector
"""

from typing import Any
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    """Input schema for a single vector with scale factor"""
    vector: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    scale_factor: Float32 = Field(description="A scalar to scale the vector by", default=1.0)

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        if len(self.vector.shape) != 1:
            raise ValueError(
                f"Vector must be 1-dimensional. "
                f"Got {self.vector.shape} instead."
            )
        return self


class OutputSchema(BaseModel):
    """Output schema for the scaler tesseract"""
    scaled: Differentiable[Array[(None,), Float32]] = Field("Scaled vector")


def apply(inputs: InputSchema) -> OutputSchema:
    result = inputs.vector * inputs.scale_factor

    return OutputSchema(scaled=result)


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    input_vector_shapedtype = abstract_inputs.vector

    return {
        "scaled": ShapeDType(
            shape=input_vector_shapedtype.shape,
            dtype=input_vector_shapedtype.dtype
        )
    }

def vector_jacobian_product(
        inputs: InputSchema, 
        vjp_inputs: set[str], 
        vjp_outputs: set[str], 
        cotangent_vector: dict[str, Any]
    ):
    """Compute vector-Jacobian product, for use in reverse-mode autodiff (backpropagation).
    
    The VJP is the product of the cotangent vector with the Jacobian matrix, which is a constant in this case.
    """
    assert vjp_inputs == {"vector"}
    assert vjp_outputs == {"scaled"}

    # Jacobian matrix is constant in this case
    jac = inputs.scale_factor

    # Assemble VJP
    out = {}
    for dx in vjp_inputs:
        out[dx] = sum(jac * cotangent_vector[dy] for dy in vjp_outputs)
    return out