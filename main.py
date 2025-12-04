"""
Main Pipeline - Tesseract Hackathon Template

Demonstrates two Tesseracts working together:
1. scaler - Scales input vectors
2. dotproduct - Computes similarity between vectors

This is meant as a crude demonstration on how Tesseracts can be invoked from Python.
It is likely best to use this as inspiration, then start from scratch (as a Python script,
Jupyter notebook, full-blown Python package, or shell script).
"""

import jax
import jax.numpy as jnp
from jax import Array
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract


def path1_manual(
    scaler: Tesseract, dotproduct: Tesseract, vec_a: Array, vec_b: Array
) -> dict[str, Array]:
    """Path 1: Call Tesseracts via Tesseract SDK with intermediate results"""
    print(f"\nInput vectors: {vec_a}, {vec_b}")

    # Scale vector A
    output_a = scaler.apply({"vector": vec_a, "scale_factor": jnp.array(2.0)})
    scaled_a = output_a["scaled"]
    print(f"Scaled A: {scaled_a}")

    # Scale vector B
    output_b = scaler.apply({"vector": vec_b, "scale_factor": jnp.array(2.0)})
    scaled_b = output_b["scaled"]
    print(f"Scaled B: {scaled_b}")

    # Compute similarity
    result = dotproduct.apply({"vector_a": scaled_a, "vector_b": scaled_b})
    print(f"Cosine similarity: {result['cosine_similarity']:.4f}")
    return result


def path2_jax(
    scaler: Tesseract, dotproduct: Tesseract, vec_a: Array, vec_b: Array
) -> dict[str, Array]:
    """Path 2: Compose Tesseracts with Tesseract-JAX"""
    jax.debug.print("\nInput vectors: {vec_a}, {vec_b}", vec_a=vec_a, vec_b=vec_b)

    # Scale both and compute similarity
    out_a = apply_tesseract(scaler, {"vector": vec_a, "scale_factor": jnp.array(2.0)})
    out_b = apply_tesseract(scaler, {"vector": vec_b, "scale_factor": jnp.array(2.0)})
    result = apply_tesseract(
        dotproduct, {"vector_a": out_a["scaled"], "vector_b": out_b["scaled"]}
    )

    jax.debug.print("Cosine similarity: {res:.4f}", res=result["cosine_similarity"])
    return result


def main() -> None:
    print("\n" + "=" * 60)
    print("  TESSERACT DEMO: Two Interacting Tesseracts")
    print("=" * 60)

    # Behind the scenes, Tesseracts are spun up with Docker and serve an HTTP enpoint that can be called.
    # At the end of the context the containers are stopped.
    # It is possible to do that manually with:
    #     t = Tesseract.from_image(image_name)
    #     t.serve()
    #     t.teardown()
    # However we recommend using Tesseracts as a context manager for a safer approach.
    scaler = Tesseract.from_image("scaler")
    dotproduct = Tesseract.from_image("dotproduct")

    with scaler, dotproduct:
        # Hardcoded test vectors
        vec_a = jnp.array([3.0, 4.0, 0.0])
        vec_b = jnp.array([1.0, 0.0, 0.0])

        # Run both paths
        print("\n" + "=" * 60)
        print("PATH 1: Calling Tesseracts with Python SDK")
        print("=" * 60)
        _ = path1_manual(scaler, dotproduct, vec_a, vec_b)

        print("\n" + "=" * 60)
        print("PATH 2: Calling Tesseracts with Tesseract-JAX")
        print("=" * 60)
        _ = path2_jax(scaler, dotproduct, vec_a, vec_b)

        # Since path 2 uses Tesseract-JAX, it can be auto-differentiated using JAX machinery
        def path2_simplified(vec_a: Array, vec_b: Array) -> Array:
            # Modify signature so we map arrays to a scalar
            return path2_jax(scaler, dotproduct, vec_a, vec_b)["cosine_similarity"]

        path2_grad = jax.grad(path2_simplified, argnums=(0, 1))
        gradients = path2_grad(vec_a, vec_b)
        print(f"Gradient of cosine similarity wrt vec_a: {gradients[0]}")
        print(f"Gradient of cosine similarity wrt vec_b: {gradients[1]}")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
