# Tesseract Hackathon Template

A ready-to-use template for building projects with [tesseract-core](https://github.com/pasteurlabs/tesseract-core) and [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation. Can be used as a starting point for participants in the [Tesseract Hackathon](link TODO).

> [!NOTE]
> Using this template is *not* required to participate in the Hackathon. You may use any tools at your disposal, including [Tesseract Core](https://github.com/pasteurlabs/tesseract-core), [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), and [Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) --- or composing Tesseracts via `docker run` calls in a glorified shell script. Your imagination is the limit!## Resources

#### See also
[Tesseract Core Documentation](https://github.com/pasteurlabs/tesseract-core) |
[Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax) |
[JAX Documentation](https://jax.readthedocs.io/) |
[Get help @ Tesseract User Forums](https://si-tesseract.discourse.group/)

## Overview

This template shows how to:
1. Define Tesseracts (`tesseracts/`).
2. Build them locally (`buildall.sh`).
3. Serve Tesseracts locally, and compose them into a (differentiable) pipeline via [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) (`main.py`).

### Included Tesseracts

Example Tesseracts are minimal and meant as starting point for you to build upon.

1. scaler (`tesseracts/scaler`)
   - Scales input vectors by a given factor.
2. dotproduct (`tesseracts/dotproduct`)
   - Computes dot product between two vectors.
   - Calculates cosine similarity.
3. dotproduct_jax (`tesseracts/dotproduct_jax`)
   - Same as dotproduct, but uses the [Tesseract JAX recipe](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html#initialize-a-new-tesseract) to enable automatic differentiation.

## Quick Start

### Prerequisites

- Python 3.10 or higher, ideally with a virtual environment (e.g. via `venv`, `conda`, or `uv`).
- Working Docker setup for the current user ([Docker Desktop recommended](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html#installing-docker)).

### Installation

1. **Create a new repository off this template and clone it**
   ```bash
   $ git clone <your-repo-url>
   $ cd <myrepo>
   ```

2. **Set up virtual environment** (if not done already)
   ```bash
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   $ pip install -r requirements.txt
   ```

4. **Build tesseracts**
   ```bash
   $ ./buildall.sh
   ```

5. **Run the demo**
   ```bash
   $ python main.py
   ```

## What the Demo Shows

Running `python main.py` demonstrates two paths:

**Path 1: Calling tesseracts separately**
- Load each tesseract
- Call scaler twice (for two vectors)
- Call dotproduct once (on scaled vectors)

**Path 2: Composing tesseracts together**
- Wrap tesseract calls in a single jax function
- Create a reusable pipeline
- Get the same result with cleaner code

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**Happy Hacking!** 🚀
