# Tesseract Hackathon Template

A ready-to-use template for building projects with [tesseract-core](https://github.com/pasteurlabs/tesseract-core) and [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation. Can be used as a starting point for participants of the [Tesseract Hackathon](link TODO).

[!NOTE]
Using this template is *not* required to participate in the Hackathon. You may use any tools at your disposal, including [Tesseract Core](https://github.com/pasteurlabs/tesseract-core), [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), and [Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) --- or composing Tesseracts via `docker run` calls in a glorified shell script. Your imagination is the limit!

## Overview

This template shows how to:
- Build and serve tesseracts locally
- Call tesseracts separately
- Compose tesseracts into a pipeline

### Included Tesseracts

**1. scaler**
- Scales input vectors by a given factor

**2. dotproduct_jax**
- Created based on Tesseract's JAX recipe
- Computes dot product between two vectors
- Calculates cosine similarity

## Quick Start

### Prerequisites

- Python 3.10 or higher
- tesseract-core and tesseract-jax installed
- JAX installed

### Installation

1. **Clone or use this template**

2. **Set up environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Build tesseracts**
   ```bash
   ./buildall.sh
   ```

4. **Run the demo**
   ```bash
   python main.py
   ```

## What the Demo Shows

Running `python main.py` demonstrates two paths:

**Path 1: Calling tesseracts separately**
- Call tesseracts via Tesseract Core SDK

**Path 2: Composing tesseracts together**
- Wrap tesseract calls in a single jax function using Tesseract-JAX
- Create a clean reusable pipeline

## Resources

- [Tesseract-Core Documentation](https://github.com/pasteurlabs/tesseract-core)
- [Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax)
- [JAX Documentation](https://jax.readthedocs.io/)

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

**Happy Hacking!** 🚀
