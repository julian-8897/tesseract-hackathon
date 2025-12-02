# Tesseract Hackathon Template

A ready-to-use template for building projects with [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), featuring two interacting tesseracts that demonstrate vector scaling and similarity computation. Can be used as a starting point for participants of the [Tesseract Hackathon](link TODO).

> [!WARNING]
> Using this template is *not* required to participate in the Hackathon. You may use any tools at your disposal, including [Tesseract Core](https://github.com/pasteurlabs/tesseract-core), [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax), and [Tesseract-Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) — or composing Tesseracts via `docker run` calls in a glorified shell script. Your imagination is the limit!

#### See also
[[Tesseract Core Documentation](https://github.com/pasteurlabs/tesseract-core)] |
[[Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax)] |
[[Tesseract showcase](https://si-tesseract.discourse.group/c/showcase/11)] |
[[Get help @ Tesseract User Forums](https://si-tesseract.discourse.group/)]

## Overview

This template shows how to:
1. Define Tesseracts (`tesseracts/*`).
2. Build them locally (`buildall.sh`).
3. Serve Tesseracts locally, and compose them into a (differentiable) pipeline via [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) (`main.py`).

### Included Tesseracts

Example Tesseracts are minimal and meant as starting point for you to build upon.

1. scaler (`tesseracts/scaler`)
   - Scales input vectors by a given factor.
2. dotproduct (`tesseracts/dotproduct`)
   - Computes dot product between two vectors.
   - Calculates cosine similarity.
   - Uses the [Tesseract JAX recipe](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html#initialize-a-new-tesseract) to enable automatic differentiation.

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

4. **Build Tesseracts**
   ```bash
   $ ./buildall.sh
   ```

5. **Run the example pipeline**
   ```bash
   $ python main.py
   ```

## What the Demo Shows

The example pipeline [`main.py`](tree/main/main.py) demonstrates two paths:

**Path 1: Calling Tesseracts manually**
- Call Tesseracts via [Tesseract Core SDK](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/tesseract-api.html).

**Path 2: Composing Tesseracts with Tesseract-JAX**
- Wrap Tesseract calls in a differentiable JAX function using [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax).

## Now go and build your own!

Some pointers to get you started:

1. **Change Tesseract definitions**
  - Just update the code in `tesseracts/`. You can add / remove Tesseracts at will, and `buildall.sh` will... build them all.
  - Make sure to check out the [Tesseract docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html) to learn how to adapt existing configuration and define Tesseracts from scratch.
2. **Use gradients to perform optimization**
   - Exploit that Tesseract pipelines with AD endpoints are [end-to-end differentiable](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html).
   - Check [showcases](https://si-tesseract.discourse.group/c/showcase/11) for inspiration, e.g. the [Rosenbrock optimization showcase](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48) for a minimal demo.
3. **Deploy Tesseracts anywhere**
   - Since built Tesseracts are just Docker images, you can [deploy them virtually anywhere](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/deploy.html).
   - This includes [HPC clusters via SLURM](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104).
   - Have a look at [Tesseract Streamlit](https://github.com/pasteurlabs/tesseract-streamlit), which supports turning Tesseracts into web apps.
   - Show us how and where you run Tesseracts over the local network, on clusters, or in the cloud!
4. **Happy Hacking!** 🚀
   - Don't let these pointers constrain you. We're looking for creative solutions, so thinking out of the box is always appreciated.
   - Have fun, and [reach out](https://si-tesseract.discourse.group/) if you need help.

## License

Licensed under Apache License 2.0. All submissions must use the Apache License 2.0 to be eligible for the Tesseract Hackathon. See [LICENSE](LICENSE) file for details.