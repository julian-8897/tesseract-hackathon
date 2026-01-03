# Tesseract Cross-Framework Autodiff Demo

**Inverse Problem: Inferring Viscosity from Burgers Equation**

This project showcases Tesseract's key capability: **cross-framework automatic differentiation**. The same inverse problem code works with either JAX or PyTorch PINN implementations, and JAX gradients flow seamlessly through PyTorch via Tesseract's VJP endpoint!

## 🎯 What This Demo Does

Given observed solution data from the Burgers equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

We use a **Physics-Informed Neural Network (PINN)** to infer the unknown viscosity parameter $\nu$.

### Key Features

✅ **Swappable Backends**: Same code works with `pinn_jax` or `pinn_pytorch`  
✅ **Cross-Framework Gradients**: `jax.grad` computes gradients through PyTorch  
✅ **Internal Autodiff**: Derivatives computed via framework-native autodiff (no finite differences)  
✅ **Interactive Demo**: Streamlit app with live training visualization  
✅ **Side-by-Side Comparison**: Compare JAX vs PyTorch performance

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker ([Docker Desktop recommended](https://docs.docker.com/desktop/))
- `uv` package manager (or use `pip`/`conda`)


### Installation

1. Clone the repository
2. Install dependencies:

```bash
uv pip install -e .
# Or with pip: pip install -e .
```

3. Build the PINN tesseracts:

```bash
./buildall.sh
```

This will build Docker images for both `pinn_jax` and `pinn_pytorch`.

## 💡 Example Results

**Command-Line Output:**
```
================================================================
  CROSS-FRAMEWORK AUTODIFF DEMO
  Tesseract enables jax.grad through PyTorch!
================================================================

✓ JAX PINN tesseract ready
Epoch  100 | ν: 0.048923 | Error: 0.001077 | Loss: 0.000542

✓ PyTorch PINN tesseract ready  
Epoch  100 | ν: 0.049156 | Error: 0.000844 | Loss: 0.000487

→ JAX is 1.2x faster than PyTorch
```

**Key Insight**: Both backends converge to the true viscosity (ν=0.05) with <3% error!

## 🔗 Resources

- [Tesseract Core Documentation](https://github.com/pasteurlabs/tesseract-core)
- [Tesseract-JAX Documentation](https://github.com/pasteurlabs/tesseract-jax)
- [Tesseract Showcase](https://si-tesseract.discourse.group/c/showcase/11)
- [Get Help @ Tesseract Forums](https://si-tesseract.discourse.group/)

## 📝 License

See [LICENSE](LICENSE) file for details.

### Quickstart

1. Create a new repository off this template and clone it
   ```bash
   $ git clone <your-repo-url>
   $ cd <myrepo>
   ```

2. Set up virtual environment (if not done already). `uv` or `conda` can also be used.
   ```bash
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   $ pip install -r requirements.txt
   ```

4. Build Tesseracts
   ```bash
   $ ./buildall.sh
   ```

5. Run the example pipeline
   ```bash
   $ python main.py
   ```

## Now go and build your own!

Some pointers to get you started:

1. **Change Tesseract definitions**.
     - Just update the code in `tesseracts/*`. You can add / remove Tesseracts at will, and `buildall.sh` will... build them all.
     - Make sure to check out the [Tesseract docs](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html) to learn how to adapt existing configuration and define Tesseracts from scratch.
2. **Use gradients to perform optimization**.
   - Exploit that Tesseract pipelines with AD endpoints are [end-to-end differentiable](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html).
   - Check [showcases](https://si-tesseract.discourse.group/c/showcase/11) for inspiration, e.g. the [Rosenbrock optimization showcase](https://si-tesseract.discourse.group/t/jax-based-rosenbrock-function-minimization/48) for a minimal demo.
3. **Deploy Tesseracts anywhere**.
   - Since built Tesseracts are just Docker images, you can [deploy them virtually anywhere](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/deploy.html).
   - This includes [HPC clusters via SLURM](https://si-tesseract.discourse.group/t/deploying-and-interacting-with-tesseracts-on-hpc-clusters-using-tesseract-runtime-serve/104).
   - Have a look at [Tesseract Streamlit](https://github.com/pasteurlabs/tesseract-streamlit) that can turn Tesseracts into web apps.
   - Show us how and where you run Tesseracts over the local network, on clusters, or in the cloud!
4. **Happy Hacking!** 🚀
   - Don't let these pointers constrain you. We're looking for creative solutions, so thinking out of the box is always appreciated.
   - Have fun, and [reach out](https://si-tesseract.discourse.group/) if you need help.

## License

Licensed under Apache License 2.0.

All submissions must use the Apache License 2.0 to be eligible for the Tesseract Hackathon. See [LICENSE](LICENSE) file for details.
