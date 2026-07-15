# Deep Learning for PDEs in Engineering Physics — Final Exam Project

Solutions to the four exam problems of the TUM course *Deep Learning for PDEs in
Engineering Physics* (Summer 2026). Every deep learning component — network
architectures, losses, PDE residuals, spectral convolutions, Fourier transforms,
training/validation loops, checkpointing, inference, and visualization — is
implemented **from scratch in PyTorch**. No pre-built PDE/deep-learning libraries
(PINN frameworks, `neuraloperator`, `deepxde`, etc.) are used; the FNO does not
even use `torch.fft` (see Problem C).

**Author:** Om Anant Kulkarni

## Results Overview

| Problem | Task | Method | Final relative L² error | Training time* |
|---|---|---|---|---|
| A | Inverse problem: recover Young's modulus k(x) from noisy displacement data (1D elastostatics) | Inverse PINN, two-stage curriculum | k: 1.18e-1, u: 5.07e-3 | ~2 min |
| B | Forward Darcy flow with discontinuous permeability (2D, two-phase medium) | Deep Ritz method | 3.21e-2 | ~43 s |
| C | Operator learning a(x,y) → u(x,y) for heat conduction (1000 labeled pairs) | Fourier Neural Operator (hand-coded DFT) | 7.25e-3 | ~20 min |
| D | Semi-supervised operator learning for Burgers' equation (200 labeled + 1800 unlabeled ICs) | Physics-Informed DeepONet, two-phase training | 4.08e-2 | ~22 min |

\* on an NVIDIA GeForce RTX 4050 Laptop GPU, Float32, seed 1234.

## Repository Structure

```
.
├── Problem_A.ipynb                  # Inverse PINN (1D elastostatics)
├── Problem_B.ipynb                  # Deep Ritz (2D heterogeneous Darcy flow)
├── Problem_C.ipynb                  # FNO from scratch, no torch.fft (heat conduction surrogate)
├── Problem_D.ipynb                  # Physics-Informed DeepONet (Burgers traffic flow)
├── Om_Anant_Kulkarni_PDE_report.pdf # Project report
├── Report and Figures/              # LaTeX source of the report + all figures
│   ├── CS_report_template.tex       # Main file (official TUM template)
│   ├── CS_report.sty
│   ├── chapters/                    # Problem chapters, conclusions, appendix
│   └── figures/                     # All result and diagnostic figures (PNG)
└── README.md
```

## Dataset

The datasets are **not** included in this repository. Download them from Kaggle:
<https://www.kaggle.com/datasets/yhzang32/dno4pdes>

and place the four files in an `archive/` folder next to the notebooks:

```
archive/
├── ProblemA_dataset.h5
├── ProblemB_dataset.h5
├── ProblemC_dataset.h5
└── ProblemD_dataset.h5
```

(Each notebook reads `archive/Problem*_dataset.h5`; adjust `DATA_PATH` in the
first code cell if you store them elsewhere.)

## How to Run

Requirements: Python ≥ 3.10 with

```
pip install torch numpy h5py matplotlib tqdm
```

Each notebook is fully self-contained: open it and **Run All**. A CUDA GPU is
used automatically if available (`device = 'cuda:0' if ... else 'cpu'`);
everything also runs on CPU, just slower. Checkpoints and training histories
are written to `checkpoints/problemX/`, and every figure is also saved to
`figures/` as a 200-dpi PNG. All runs are seeded (`numpy` + `torch`, seed 1234).

Each notebook follows the same layout: problem statement → method description
and justification → data loading → network classes → loss classes → training
loop (with per-epoch L² relative test error and best-loss checkpointing) →
error-vs-epoch curve → inference from the best checkpoint with all required
figures → setup summary table.

## Method Summaries

### Problem A — Inverse PINN with a two-stage curriculum
Two MLPs approximate the displacement u(x) and the modulus k(x). Hard
constraints encode the physics exactly: `u = x(L−x)·N(x)` (boundary conditions)
and `k = softplus(N(x))` (positivity — which is also what makes the inverse
problem identifiable through the flux relation `k·u' = −fx + C`). Joint training
stalls in a local minimum where k stays constant, so training is staged:
(1) fit u to the 500 noisy sensors until the data loss reaches the noise floor,
(2) freeze u and recover k from the autodiff PDE residual plus a Tikhonov
smoothness penalty. The remaining k-error concentrates at x = 0.5, where the
flux degenerates and the noise bounds the attainable accuracy.

### Problem B — Deep Ritz method
The piecewise-constant permeability (values 2 / 10 on a 128×128 pixel image)
rules out strong-form PINNs (∇μ is undefined at the interfaces). The Darcy
problem is symmetric elliptic, so the solution minimizes the energy
E(u) = ∫ μ/2 |∇u|² dx, which only *evaluates* μ and only needs first
derivatives of u. The Dirichlet data (u = 1−x₁ on all four edges) is built into
the ansatz `u = (1−x₁) + x₁(1−x₁)x₂(1−x₂)·N(x)`, so there are no boundary
penalties and no loss weights at all. The Monte-Carlo energy uses stratified
sampling (one jittered point per pixel per epoch). A weak-form ParticleWNN was
also prototyped and rejected: its ball quadrature interacts badly with the
pixelated interfaces (training loss decreases while the true error grows).

### Problem C — Fourier Neural Operator, DFT written by hand
Standard FNO layout (lifting → 4 spectral-convolution layers with 1×1-conv skip
paths and GELU → projection), trained with a per-sample relative-L² loss on the
1000 labeled pairs. Since pre-built FFTs are not allowed, the truncated 2D
Fourier transform is implemented from its definition: partial DFT matrices
F_kn = exp(−2πikn/S) for the 2·12 kept modes per dimension, applied as matrix
products (cost O(2mS) per 1D transform, exact — verified in-notebook against
the direct DFT summation and the inverse round-trip identity). Complex mode
weights are stored as separate real/imaginary parameter tensors.

### Problem D — Physics-Informed DeepONet
The operator is `u(a)(x,t) = (1−x²) Σₖ bₖ(a)·τₖ(x,t) + b₀` with a branch net
encoding the initial condition at 256 sensors and a trunk net taking continuous
(x,t) — so the Burgers residual `u_t + u·u_x − ν·u_xx` is available by exact
automatic differentiation and can be imposed on the **1800 unlabeled** initial
conditions (plus an IC loss), while the 200 labeled ones give a supervised
loss. The (1−x²) factor enforces the boundary conditions exactly. Training is
two-phase (cheap data+IC warm-up, then physics fine-tuning): the supervised
phase plateaus at 10.1% test error, and the physics phase lowers it to 4.1% —
a factor 2.5 gained purely from unlabeled data.

## Report

`Om_Anant_Kulkarni_PDE_report.pdf` contains the full write-up (method selection
and justification, loss functions, architectures, experiment setups, results,
interpretation, and diagnostic appendix figures). The LaTeX source, based on
the official course template, is in `Report and Figures/`.
