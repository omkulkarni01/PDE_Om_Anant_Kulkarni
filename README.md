# Deep Learning for PDEs in Engineering Physics – Final Project

This repo contains my solutions for the four exam problems of the TUM course
*Deep Learning for PDEs in Engineering Physics* (Summer 2026).

The course rules do not allow pre-built implementations of PINNs, FNOs, etc.
from open source packages, so I wrote all of the key parts myself in plain
PyTorch: the networks, the loss functions, the PDE residuals via autograd, the
training and validation loops, checkpointing, and the plotting. For Problem C
this also meant writing the Fourier transform by hand instead of calling
`torch.fft` (details below).

Author: Om Anant Kulkarni

## What is in here

```
Problem_A.ipynb                    inverse PINN, 1D elastostatics
Problem_B.ipynb                    Deep Ritz, 2D Darcy flow with two-phase permeability
Problem_C.ipynb                    FNO written from scratch, heat conduction surrogate
Problem_D.ipynb                    physics-informed DeepONet, Burgers equation
Om_Anant_Kulkarni_PDE_report.pdf   the project report
Report and Figures/                LaTeX source of the report + all figures
```

The notebooks still contain the outputs of my training runs, so the logs,
error curves and figures can be inspected without re-running anything.

## Results

All errors are the relative L2 error on the test data, as defined in the
problem sheet. Timings are from my laptop (RTX 4050, 6 GB, float32, seed 1234).

| Problem | Method | Final test error | Time |
|---|---|---|---|
| A | inverse PINN (two training stages) | k: 1.18e-1, u: 5.07e-3 | ~2 min |
| B | Deep Ritz | 3.21e-2 | ~43 s |
| C | FNO (hand-written DFT) | 7.25e-3 | ~20 min |
| D | PI-DeepONet (two training phases) | 4.08e-2 | ~22 min |

## Running the notebooks

You need Python 3.10+ and:

```
pip install torch numpy h5py matplotlib tqdm
```

The datasets are not in the repo (they are large). Get them from
https://www.kaggle.com/datasets/yhzang32/dno4pdes and put the four h5 files
into a folder called `archive/` next to the notebooks. If you keep them
somewhere else, change `DATA_PATH` in the first code cell.

Then just open a notebook and run all cells. CUDA is used when available,
otherwise everything falls back to CPU (slower but works). Checkpoints and the
error histories go to `checkpoints/`, figures are additionally saved as PNGs
into `figures/`. Seeds are fixed, so reruns should give the same numbers up to
GPU nondeterminism.

## Short notes on the methods

**Problem A.** Two small MLPs, one for the displacement u and one for the
modulus k. The boundary conditions are built into the ansatz (u = x(L-x)*N(x))
and k goes through a softplus, which keeps it positive. The positivity is
actually needed for uniqueness here, not just for looks: integrating the PDE
once gives k u' = -fx + C, and only the positive branch pins down C. Training
both networks jointly got stuck with k almost constant, so I train in two
stages: first fit u to the noisy sensors (the data loss ends up exactly at the
noise variance), then freeze u and fit k from the PDE residual with a small
smoothness penalty. The leftover error in k sits around x = 0.5 where u' = 0
and the data simply does not contain much information about k.

**Problem B.** The permeability is piecewise constant, so a normal strong-form
PINN would have to differentiate it, which makes no sense at the interfaces.
The Deep Ritz method avoids this: minimize the energy integral mu/2 |grad u|^2,
where mu is only evaluated. The boundary condition u = 1 - x1 holds on all four
edges, so I put it directly into the ansatz and there are no loss weights left
to tune. For the integral I sample one jittered point per pixel per epoch
instead of plain uniform sampling, which removes most of the Monte Carlo noise.
I first tried ParticleWNN for this problem; it looked fine on paper but the
quadrature over the small balls breaks down at the pixelated interfaces (the
loss kept going down while the actual error went up), so I dropped it.

**Problem C.** Standard FNO structure: lifting, four spectral conv layers with
1x1 conv skip connections and GELU, projection. Trained on the 1000 pairs with
a per-sample relative L2 loss. Since FFT routines were not allowed, I build
the truncated DFT matrices from the definition (only the 12 lowest positive
and negative modes per dimension are ever needed) and apply the transform as
two matrix multiplications. The notebook checks these matrices against the
direct DFT sum and the inverse round trip before training. The complex mode
weights are stored as separate real and imaginary tensors.

**Problem D.** Only 200 of the 2000 initial conditions come with solutions, so
a purely data-driven model would throw away 90% of the data. A DeepONet with a
trunk net taking continuous (x,t) lets me put the Burgers residual (via
autograd) and the initial condition on the 1800 unlabeled samples as extra
loss terms. The factor (1-x^2) in front of the output enforces the boundary
conditions exactly. Training runs in two phases because a residual epoch is
roughly 40x more expensive than a data epoch: first a cheap warm-up on data +
IC loss, then the full physics loss with a smaller learning rate. The warm-up
alone plateaus at about 10% test error, the physics phase brings it down to
4.1%, which is the main point of the exercise.

## Report

The full write-up is in `Om_Anant_Kulkarni_PDE_report.pdf` (methods, losses,
setups, results and interpretation, plus an appendix with extra diagnostic
figures). The LaTeX source, based on the official course template, is in
`Report and Figures/`.
