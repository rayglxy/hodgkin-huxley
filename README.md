# Hodgkin-Huxley Neuron Simulator
### Bifurcation Analysis & Phase Portrait of a Biophysical Neuron Model

![Phase Portrait](figures/phase_portrait.png)

## Overview

This project implements the Hodgkin-Huxley (1952) conductance-based neuron model 
from scratch in Python, then extends the analysis beyond standard textbook treatment 
into dynamical systems theory. The goal is not simply to simulate an action potential, 
but to characterise the mathematical structure of neural excitability — specifically, 
the bifurcation that governs the transition from quiescence to rhythmic firing.

The Hodgkin-Huxley model earned its authors the 1963 Nobel Prize in Physiology or 
Medicine. It remains the foundational quantitative description of how neurons generate 
electrical signals, and underlies virtually every biophysically realistic neural 
simulation in modern computational neuroscience and brain-computer interface research.

## Scientific context

This project identifies and characterises the Hopf bifurcation in the HH model — 
the critical injected current at which the resting fixed point loses stability and 
a stable limit cycle emerges. This is the mathematical event that corresponds 
biologically to a neuron beginning to fire rhythmically under sustained stimulation.

## Results

| Analysis | Result |
|----------|--------|
| Bifurcation point | I = 2.5 µA/cm² |
| Firing rate range | 0 – 120 Hz across physiological current range |
| Limit cycle | Stable, reproduced in V-n phase space |
| Fixed point | Identified at intersection of V and n nullclines |
| Validation | Action potential morphology consistent with HH 1952 figures |

![f-I Curve](figures/fi_curve.png)

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_model_implementation` | Implementation of the four ODEs, numerical integration via RK45, action potential validation against original 1952 figures |
| `02_parameter_sweep` | Systematic sweep of injected current (0–50 µA/cm²), spike detection, f-I curve, bifurcation point identification |
| `03_phase_portrait` | Phase portrait in V-n state space, V-nullcline and n-nullcline computation, limit cycle visualisation, fixed point analysis |

## Repository structure
HH_Neuron_2026/
├── notebooks/                  # Main analysis — run in order
│   ├── 01_model_implementation.ipynb
│   ├── 02_parameter_sweep.ipynb
│   └── 03_phase_portrait.ipynb
├── src/
│   └── hodgkin_huxley.py       # Clean importable Python module
├── figures/                    # Exported publication-quality figures
│   ├── action_potential.png
│   ├── fi_curve.png
│   └── phase_portrait.png
├── matlab/
│   └── hh_model.m              # MATLAB reimplementation
├── data/                       # External datasets (if applicable)
├── requirements.txt
└── README.md

## How to reproduce

```bash
git clone https://github.com/YOUR_USERNAME/hodgkin-huxley.git
cd hodgkin-huxley
pip install -r requirements.txt
jupyter notebook
```

Each notebook is self-contained 
and includes narrative explanation alongside the code.

The `src/hodgkin_huxley.py` module can also be imported directly:

```python
from src.hodgkin_huxley import simulate, firing_rate

t, V, _ = simulate(I_ext=10.0, t_end=100)
rate = firing_rate(I_ext=10.0)
```

## Methods

The model consists of four coupled ordinary differential equations describing 
the time evolution of membrane voltage V and three gating variables m, h, n. 
Numerical integration uses the RK45 adaptive solver (`scipy.integrate.solve_ivp`) 
with maximum step size 0.05 ms. Nullclines are computed under the quasi-steady-state 
approximation for m and h. All parameters taken directly from Hodgkin & Huxley (1952).

## Reference

Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane current 
and its application to conduction and excitation in nerve. 
*Journal of Physiology*, 117, 500–544.

## Author
Renata Tukanova
Computational neuroscience portfolio project.  
Python 3.10 · NumPy 2.2 · SciPy 1.15 · Matplotlib · Jupyter