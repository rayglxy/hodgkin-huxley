"""
Hodgkin-Huxley (1952) squid-axon model, with the tools needed to study how
firing starts and stops.

Conventions (original 1952 paper): voltage V is the DEPOLARISATION from rest
in mV (rest = 0), time in ms, current density in µA/cm², conductance in mS/cm²,
temperature 6.3 °C.

    from src.hodgkin_huxley import simulate, fixed_point, stability

Reference: Hodgkin & Huxley (1952), J. Physiol. 117, 500-544.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- parameters (Hodgkin & Huxley 1952) -------------------------------------
C_M = 1.0                                 # µF/cm²
G_NA, G_K, G_L = 120.0, 36.0, 0.3         # mS/cm²
E_NA, E_K, E_L = 115.0, -12.0, 10.6       # mV, relative to rest
SPIKE_THRESHOLD = 50.0                    # mV; spikes peak near 100, sub-threshold stays < 20


# --- channel kinetics --------------------------------------------------------
def _x_over_expm1(x):
    """x / (exp(x) - 1), finite at x = 0 (the raw formula is 0/0 there)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x / 2.0, safe / np.expm1(safe))


def alpha_m(V): return _x_over_expm1((25.0 - V) / 10.0)
def beta_m(V):  return 4.0 * np.exp(-V / 18.0)
def alpha_h(V): return 0.07 * np.exp(-V / 20.0)
def beta_h(V):  return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)
def alpha_n(V): return 0.1 * _x_over_expm1((10.0 - V) / 10.0)
def beta_n(V):  return 0.125 * np.exp(-V / 80.0)

def m_inf(V): return alpha_m(V) / (alpha_m(V) + beta_m(V))
def h_inf(V): return alpha_h(V) / (alpha_h(V) + beta_h(V))
def n_inf(V): return alpha_n(V) / (alpha_n(V) + beta_n(V))


# --- the four equations ------------------------------------------------------
def derivatives(state, I):
    """Right-hand side of the model for state = [V, m, h, n] and current I."""
    V, m, h, n = state
    I_ion = (G_NA * m**3 * h * (V - E_NA) + G_K * n**4 * (V - E_K) + G_L * (V - E_L))
    return np.array([
        (I - I_ion) / C_M,
        alpha_m(V) * (1 - m) - beta_m(V) * m,
        alpha_h(V) * (1 - h) - beta_h(V) * h,
        alpha_n(V) * (1 - n) - beta_n(V) * n,
    ])


def simulate(I, t_end=100.0, dt=0.01, state0=None):
    """Integrate the model.

    I       constant current (float) or a function I(t) in µA/cm²
    state0  starting [V, m, h, n]; default is the resting state at I = 0
    Returns t (ms) and the solution array y with rows V, m, h, n.
    """
    current = I if callable(I) else (lambda t: I)
    y0 = fixed_point(0.0) if state0 is None else np.asarray(state0, dtype=float)
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(lambda t, y: derivatives(y, current(t)), (0.0, t_end), y0,
                    t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10, max_step=0.1)
    return sol.t, sol.y


# --- spikes ------------------------------------------------------------------
def spike_times(t, V, threshold=SPIKE_THRESHOLD):
    """Times at which V crosses the threshold going up."""
    up = (V[:-1] < threshold) & (V[1:] >= threshold)
    return t[1:][up]


def firing_rate(t, V, discard=200.0):
    """Sustained firing rate in Hz, ignoring the first `discard` ms.

    The start-up transient is discarded because a neuron can fire one or a few
    spikes when the current is switched on and then fall silent. Counting those
    as "firing" is what makes the onset look much lower than it is.
    """
    s = spike_times(t, V)
    s = s[s >= discard]
    window = t[-1] - discard
    return 1000.0 * len(s) / window


# --- fixed point and its stability -------------------------------------------
def fixed_point(I):
    """The resting state for a constant current I: the point where nothing changes."""
    def net_current(V):
        return I - (G_NA * m_inf(V)**3 * h_inf(V) * (V - E_NA)
                    + G_K * n_inf(V)**4 * (V - E_K) + G_L * (V - E_L))
    V = brentq(net_current, -20.0, 60.0, xtol=1e-12)
    return np.array([V, float(m_inf(V)), float(h_inf(V)), float(n_inf(V))])


def jacobian(state, I, eps=1e-6):
    """Matrix of how each derivative changes with each variable (central differences)."""
    J = np.zeros((4, 4))
    for k in range(4):
        d = np.zeros(4); d[k] = eps
        J[:, k] = (derivatives(state + d, I) - derivatives(state - d, I)) / (2 * eps)
    return J


def stability(I):
    """Largest real part of the Jacobian's eigenvalues at the fixed point.

    Negative: a small push away from rest dies out (rest is stable).
    Positive: a small push grows (rest is unstable; the neuron must fire).
    """
    eig = np.linalg.eigvals(jacobian(fixed_point(I), I))
    lead = eig[np.argmax(eig.real)]
    return lead.real, abs(lead.imag)


def hopf_current(lo=5.0, hi=15.0):
    """Current at which the resting state loses stability (Hopf bifurcation)."""
    return brentq(lambda I: stability(I)[0], lo, hi, xtol=1e-6)
