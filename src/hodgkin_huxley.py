"""
hodgkin_huxley.py
-----------------
Clean implementation of the Hodgkin-Huxley (1952) neuron model.

Usage:
    from hodgkin_huxley import simulate, count_spikes

Reference:
    Hodgkin & Huxley (1952). A quantitative description of membrane current
    and its application to conduction and excitation in nerve.
    Journal of Physiology, 117, 500-544.
"""

import numpy as np
from scipy.integrate import solve_ivp


# --- Channel kinetics ---

def alpha_m(V): return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)
def beta_m(V):  return 4.0 * np.exp(-V / 18)

def alpha_h(V): return 0.07 * np.exp(-V / 20)
def beta_h(V):  return 1.0 / (np.exp((30 - V) / 10) + 1)

def alpha_n(V): return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)
def beta_n(V):  return 0.125 * np.exp(-V / 80)


# --- Model parameters (Hodgkin & Huxley 1952) ---

PARAMS = {
    'C_m':  1.0,    # membrane capacitance (µF/cm²)
    'g_Na': 120.0,  # max Na conductance (mS/cm²)
    'g_K':  36.0,   # max K conductance (mS/cm²)
    'g_L':  0.3,    # leak conductance (mS/cm²)
    'E_Na': 115.0,  # Na reversal potential (mV)
    'E_K':  -12.0,  # K reversal potential (mV)
    'E_L':  10.6,   # leak reversal potential (mV)
}


# --- ODE system ---

def hodgkin_huxley(t, state, I_ext, params=PARAMS):
    """
    The four coupled ODEs of the Hodgkin-Huxley model.

    Parameters
    ----------
    t : float
        Current time (ms)
    state : list [V, m, h, n]
        Current values of the four state variables
    I_ext : float
        Externally injected current (µA/cm²)
    params : dict
        Model parameters (default: HH 1952 values)

    Returns
    -------
    list : [dV/dt, dm/dt, dh/dt, dn/dt]
    """
    V, m, h, n = state
    C_m  = params['C_m']
    g_Na = params['g_Na']
    g_K  = params['g_K']
    g_L  = params['g_L']
    E_Na = params['E_Na']
    E_K  = params['E_K']
    E_L  = params['E_L']

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K  = g_K  * n**4      * (V - E_K)
    I_L  = g_L              * (V - E_L)

    dV = (I_ext - I_Na - I_K - I_L) / C_m
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dV, dm, dh, dn]


# --- Initial conditions ---

def resting_state():
    """Return the resting state initial conditions [V0, m0, h0, n0]."""
    V0 = 0.0
    m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
    h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
    n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
    return [V0, m0, h0, n0]


# --- Simulation ---

def simulate(I_ext, t_end=100, dt=0.01):
    """
    Run the HH model simulation.

    Parameters
    ----------
    I_ext : float
        Injected current (µA/cm²)
    t_end : float
        Simulation duration (ms)
    dt : float
        Time step for output (ms)

    Returns
    -------
    t : np.ndarray — time points (ms)
    V : np.ndarray — voltage trace (mV)
    full : np.ndarray — full solution [V, m, h, n] x time
    """
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(
        hodgkin_huxley,
        [0, t_end],
        resting_state(),
        t_eval=t_eval,
        args=(I_ext,),
        method='RK45',
        max_step=0.05
    )
    return sol.t, sol.y[0], sol.y


# --- Spike detection ---

def count_spikes(V_trace, threshold=50.0):
    """Count upward threshold crossings in a voltage trace."""
    crossings = 0
    for i in range(1, len(V_trace)):
        if V_trace[i-1] < threshold and V_trace[i] >= threshold:
            crossings += 1
    return crossings


def firing_rate(I_ext, t_end=200):
    """Return firing rate in Hz for a given injected current."""
    _, V, _ = simulate(I_ext, t_end=t_end)
    n_spikes = count_spikes(V)
    return (n_spikes / t_end) * 1000