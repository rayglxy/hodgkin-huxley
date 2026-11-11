"""Checks that the model behaves like the published one.  Run:  pytest -q"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hodgkin_huxley import (alpha_m, alpha_n, derivatives, fixed_point, simulate,
                                spike_times, firing_rate, stability, hopf_current)


def test_rate_functions_are_finite_at_their_singular_voltages():
    # the textbook formulas are 0/0 at exactly these voltages
    assert np.isclose(alpha_m(25.0), 1.0) and np.isclose(alpha_n(10.0), 0.1)
    assert np.isclose(alpha_m(25.0), alpha_m(25.0 + 1e-6), atol=1e-6)


def test_rest_is_a_true_fixed_point():
    rest = fixed_point(0.0)
    assert abs(rest[0]) < 0.01                       # rest sits at 0 mV by definition
    assert np.allclose(derivatives(rest, 0.0), 0.0, atol=1e-9)


def test_no_current_no_spikes():
    t, y = simulate(0.0, t_end=200, dt=0.05)
    assert len(spike_times(t, y[0])) == 0


def test_action_potential_has_the_published_size():
    t, y = simulate(10.0, t_end=100, dt=0.01)
    assert 85 < y[0].max() < 110                     # about 100 mV above rest
    assert -15 < y[0].min() < -5                     # undershoot about 10 mV below rest


def test_weak_current_gives_one_spike_then_silence():
    t, y = simulate(4.0, t_end=1000, dt=0.05)
    assert len(spike_times(t, y[0])) == 1
    assert firing_rate(t, y[0]) == 0.0


def test_sustained_firing_starts_between_6_2_and_6_3():
    rates = []
    for I in (6.2, 6.3):
        t, y = simulate(I, t_end=1000, dt=0.05)
        rates.append(firing_rate(t, y[0]))
    assert rates[0] == 0.0 and 45 < rates[1] < 60    # jumps straight to about 52 Hz


def test_rest_loses_stability_near_9_78():
    assert abs(hopf_current() - 9.78) < 0.02         # known value for these parameters
    assert stability(8.0)[0] < 0 < stability(12.0)[0]


def test_bistable_at_8():
    # same current, two outcomes depending only on where the neuron starts
    t, y = simulate(8.0, t_end=600, dt=0.05, state0=fixed_point(8.0))
    assert len(spike_times(t, y[0])) == 0
    t, y = simulate(8.0, t_end=600, dt=0.05)         # switched on from rest at zero current
    assert firing_rate(t, y[0]) > 50
