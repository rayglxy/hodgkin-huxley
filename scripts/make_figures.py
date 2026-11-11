#!/usr/bin/env python3
"""Compute every result in the README and draw the three static figures.

    python3 scripts/make_figures.py

Writes figures/*.png and *.svg, and data/results.json (all quoted numbers).
Takes about four minutes; the current sweep is cached in data/sweep.json.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.hodgkin_huxley import (simulate, spike_times, firing_rate, fixed_point, stability,
                                hopf_current, G_NA, G_K, E_NA, E_K)

plt.style.use(os.path.join(ROOT, 'figures', 'neuro.mplstyle'))
FIG, DATA = os.path.join(ROOT, 'figures'), os.path.join(ROOT, 'data')
INK, MUTED, GREY, BLUE, ORANGE, AQUA = '#0b0b0b', '#52514e', '#8a8985', '#2a78d6', '#eb6834', '#1baf7a'
BAND = '#eef3fb'


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'))
    fig.savefig(os.path.join(FIG, name + '.svg'))
    plt.close(fig)


# ── 1. where firing can start and stop ──────────────────────────────────────
def fold_current():
    """Lowest current at which firing, once started, keeps going."""
    _, y = simulate(7.0, t_end=300, dt=0.05)
    firing_state = y[:, -1]
    lo, hi = 6.0, 6.5
    for _ in range(12):
        mid = (lo + hi) / 2
        t, y = simulate(mid, t_end=1500, dt=0.05, state0=firing_state)
        if (spike_times(t, y[0]) > 1000).any():
            hi = mid
        else:
            lo = mid
    return hi


def step_onset():
    """Lowest current that gives sustained firing when switched on from rest."""
    lo, hi = 6.0, 6.5
    for _ in range(10):
        mid = (lo + hi) / 2
        t, y = simulate(mid, t_end=1500, dt=0.05)
        if (spike_times(t, y[0]) > 1000).any():
            hi = mid
        else:
            lo = mid
    return hi


def rheobase():
    """Lowest current that gives at least one spike when switched on from rest."""
    lo, hi = 1.5, 3.0
    for _ in range(10):
        mid = (lo + hi) / 2
        t, y = simulate(mid, t_end=200, dt=0.05)
        lo, hi = (lo, mid) if len(spike_times(t, y[0])) else (mid, hi)
    return hi


def sweep(step=0.1, top=20.0, dwell=400.0, measure=250.0):
    """Raise the current slowly in small steps, then lower it again.

    The neuron is never reset: each step starts from wherever the last one
    ended. That is what exposes the two different thresholds.
    """
    cache = os.path.join(DATA, 'sweep.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    up = np.round(np.arange(0.0, top + step / 2, step), 3)
    out = {'up': [], 'down': []}
    state = fixed_point(0.0)
    for direction, values in (('up', up), ('down', up[::-1])):
        for I in values:
            t, y = simulate(float(I), t_end=dwell, dt=0.05, state0=state)
            state = y[:, -1]
            late = t >= dwell - measure
            sp = spike_times(t[late], y[0][late])
            rate = 1000.0 / float(np.mean(np.diff(sp))) if len(sp) >= 2 else 0.0   # from the spacing between spikes
            out[direction].append(dict(I=float(I), rate=rate,
                                       vmax=float(y[0][late].max()), vmin=float(y[0][late].min())))
    json.dump(out, open(cache, 'w'))
    return out


# ── figure 1: one action potential, taken apart ──────────────────────────────
def fig_action_potential():
    t, y = simulate(10.0, t_end=60, dt=0.01)
    V, m, h, n = y
    s = spike_times(t, V)
    w = (t >= s[1] - 3) & (t <= s[1] + 11)          # second spike: past the switch-on transient
    tt = t[w] - (s[1] - 3)
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.28))
    ax[0].plot(tt, V[w], color=INK)
    ax[0].set_ylabel('Membrane voltage\n(mV from rest)')
    ax[0].set_title('One action potential, taken apart  (I = 10 µA/cm²)')
    for g, name, c in ((m[w], 'm  sodium opens', ORANGE), (h[w], 'h  sodium shuts', GREY), (n[w], 'n  potassium opens', BLUE)):
        ax[1].plot(tt, g, color=c)
        ax[1].text(tt[-1] + 0.2, g[-1], name, color=INK, va='center', fontsize=8.5)
    ax[1].set_ylabel('Gate open\nfraction'); ax[1].set_ylim(-0.03, 1.03)
    i_na = G_NA * m**3 * h * (V - E_NA); i_k = G_K * n**4 * (V - E_K)
    ax[2].plot(tt, -i_na[w], color=ORANGE); ax[2].plot(tt, -i_k[w], color=BLUE)
    k = np.argmax(-i_na[w]); ax[2].text(tt[k] + 0.5, -i_na[w][k], 'sodium current (in)', va='top', fontsize=8.5)
    k = np.argmin(-i_k[w]); ax[2].text(tt[k] + 0.6, -i_k[w][k], 'potassium current (out)', va='bottom', fontsize=8.5)
    ax[2].set_ylabel('Current that\ndepolarises (µA/cm²)'); ax[2].set_xlabel('Time (ms)')
    ax[2].set_xlim(0, 19)
    save(fig, 'action_potential')
    return dict(spike_peak_mV=float(V[w].max()), undershoot_mV=float(V[w].min()))


# ── figure 2: the bistable range ─────────────────────────────────────────────
def fig_bifurcation(sw, I_fold, I_hopf):
    I = np.array([r['I'] for r in sw['up']])
    up = np.array([r['rate'] for r in sw['up']])
    dn = np.array([r['rate'] for r in sw['down']])[::-1]
    vmax = np.array([r['vmax'] for r in sw['down']])[::-1]
    vmin = np.array([r['vmin'] for r in sw['down']])[::-1]
    v_rest = np.array([fixed_point(x)[0] for x in I])
    firing = dn > 0

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True, gridspec_kw=dict(hspace=0.16))
    for a in ax:
        a.axvspan(I_fold, I_hopf, color=BAND, zorder=0)
        a.grid(axis='x', visible=False)
    a = ax[0]
    stable = I <= I_hopf
    a.plot(I[stable], v_rest[stable], color=INK)
    a.plot(I[~stable], v_rest[~stable], color=INK, linestyle=(0, (3, 3)), linewidth=1.4)
    a.plot(I[firing], vmax[firing], color=BLUE); a.plot(I[firing], vmin[firing], color=BLUE)
    a.text(19.8, vmax[firing][-1] - 7, 'firing: voltage swings\nbetween these two lines', color=BLUE, ha='right', va='top', fontsize=8.5)
    a.text(1.0, 9, 'resting state (stable)', color=INK, fontsize=8.5)
    a.text(19.8, v_rest[-1] + 5, 'resting state (unstable)', color=INK, ha='right', fontsize=8.5)
    a.text((I_fold + I_hopf) / 2, 62, 'both\npossible', color=MUTED, ha='center', va='center', fontsize=9)
    a.set_ylabel('Voltage (mV from rest)')
    a.set_title('Between %.2f and %.2f µA/cm² the same neuron can rest or fire' % (I_fold, I_hopf))

    a = ax[1]
    a.plot(I, up, color=ORANGE); a.plot(I, dn, color=BLUE)
    i_on = I[np.argmax(up > 0)]
    a.annotate('', xy=(i_on, 60), xytext=(i_on, 6), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.4))
    a.annotate('', xy=(I_fold, 6), xytext=(I_fold, 46), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.4))
    a.text(i_on + 0.4, 22, 'raising the current:\nfiring starts here', color=INK, fontsize=8.5, va='center')
    a.text(I_fold - 0.4, 26, 'lowering it:\nfiring stops here', color=INK, fontsize=8.5, va='center', ha='right')
    a.set_ylabel('Firing rate (Hz)'); a.set_xlabel('Injected current (µA/cm²)')
    a.set_xlim(0, 20); a.set_ylim(-3, 95)
    save(fig, 'bistability')
    return dict(sweep_up_onset=float(i_on), sweep_down_offset=float(I[np.argmax(dn > 0)]),
                rate_at_onset_Hz=float(dn[np.argmax(dn > 0)]), rate_at_20_Hz=float(dn[-1]))


# ── figure 3: switching it on and off with the same pulse ────────────────────
def switch_protocol(I_hold=8.0, amp=5.0, width=1.0, t_on=60.0, phase=0.54):
    """Hold the current fixed inside the bistable range. A brief pulse starts
    firing; the identical pulse, timed between two spikes, stops it."""
    rest = fixed_point(I_hold)
    def with_pulses(times):
        return lambda t: I_hold + (amp if any(a <= t < a + width for a in times) else 0.0)
    t, y = simulate(with_pulses([t_on]), t_end=260, dt=0.01, state0=rest)
    s = spike_times(t, y[0])
    period = float(np.diff(s)[-1])
    t_off = float(s[s > 170][0] + phase * period)
    t, y = simulate(with_pulses([t_on, t_off]), t_end=330, dt=0.01, state0=rest)
    return t, y, with_pulses([t_on, t_off]), t_on, t_off, period


def fig_switch():
    t, y, cur, t_on, t_off, period = switch_protocol()
    s = spike_times(t, y[0])
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    ax[0].plot(t, y[0], color=INK, linewidth=1.3)
    ax[0].set_ylabel('Voltage (mV from rest)')
    ax[0].set_title('One brief pulse starts firing. The same pulse stops it.')
    ax[0].set_ylim(-16, 132)
    for x, label in ((t_on, 'pulse'), (t_off, 'same pulse,\nbetween two spikes')):
        ax[0].axvline(x, color=ORANGE, linewidth=1, linestyle=(0, (3, 3)))
        ax[0].text(x - 3, 128, label, color=INK, ha='right', va='top', fontsize=8.5)
    ax[1].plot(t, [cur(x) for x in t], color=ORANGE, linewidth=1.3)
    ax[1].set_ylabel('Current\n(µA/cm²)'); ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylim(6, 14.5); ax[1].set_xlim(0, t[-1])
    for a in ax: a.grid(axis='x', visible=False)
    save(fig, 'switch')
    return dict(hold_current=8.0, pulse_amplitude=5.0, pulse_width_ms=1.0,
                spikes_before_on=int((s < t_on).sum()),
                spikes_between=int(((s > t_on) & (s < t_off)).sum()),
                spikes_after_off=int((s > t_off + 5).sum()),
                off_pulse_delay_after_spike_ms=round(0.54 * period, 2), period_ms=round(period, 2))


def main():
    os.makedirs(DATA, exist_ok=True)
    I_hopf = hopf_current()
    res = dict(
        rest_at_zero_current=[round(float(x), 4) for x in fixed_point(0.0)],
        single_spike_threshold=round(rheobase(), 2),
        sustained_onset_from_rest_step=round(step_onset(), 2),
        firing_stops_below=round(fold_current(), 3),
        rest_unstable_above=round(I_hopf, 3),
        small_oscillation_Hz_at_instability=round(stability(I_hopf)[1] / (2 * np.pi) * 1000, 1),
    )
    res['action_potential'] = fig_action_potential()
    sw = sweep()
    res['sweep'] = fig_bifurcation(sw, res['firing_stops_below'], I_hopf)
    res['switch'] = fig_switch()
    json.dump(res, open(os.path.join(DATA, 'results.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
