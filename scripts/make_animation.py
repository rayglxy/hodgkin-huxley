#!/usr/bin/env python3
"""Animated version of the switching experiment.

    python3 scripts/make_animation.py

The injected current is held at 8 µA/cm² the whole time. Left: the neuron's
state drawn as a point in the voltage / potassium-gate plane. At rest the
point sits still; while firing it runs round a loop. Right: voltage and
current against time. One brief pulse knocks the point onto the loop; the
same pulse, timed between two spikes, knocks it back.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from make_figures import switch_protocol, INK, MUTED, GREY, BLUE, ORANGE
from src.hodgkin_huxley import fixed_point

plt.style.use(os.path.join(ROOT, 'figures', 'neuro.mplstyle'))


def main(out=os.path.join(ROOT, 'figures', 'switch.gif'), n_frames=150, fps=15):
    t, y, cur, t_on, t_off, _ = switch_protocol()
    keep = (t >= 30) & (t <= 260)
    t, V, n = t[keep] - 30, y[0][keep], y[3][keep]
    I = np.array([cur(x + 30) for x in t])
    t_on, t_off = t_on - 30, t_off - 30
    rest = fixed_point(8.0)
    loop = (t > t_on + 60) & (t < t_off - 2)            # the settled firing loop

    fig = plt.figure(figsize=(8.4, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[3, 1], wspace=0.28, hspace=0.18)
    ax_p, ax_v, ax_i = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    fig.suptitle('Same steady current. One pulse starts firing, the same pulse stops it.',
                 x=0.06, ha='left', fontsize=11, fontweight='bold')

    ax_p.plot(V[loop], n[loop], color=GREY, linewidth=1, alpha=0.6)
    ax_p.plot(rest[0], rest[3], 'o', color=GREY, markersize=5)
    ax_p.text(rest[0] + 5, rest[3] - 0.035, 'rest', color=MUTED, fontsize=8.5)
    ax_p.text(60, 0.73, 'firing loop', color=MUTED, fontsize=8.5)
    ax_p.set_xlabel('Voltage (mV from rest)'); ax_p.set_ylabel('Potassium gate n')
    ax_p.set_xlim(-18, 105); ax_p.set_ylim(0.3, 0.8); ax_p.grid(False)
    trail, = ax_p.plot([], [], color=BLUE, linewidth=2)
    dot, = ax_p.plot([], [], 'o', color=BLUE, markersize=8, markeredgecolor='#fcfcfb')
    state = ax_p.text(0.04, 0.95, '', transform=ax_p.transAxes, va='top', fontsize=10, fontweight='bold')

    ax_v.plot(t, V, color='#dddcd6', linewidth=1)
    line_v, = ax_v.plot([], [], color=INK, linewidth=1.3)
    ax_v.set_ylabel('Voltage (mV)'); ax_v.set_xlim(0, t[-1]); ax_v.set_ylim(-16, 105)
    ax_v.tick_params(labelbottom=False)
    ax_i.plot(t, I, color='#f3d5c7', linewidth=1)
    line_i, = ax_i.plot([], [], color=ORANGE, linewidth=1.3)
    ax_i.set_ylabel('Current'); ax_i.set_xlabel('Time (ms)'); ax_i.set_xlim(0, t[-1]); ax_i.set_ylim(6, 14.5)
    for a in (ax_v, ax_i): a.grid(axis='x', visible=False)

    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    tail = int(12 / (t[1] - t[0]))                       # 12 ms of trail

    def draw(k):
        i = idx[k]
        trail.set_data(V[max(0, i - tail):i + 1], n[max(0, i - tail):i + 1])
        dot.set_data([V[i]], [n[i]])
        line_v.set_data(t[:i + 1], V[:i + 1]); line_i.set_data(t[:i + 1], I[:i + 1])
        state.set_text('resting' if (t[i] < t_on or t[i] > t_off + 4) else 'firing')
        return trail, dot, line_v, line_i, state

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    print(out, f'{os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
