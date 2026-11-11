# Hodgkin-Huxley neuron: one current, two states

Between **6.26 and 9.78 µA/cm²** the Hodgkin-Huxley neuron can either sit silent or fire at 52 to 68 Hz under exactly the same input. A single 1 ms pulse switches it on, and the identical pulse, timed between two spikes, switches it off.

![One pulse starts firing, the same pulse stops it](figures/switch.gif)

## What it is

The four-equation model of the squid giant axon (Hodgkin and Huxley, 1952) written from scratch in Python, then used to answer one question: at what input does a neuron start firing, and is that the same input at which it stops?

## Results

![Bistable range](figures/bistability.png)

| Quantity | Value (µA/cm²) | How it was found |
|---|---|---|
| Smallest current that gives a single spike | 2.24 | Switch current on from rest, bisect |
| Smallest current that gives sustained firing when switched on from rest | 6.26 | Same, counting only spikes after the first second |
| Firing, once started, stops below | 6.264 | Start in the firing state, lower the current, bisect |
| Resting state becomes unstable above | 9.779 | Eigenvalues of the Jacobian at the resting state cross zero |

- When firing starts it jumps straight to about 52 Hz; there is no slow firing. At 20 µA/cm² the rate is 86 Hz.
- Raising the current slowly from rest, firing began at 10.1 µA/cm². Lowering it again, firing continued down to 6.3. The neuron's response depends on its history.
- At 8 µA/cm²: 0 spikes before the first pulse, 8 spikes between the pulses, 0 after the second. Both pulses are +5 µA/cm² for 1 ms; the second arrives 8.65 ms after a spike.

![Switching](figures/switch.png)

![One action potential](figures/action_potential.png)

## Why it was done this way

- **Count only sustained firing.** Between 2.24 and 6.26 µA/cm² the neuron fires one to six spikes when the current is switched on and then stops. Counting those start-up spikes makes the onset look far lower than it is. An earlier version of this repository reported 2.5 for that reason.
- **Two ways of finding the upper threshold.** The simulation sweep gives 10.1; the eigenvalue calculation gives 9.779. They differ because just above the threshold the resting state loses its grip very slowly, so a sweep with finite waiting time overshoots. The eigenvalue result is the exact one.
- **Rate from spike spacing.** Firing rate is one divided by the time between spikes, not spikes counted in a window, which would round to multiples of a few Hz.
- **Rate formulas made safe.** Two of the textbook rate formulas are 0/0 at one voltage each; the code uses the limiting value there.

## Run it

```bash
pip install -r requirements.txt
```

```bash
pytest -q
```

```bash
python3 scripts/make_figures.py
```

```bash
python3 scripts/make_animation.py
```

The eight tests check the action potential size, the resting state, both thresholds and the bistability. `make_figures.py` takes about four minutes and writes every number above to `data/results.json`.

## Limitations

- This is the original squid-axon model at 6.3 °C with one compartment. Mammalian cortical neurons have other channels and many start firing gradually.
- Voltages are measured from rest (rest = 0 mV), as in the 1952 paper.
- No noise is included. With noise, a neuron in the bistable range would switch states at random.
- The bistability of this model is a known textbook result. What is here is an independent reproduction with tests, not a discovery.

## Reference

Hodgkin, A.L. and Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117, 500-544.

## Licence

MIT (see `LICENSE`).

Renata Tukanova, 2026. A MATLAB version of the basic model is in `matlab/hh_model.m`.
