<h1 align="center"><b>Quantum Circuit Attribution for Interpretable QML</b></h1>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PennyLane" src="https://img.shields.io/badge/PennyLane-0.45-orange">
  <img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

<p align="center">
  <img alt="quantum-machine-learning" src="https://img.shields.io/badge/-quantum--machine--learning-grey">
  <img alt="explainable-ai" src="https://img.shields.io/badge/-explainable--ai-grey">
  <img alt="variational-quantum-circuits" src="https://img.shields.io/badge/-variational--quantum--circuits-grey">
  <img alt="interpretability" src="https://img.shields.io/badge/-interpretability-grey">
</p>

Gate- and qubit-level attribution for variational quantum circuits
(VQCs): identify which quantum operations drive a given prediction,
using two independent methods (occlusion and parameter-shift
gradients), visualized as circuit-diagram heatmaps and validated
against ground-truth and known analytic results.

## Overview

Most QML interpretability work explains model *outputs* using
classical post-hoc methods. This project instead attributes
predictions back to *internal circuit structure* — which gates,
qubits, and layers mattered — using:

1. **Occlusion attribution:** zero out one gate's parameter at a
   time, re-run the circuit, measure the output shift.
2. **Parameter-shift attribution:** the analytic gradient of the
   output w.r.t. each gate's parameter, via PennyLane's
   parameter-shift rule.

Agreement between the two methods (a discrete perturbation vs. a
local sensitivity measure) serves as cross-validation of the result.

## Related Work

Gate-level explainability for quantum circuits is an active research
area, not an untouched gap — this project was built to explicitly
verify that before finalizing its framing:

- *Explaining Quantum Circuits with Shapley Values* (2023,
  *Quantum Machine Intelligence*) — Shapley-value gate attribution
  for VQCs, tested on real quantum hardware.
- *Q-SHAP* (2026) — gate importance via state fidelity, Shapley
  attribution, and entanglement flow tracking across benchmark
  circuits.
- A 2026 IEEE paper on qSHAP — SHAP-style attribution for PQC
  classifiers, comparing Integrated Gradients vs. baseline-SHAP.
- The specific "RZ gates commuting with a diagonal measurement have
  zero gradient" result (see Results below) is also independently
  established in prior work on redundant/irrelevant gates in PQCs.

This project's contribution is an independent implementation and
comparison of occlusion vs. parameter-shift attribution, including
cross-validation against a known analytic result and testing at two
different circuit scales — rather than a claim of an unstudied gap.

## Results

**2-qubit classifier (moons dataset):** 87.5% test accuracy.
Occlusion and parameter-shift attribution agree strongly
(Spearman r=0.81, p=0.0004). Both methods independently identify
the same 4 gates as having exactly zero effect on the output.

**Ground-truth validation:** on a synthetic dataset where only one
feature determines the label, attribution correctly ranks the
relevant feature ~134x above the irrelevant one.

**8-qubit classifier (8-feature dataset):** 78% test accuracy
(mini-batch Adam; full-batch training did not converge as cleanly at
this scale). Occlusion and parameter-shift attribution still agree
significantly (Spearman r=0.69, p<0.0001). The zero-effect-gate
pattern replicates: all zero-importance gates fall in the final
ansatz layer, and are confirmed as exactly zero by both methods
independently.

**Interpretation of the zero-effect gates:** the pattern is fully
explained by RZ gates commuting with a diagonal (Z-basis) measurement
observable — their analytic gradient is provably zero regardless of
parameter value, a known result in the PQC literature (see Related
Work). One exception (a final-layer RY gate on the CNOT ring's
wraparound qubit) is not explained by that argument alone and is
noted as an open question.

## Setup

```
pip install -r requirements.txt
```

## Usage

All scripts live in `src/`. Run any of them with `python3 <script>.py`
from inside that directory:

- `data.py` — dataset generators (moons, single-feature ground-truth, 8-feature)
- `encoding.py` — angle encoding, one feature per qubit
- `circuit.py` — variational ansatz + full classifier circuit
- `train.py` — full-batch and mini-batch training
  ```
  python3 train.py              # train the 2-qubit classifier
  ```
- `attribution.py` — occlusion-based gate attribution
  ```
  python3 attribution.py        # occlusion attribution on a test sample
  ```
- `attribution_shift.py` — parameter-shift attribution + comparison
  ```
  python3 attribution_shift.py  # parameter-shift attribution + comparison
  ```
- `validate.py` — ground-truth validation check
  ```
  python3 validate.py           # ground-truth validation check
  ```
- `visualize.py` — circuit-diagram attribution heatmap
  ```
  python3 visualize.py          # generate the attribution heatmap
  ```
- `run_8q_training.py` — 8-qubit training entry point
  ```
  python3 run_8q_training.py    # train the 8-qubit classifier
  ```

Generated outputs (trained weights, heatmap images) are saved
alongside the scripts and in `results/`.

## Tech Stack

Python, PennyLane (`default.qubit` simulator), NumPy, scikit-learn,
matplotlib, SciPy.

## Development Log

- Built the pipeline in stages: dataset + encoding, trained
  classifier, occlusion attribution, visualization + validation,
  parameter-shift attribution + comparison.
- Hit and fixed a real bug: `qml.grad(..., argnum=0)` failed because
  the installed PennyLane version uses `argnums` (plural). Diagnosed
  via `help(qml.grad)` rather than guessing, fixed in one edit.
- Investigated an unexpected result (several gates always showing
  exactly zero occlusion importance) rather than assuming it was a
  bug: ruled out near-zero trained parameters as the cause, confirmed
  it was consistent across samples, then confirmed it independently
  via parameter-shift gradients (also exactly zero). Later found this
  matches a known analytic result for RZ gates under diagonal
  measurement (see Related Work).
- Scaled the circuit from 2 to 8 qubits. Full-batch training became
  too slow at this scale (~14s per gradient step); switched to
  mini-batch training, which is standard practice and also resolved
  the runtime issue. Initial mini-batch runs converged poorly
  (58-68% accuracy); diagnosed via gradient-norm inspection (present
  but modest, not a severe barren plateau), then improved by
  switching to Adam with a larger batch size, reaching 78% accuracy.
- Ran a literature check before finalizing the novelty framing —
  found this is an active research area with close prior work (see
  Related Work above) and corrected the project's claims accordingly.

## Limitations

- Attribution is empirical for the general case; only the RZ
  zero-gradient result has a known analytic explanation.
- Tested at 2 and 8 qubits only; scaling behavior beyond that is
  untested.
- The single unexplained zero-importance gate (final-layer RY on the
  CNOT ring's wraparound qubit) needs further analysis.
- 8-qubit accuracy (78%) is noticeably lower than at 2 qubits (87.5%);
  further optimizer/architecture tuning was not pursued past this
  point.

## License

MIT — see [LICENSE](LICENSE).
