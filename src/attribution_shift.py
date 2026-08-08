"""
attribution_shift.py

Second attribution method: |gradient| of output w.r.t. each gate's
parameter, via PennyLane's parameter-shift rule. Complements
attribution.py's occlusion method — occlusion is a large discrete
perturbation, parameter-shift is a local sensitivity measure.
Agreement between the two is cross-validation of the result.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from circuit import N_QUBITS, N_LAYERS, dev, ansatz
from encoding import angle_encoding

N_ROTATIONS_PER_QUBIT = 2


@qml.qnode(dev, diff_method="parameter-shift")
def _circuit_for_grad(x, params):
    angle_encoding(x, wires=range(N_QUBITS))
    ansatz(params, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def gate_attribution_shift(x, params):
    """Parameter-shift attribution scores for every gate on input x, sorted descending."""
    x_pnp = pnp.array(x, requires_grad=True)
    params_pnp = pnp.array(params, requires_grad=True)

    baseline = float(_circuit_for_grad(x_pnp, params_pnp))

    grad_x_fn = qml.grad(_circuit_for_grad, argnums=0)
    grad_params_fn = qml.grad(_circuit_for_grad, argnums=1)

    grad_x = np.array(grad_x_fn(x_pnp, params_pnp))
    grad_params = np.array(grad_params_fn(x_pnp, params_pnp)).reshape(
        N_LAYERS, N_QUBITS, N_ROTATIONS_PER_QUBIT
    )

    scores = []
    for q in range(N_QUBITS):
        scores.append((f"encoding[qubit={q}] (feature_{q})", abs(grad_x[q])))

    rot_names = {0: "RY", 1: "RZ"}
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            for rot in range(N_ROTATIONS_PER_QUBIT):
                scores.append(
                    (
                        f"ansatz[layer={layer}, qubit={q}, {rot_names[rot]}]",
                        abs(grad_params[layer, q, rot]),
                    )
                )

    scores.sort(key=lambda s: s[1], reverse=True)
    return baseline, scores


def compare_methods(x, params, top_n=14):
    """Run both attribution methods on the same sample and compare rankings."""
    from attribution import gate_attribution
    from scipy.stats import spearmanr

    baseline_occ, scores_occ = gate_attribution(x, params)
    baseline_shift, scores_shift = gate_attribution_shift(x, params)

    dict_occ = dict(scores_occ)
    dict_shift = dict(scores_shift)
    gate_names = list(dict_occ.keys())  # same gate set for both methods

    occ_vals = [dict_occ[g] for g in gate_names]
    shift_vals = [dict_shift[g] for g in gate_names]

    corr, pval = spearmanr(occ_vals, shift_vals)

    print(f"Baseline output — occlusion method:      {baseline_occ:.4f}")
    print(f"Baseline output — parameter-shift method: {baseline_shift:.4f}")
    print(
        f"(should match — both are just the unperturbed circuit output)\n"
    )

    print(f"{'Gate':45s} {'Occlusion':>12s} {'Param-Shift':>12s}")
    print("-" * 71)
    for g in sorted(gate_names, key=lambda g: dict_occ[g], reverse=True)[:top_n]:
        print(f"{g:45s} {dict_occ[g]:12.4f} {dict_shift[g]:12.4f}")

    print(f"\nSpearman rank correlation between methods: {corr:.4f} (p={pval:.4f})")

    # gates that occlusion found to have zero importance
    zero_gates = [g for g, v in dict_occ.items() if v < 1e-9]
    print(f"\nGates with ZERO occlusion importance ({len(zero_gates)} found):")
    for g in zero_gates:
        print(f"  {g}: occlusion={dict_occ[g]:.6f}, param-shift={dict_shift[g]:.6f}")

    return corr, dict_occ, dict_shift


if __name__ == "__main__":
    from data import make_moons_dataset

    params = np.load("trained_params.npy")
    X_train, X_test, y_train, y_test = make_moons_dataset()

    x = X_test[0]
    compare_methods(x, params)
