"""
attribution.py

Occlusion-based gate attribution: zero out one gate's parameter at a
time, re-run the circuit, measure the output shift. Larger shift =
more important. Applied to both encoding gates (input features) and
ansatz gates (trained parameters).
"""

import numpy as np
import pennylane as qml

from circuit import N_QUBITS, N_LAYERS, dev
from encoding import angle_encoding

N_ROTATIONS_PER_QUBIT = 2  # RY, RZ


def _run_circuit(x, params, occlude=None):
    """
    One forward pass, optionally zeroing exactly one gate.
    occlude: None, or ("encoding", qubit_idx), or
             ("ansatz", layer_idx, qubit_idx, rotation_idx)  (0=RY, 1=RZ)
    """
    params = params.reshape(N_LAYERS, N_QUBITS, N_ROTATIONS_PER_QUBIT)

    @qml.qnode(dev)
    def circuit():
        # --- encoding ---
        for q in range(N_QUBITS):
            angle = 0.0 if occlude == ("encoding", q) else x[q]
            qml.RY(angle, wires=q)

        # --- ansatz ---
        for layer in range(N_LAYERS):
            for q in range(N_QUBITS):
                ry_angle = params[layer, q, 0]
                rz_angle = params[layer, q, 1]
                if occlude == ("ansatz", layer, q, 0):
                    ry_angle = 0.0
                if occlude == ("ansatz", layer, q, 1):
                    rz_angle = 0.0
                qml.RY(ry_angle, wires=q)
                qml.RZ(rz_angle, wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

        return qml.expval(qml.PauliZ(0))

    return circuit()


def gate_attribution(x, params):
    """Occlusion attribution scores for every gate on input x, sorted descending."""
    baseline = _run_circuit(x, params, occlude=None)

    scores = []

    # Encoding gates
    for q in range(N_QUBITS):
        occluded_out = _run_circuit(x, params, occlude=("encoding", q))
        importance = abs(occluded_out - baseline)
        scores.append((f"encoding[qubit={q}] (feature_{q})", importance))

    # Ansatz gates
    rot_names = {0: "RY", 1: "RZ"}
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            for rot in range(N_ROTATIONS_PER_QUBIT):
                occluded_out = _run_circuit(
                    x, params, occlude=("ansatz", layer, q, rot)
                )
                importance = abs(occluded_out - baseline)
                scores.append(
                    (f"ansatz[layer={layer}, qubit={q}, {rot_names[rot]}]", importance)
                )

    scores.sort(key=lambda s: s[1], reverse=True)
    return baseline, scores


if __name__ == "__main__":
    from data import make_moons_dataset

    params = np.load("trained_params.npy")
    X_train, X_test, y_train, y_test = make_moons_dataset()

    # Run attribution on the first test sample
    x = X_test[0]
    y = y_test[0]
    baseline, scores = gate_attribution(x, params)

    print(f"Sample: x={x}, true label={y}, baseline output={baseline:.4f}")
    print(f"(predicted label: {0 if baseline > 0 else 1})\n")
    print("Gate attribution (occlusion), sorted by importance:")
    for name, score in scores:
        print(f"  {score:.4f}  {name}")
