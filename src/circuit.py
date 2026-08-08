"""
circuit.py

Variational ansatz + full classifier circuit (encoding + ansatz +
measurement). Ansatz: RY + RZ per qubit per layer, then a CNOT ring.
Named/indexable layers so attribution.py can target individual gates.
"""

import pennylane as qml
import numpy as np
from encoding import angle_encoding

N_QUBITS = 2
N_LAYERS = 3
N_PARAMS = N_LAYERS * N_QUBITS * 2  # RY + RZ per qubit per layer


def ansatz(params, wires, n_layers=N_LAYERS):
    """params layout: [layer, qubit, 0]=RY angle, [layer, qubit, 1]=RZ angle."""
    params = params.reshape(n_layers, len(wires), 2)
    for layer in range(n_layers):
        for q_idx, w in enumerate(wires):
            qml.RY(params[layer, q_idx, 0], wires=w)
            qml.RZ(params[layer, q_idx, 1], wires=w)
        for q_idx in range(len(wires)):
            qml.CNOT(wires=[wires[q_idx], wires[(q_idx + 1) % len(wires)]])


dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def classifier_circuit(x, params):
    """Full circuit: encode x, apply ansatz, measure PauliZ on qubit 0."""
    angle_encoding(x, wires=range(N_QUBITS))
    ansatz(params, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, size=N_PARAMS)
    x = rng.uniform(-np.pi, np.pi, size=N_QUBITS)

    output = classifier_circuit(x, params)
    print("Sample output (expval PauliZ on qubit 0):", output)
    print("\nCircuit diagram:")
    print(qml.draw(classifier_circuit)(x, params))
    print("\nTotal trainable parameters:", N_PARAMS)
