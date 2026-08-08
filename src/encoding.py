"""
encoding.py

Angle encoding: maps each classical feature onto one qubit via an
RY rotation (one feature -> one qubit).
"""

import pennylane as qml
import numpy as np


def angle_encoding(x, wires):
    """Encode feature vector x onto wires via RY rotations, one feature per qubit."""
    assert len(x) == len(wires), "Need one feature per wire for angle encoding"
    for xi, w in zip(x, wires):
        qml.RY(xi, wires=w)


if __name__ == "__main__":
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        angle_encoding(x, wires=range(n_qubits))
        return qml.state()

    sample = np.array([0.5, -1.2])
    state = circuit(sample)
    print("Encoded state for x =", sample)
    print(state)
    print("\nCircuit diagram:")
    print(qml.draw(circuit)(sample))
