"""
visualize.py

Renders gate attribution scores as a heatmap over the circuit diagram:
each rotation gate is a colored box (color = importance), CNOTs are
drawn as plain connectors (unattributed, no trainable parameter).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from circuit import N_QUBITS, N_LAYERS

# Gate sequence per qubit, in circuit order. Each entry is either
# ("rot", label, lookup_key) for an attributed rotation gate, or
# ("cnot",) for an entangling gate (unattributed, drawn as a connector).
def _build_gate_sequence():
    seq = []
    seq.append(("rot", "RY\n(enc)", ("encoding",)))
    for layer in range(N_LAYERS):
        seq.append(("rot", "RY", ("ansatz", layer, 0)))
        seq.append(("rot", "RZ", ("ansatz", layer, 1)))
        seq.append(("cnot",))
    return seq


def plot_attribution_heatmap(x, scores, baseline, true_label=None, save_path=None):
    """scores: list of (gate_description, importance) from gate_attribution()."""
    # Re-key scores for placement lookup
    score_lookup = {}
    for name, val in scores:
        if name.startswith("encoding"):
            q = int(name.split("qubit=")[1].split("]")[0])
            score_lookup[("encoding", q)] = val
        else:
            layer = int(name.split("layer=")[1].split(",")[0])
            q = int(name.split("qubit=")[1].split(",")[0])
            rot = 0 if "RY" in name.split("]")[0].split(",")[-1] else 1
            score_lookup[("ansatz", layer, q, rot)] = val

    max_score = max(v for v in score_lookup.values()) or 1.0
    cmap = plt.get_cmap("Reds")

    gate_seq = _build_gate_sequence()

    fig, ax = plt.subplots(figsize=(11, 3 + 0.3 * N_QUBITS))

    box_w, box_h = 0.7, 0.6
    x_pos = 0.5
    x_positions_by_kind = []  # track x position for each seq entry, for CNOT lines

    for entry in gate_seq:
        x_positions_by_kind.append(x_pos)

        if entry[0] == "rot":
            _, label, key = entry
            for q in range(N_QUBITS):
                if key == ("encoding",):
                    score = score_lookup.get(("encoding", q), 0.0)
                    gate_label = label
                else:
                    _, layer, rot = key
                    score = score_lookup.get(("ansatz", layer, q, rot), 0.0)
                    gate_label = label

                color = cmap(score / max_score)
                rect = patches.FancyBboxPatch(
                    (x_pos - box_w / 2, N_QUBITS - 1 - q - box_h / 2),
                    box_w, box_h,
                    boxstyle="round,pad=0.02",
                    linewidth=1, edgecolor="black", facecolor=color,
                )
                ax.add_patch(rect)
                ax.text(
                    x_pos, N_QUBITS - 1 - q, gate_label,
                    ha="center", va="center", fontsize=8,
                )
            x_pos += 1.1

        elif entry[0] == "cnot":
            # control = qubit 0, target = qubit 1 (simplified: one connector per ring)
            ax.plot([x_pos, x_pos], [N_QUBITS - 1, 0], color="black", linewidth=1.5, zorder=1)
            ax.plot(x_pos, N_QUBITS - 1, "o", color="black", markersize=8, zorder=2)
            ax.plot(x_pos, 0, "o", color="white", markeredgecolor="black",
                     markersize=10, zorder=2)
            ax.text(x_pos, 0, "+", ha="center", va="center", fontsize=10, zorder=3)
            x_pos += 0.8

    # wires
    for q in range(N_QUBITS):
        ax.plot([0, x_pos], [N_QUBITS - 1 - q, N_QUBITS - 1 - q],
                 color="gray", linewidth=0.8, zorder=0)
        ax.text(-0.3, N_QUBITS - 1 - q, f"q{q}", ha="right", va="center", fontsize=10)

    ax.set_xlim(-0.7, x_pos + 0.3)
    ax.set_ylim(-0.8, N_QUBITS - 0.2)
    ax.axis("off")

    label_str = f", true label={true_label}" if true_label is not None else ""
    ax.set_title(
        f"Gate Attribution Heatmap  |  x={np.round(x, 2)}{label_str}, "
        f"baseline output={baseline:.3f}",
        fontsize=11,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_score))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Occlusion importance (|Δ output|)", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved heatmap to {save_path}")
    plt.close()


if __name__ == "__main__":
    from data import make_moons_dataset
    from attribution import gate_attribution

    params = np.load("trained_params.npy")
    X_train, X_test, y_train, y_test = make_moons_dataset()

    x, y = X_test[0], y_test[0]
    baseline, scores = gate_attribution(x, params)

    plot_attribution_heatmap(
        x, scores, baseline, true_label=y,
        save_path="../results/attribution_heatmap_sample0.png",
    )
