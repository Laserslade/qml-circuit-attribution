"""
validate.py

Ground-truth check for the attribution method: trains on
make_single_feature_dataset() (label depends only on feature 0), then
checks that attribution correctly ranks feature 0 above feature 1.
"""

import numpy as np

from data import make_single_feature_dataset
from train import train, accuracy
from attribution import gate_attribution


def run_validation(n_samples=20, seed=0):
    print("Training classifier on single-feature ground-truth dataset...")
    params, history, (X_train, X_test, y_train, y_test) = train(
        n_epochs=60, lr=0.3, seed=seed, verbose=False,
        dataset_fn=make_single_feature_dataset,
    )
    final_acc = accuracy(params, X_test, y_test)
    print(f"Final test accuracy: {final_acc:.3f}\n")

    if final_acc < 0.85:
        print(
            "WARNING: classifier accuracy is low. Attribution results on a "
            "poorly-trained model won't be meaningful — stopping before "
            "drawing conclusions."
        )
        return

    # Average encoding-gate importance for qubit 0 vs qubit 1 across samples
    q0_scores, q1_scores = [], []
    for i in range(min(n_samples, len(X_test))):
        x = X_test[i]
        baseline, scores = gate_attribution(x, params)
        score_dict = dict(scores)
        q0_scores.append(score_dict["encoding[qubit=0] (feature_0)"])
        q1_scores.append(score_dict["encoding[qubit=1] (feature_1)"])

    q0_mean, q1_mean = np.mean(q0_scores), np.mean(q1_scores)
    print(f"Mean encoding importance, feature 0 (should matter):     {q0_mean:.4f}")
    print(f"Mean encoding importance, feature 1 (should NOT matter): {q1_mean:.4f}")
    print(f"Ratio (feature0 / feature1): {q0_mean / max(q1_mean, 1e-9):.2f}x")

    if q0_mean > q1_mean:
        print("\nPASS: attribution correctly ranks feature 0 above feature 1.")
    else:
        print("\nFAIL: attribution did NOT rank feature 0 above feature 1 — investigate.")

    return q0_mean, q1_mean, final_acc


if __name__ == "__main__":
    run_validation()
