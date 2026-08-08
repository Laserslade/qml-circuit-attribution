"""
train.py

Trains the classifier from circuit.py via PennyLane autodiff.
Labels {0,1} map to targets {+1,-1} against expval(PauliZ); squared-error
loss, threshold at 0 for classification.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's autograd-wrapped numpy

from data import make_moons_dataset
from circuit import classifier_circuit, N_PARAMS


def label_to_target(y):
    # y=0 -> +1, y=1 -> -1
    return 1.0 - 2.0 * y


def cost_fn(params, X, y):
    targets = label_to_target(y)
    preds = pnp.array([classifier_circuit(x, params) for x in X])
    return pnp.mean((preds - targets) ** 2)


def accuracy(params, X, y):
    preds = np.array([classifier_circuit(x, params) for x in X])
    pred_labels = (preds < 0).astype(int)  # expval<0 -> label 1, else label 0
    return np.mean(pred_labels == y)


def train(n_epochs=60, lr=0.3, seed=0, verbose=True, dataset_fn=None):
    if dataset_fn is None:
        dataset_fn = make_moons_dataset
    X_train, X_test, y_train, y_test = dataset_fn(seed=seed)

    rng = np.random.default_rng(seed)
    params = pnp.array(
        rng.uniform(-np.pi, np.pi, size=N_PARAMS), requires_grad=True
    )

    opt = qml.GradientDescentOptimizer(stepsize=lr)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        params, loss = opt.step_and_cost(
            lambda p: cost_fn(p, X_train, y_train), params
        )

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            train_acc = accuracy(params, X_train, y_train)
            test_acc = accuracy(params, X_test, y_test)
            history["epoch"].append(epoch)
            history["train_loss"].append(float(loss))
            history["train_acc"].append(float(train_acc))
            history["test_acc"].append(float(test_acc))
            if verbose:
                print(
                    f"Epoch {epoch:3d} | loss {loss:.4f} | "
                    f"train acc {train_acc:.3f} | test acc {test_acc:.3f}"
                )

    return params, history, (X_train, X_test, y_train, y_test)


def train_minibatch(n_steps=150, batch_size=16, lr=0.3, seed=0, verbose=True,
                     dataset_fn=None, eval_every=15, eval_subset=40,
                     optimizer="gd"):
    """
    Mini-batch training: samples a random batch each step instead of the
    full training set, needed for reasonable step time at higher qubit
    counts. Evaluation uses a random test subset per checkpoint, with a
    full-test-set pass at the end. optimizer: "gd" or "adam".
    """
    if dataset_fn is None:
        dataset_fn = make_moons_dataset
    X_train, X_test, y_train, y_test = dataset_fn(seed=seed)

    rng = np.random.default_rng(seed)
    params = pnp.array(
        rng.uniform(-np.pi, np.pi, size=N_PARAMS), requires_grad=True
    )

    if optimizer == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    else:
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    history = {"step": [], "loss": [], "eval_acc": []}

    for step in range(n_steps):
        idx = rng.choice(len(X_train), size=min(batch_size, len(X_train)), replace=False)
        X_batch, y_batch = X_train[idx], y_train[idx]

        params, loss = opt.step_and_cost(
            lambda p: cost_fn(p, X_batch, y_batch), params
        )

        if step % eval_every == 0 or step == n_steps - 1:
            eval_idx = rng.choice(len(X_test), size=min(eval_subset, len(X_test)), replace=False)
            eval_acc = accuracy(params, X_test[eval_idx], y_test[eval_idx])
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["eval_acc"].append(float(eval_acc))
            if verbose:
                print(f"Step {step:4d} | loss {loss:.4f} | eval acc (subset) {eval_acc:.3f}")

    final_test_acc = accuracy(params, X_test, y_test)
    if verbose:
        print(f"\nFinal accuracy on FULL test set: {final_test_acc:.3f}")

    return params, history, (X_train, X_test, y_train, y_test), final_test_acc


if __name__ == "__main__":
    params, history, data = train()
    np.save("trained_params.npy", np.array(params))
    print("\nSaved trained parameters to trained_params.npy")
    print("Final test accuracy:", history["test_acc"][-1])
