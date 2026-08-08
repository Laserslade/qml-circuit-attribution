"""
run_8q_training.py

Trains the classifier at N_QUBITS=8 using mini-batch Adam (full-batch
gradient descent is too slow at this scale). See PROGRESS.md /
README.md for background on why mini-batch training was adopted.
"""

import numpy as np
from train import train_minibatch
from data import make_high_dim_dataset

params, history, data, final_acc = train_minibatch(
    n_steps=110, batch_size=32, lr=0.05, seed=0, verbose=True,
    dataset_fn=make_high_dim_dataset, eval_every=10, eval_subset=40,
    optimizer="adam",
)
np.save("trained_params_8q.npy", np.array(params))
print("Saved trained_params_8q.npy")
print("Final test accuracy:", final_acc)
