"""
data.py

Dataset generators for the VQC classifier.
- make_moons_dataset: 2 features, both matter (nonlinear boundary).
- make_single_feature_dataset: ground-truth check — only feature 0 matters.
- make_high_dim_dataset: 8 features, for testing at larger qubit counts.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def make_moons_dataset(n_samples=200, noise=0.15, test_size=0.2, seed=42):
    """Standard moons dataset, features scaled to [-pi, pi] for angle encoding."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # Scale each feature independently to [-pi, pi] so it's a valid rotation angle
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[1]):
        col = X[:, i]
        col_min, col_max = col.min(), col.max()
        X_scaled[:, i] = 2 * np.pi * (col - col_min) / (col_max - col_min) - np.pi

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_single_feature_dataset(n_samples=200, seed=42):
    """Ground-truth check: label = feature_0 > 0. feature_1 is pure noise."""
    rng = np.random.default_rng(seed)
    feature_0 = rng.uniform(-np.pi, np.pi, size=n_samples)
    feature_1 = rng.uniform(-np.pi, np.pi, size=n_samples)  # noise, irrelevant
    X = np.stack([feature_0, feature_1], axis=1)
    y = (feature_0 > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_high_dim_dataset(n_samples=250, n_features=8, n_informative=6,
                           test_size=0.2, seed=42):
    """8-feature classification dataset for testing at larger qubit counts."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=0,
        n_clusters_per_class=1, random_state=seed,
    )

    X_scaled = np.zeros_like(X)
    for i in range(X.shape[1]):
        col = X[:, i]
        col_min, col_max = col.min(), col.max()
        X_scaled[:, i] = 2 * np.pi * (col - col_min) / (col_max - col_min) - np.pi

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick sanity check when run directly
    X_train, X_test, y_train, y_test = make_moons_dataset()
    print("Moons dataset:")
    print("  X_train shape:", X_train.shape, " y_train shape:", y_train.shape)
    print("  X range:", X_train.min(), "to", X_train.max())
    print("  Class balance (train):", np.bincount(y_train))

    X_train2, X_test2, y_train2, y_test2 = make_single_feature_dataset()
    print("\nSingle-feature validation dataset:")
    print("  X_train shape:", X_train2.shape)
    print("  Class balance (train):", np.bincount(y_train2))
