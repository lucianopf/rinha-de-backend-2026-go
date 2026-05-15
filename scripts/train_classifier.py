#!/usr/bin/env python3
"""
Train logistic regression classifier on 3M reference vectors.
Output: resources/model.bin (bias + 14 weights, float64 little-endian)

Binary format:
  [0:8]    float64 bias (intercept)
  [8:120]  float64 weights[14]

Total: 120 bytes. Inference: score = sigmoid(bias + sum(weight[i] * feature[i]))
"""

import gzip
import json
import struct
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load references.json.gz → X (N,14) float32, y (N,) int32 (0=legit, 1=fraud)."""
    print(f"Loading {path}...")
    t0 = time.time()

    with gzip.open(path, "rt") as f:
        data = json.load(f)

    n = len(data)
    X = np.empty((n, 14), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)

    for i, entry in enumerate(data):
        X[i] = entry["vector"]
        y[i] = 1 if entry["label"] == "fraud" else 0

    elapsed = time.time() - t0
    fraud_rate = y.sum() / n
    print(f"  Loaded {n:,} vectors in {elapsed:.1f}s")
    print(f"  Fraud rate: {fraud_rate:.4f} ({y.sum():,} fraud, {n - y.sum():,} legit)")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train logistic regression with class balancing."""
    print("\nSplitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Train with balanced class weights
    print("\nTraining LogisticRegression (balanced, liblinear, max_iter=200)...")
    t0 = time.time()
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (600,000 samples)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"], digits=4))
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(f"  FP rate: {fp_rate:.4%}  FN rate: {fn_rate:.4%}  Accuracy: {accuracy:.4%}")

    # Full dataset evaluation
    print("\n" + "=" * 60)
    print("FULL DATASET EVALUATION (3,000,000 samples)")
    print("=" * 60)
    y_full_pred = model.predict(X)
    cm_full = confusion_matrix(y, y_full_pred)
    fp_rate_full = cm_full[0, 1] / (cm_full[0, 0] + cm_full[0, 1])
    fn_rate_full = cm_full[1, 0] / (cm_full[1, 0] + cm_full[1, 1])
    accuracy_full = (cm_full[0, 0] + cm_full[1, 1]) / cm_full.sum()
    print(f"  TN={cm_full[0,0]:,}  FP={cm_full[0,1]:,}")
    print(f"  FN={cm_full[1,0]:,}  TP={cm_full[1,1]:,}")
    print(f"  FP rate: {fp_rate_full:.4%}  FN rate: {fn_rate_full:.4%}  Accuracy: {accuracy_full:.4%}")

    return model


def export_model(model: LogisticRegression, path: str):
    """Export weights + bias as binary (float64 LE)."""
    bias = float(model.intercept_[0])
    weights = model.coef_[0].astype(np.float64)

    print(f"\nExporting model to {path}...")
    print(f"  Bias: {bias:.6f}")
    for i, w in enumerate(weights):
        print(f"  Weight[{i:2d}]: {w:+.6f}")

    with open(path, "wb") as f:
        f.write(struct.pack("<d", bias))
        for w in weights:
            f.write(struct.pack("<d", float(w)))

    size = 8 + 14 * 8
    print(f"  Written {size} bytes")


def main():
    data_path = "resources/references.json.gz"
    output_path = "resources/model.bin"

    X, y = load_data(data_path)
    model = train_model(X, y)
    export_model(model, output_path)

    # Quick sanity: sigmoid of dot product for a few samples
    print("\nSanity check (first 5 samples):")
    bias = model.intercept_[0]
    weights = model.coef_[0]
    for i in range(5):
        z = bias + np.dot(weights, X[i])
        prob = 1.0 / (1.0 + np.exp(-z))
        actual = "fraud" if y[i] else "legit"
        print(f"  [{i}] prob={prob:.4f} → {'fraud' if prob >= 0.5 else 'legit'} (actual={actual})")


if __name__ == "__main__":
    main()
