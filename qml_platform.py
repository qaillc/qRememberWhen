#!/usr/bin/env python3
"""
QubitPred: Quantum Machine Learning Platform MVP using Qiskit 2 and Aer Simulator
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit import Aer, transpile
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA

def load_data(csv_path, test_size, seed):
    if csv_path:
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    else:
        from qiskit_machine_learning.datasets import ad_hoc_data
        X_train, X_test, y_train, y_test = ad_hoc_data(
            training_size=100, test_size=test_size, n=2, gap=0.3, plot_data=False
        )
        return X_train, X_test, y_train, y_test

def build_feature_map(num_features, reps):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps)

def build_ansatz(num_qubits, reps):
    return TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks='ry',
        entanglement_blocks='cz',
        reps=reps
    )

def main():
    parser = argparse.ArgumentParser(description="QubitPred QML Platform")
    parser.add_argument("--data", type=str, default=None,
                        help="CSV file path (last column label)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion")
    parser.add_argument("--reps", type=int, default=2,
                        help="Repetitions for feature map & ansatz")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    algorithm_globals.random_seed = args.seed

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(args.data, args.test_size, args.seed)
    num_features = X_train.shape[1]
    print(f"Number of qubits/features: {num_features}")

    print("Building feature map...")
    feature_map = build_feature_map(num_features, args.reps)
    print("Building ansatz...")
    ansatz = build_ansatz(num_features, args.reps)

    backend = Aer.get_backend("statevector_simulator")

    print("Training VQC...")
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=100),
        quantum_instance=backend
    )
    vqc.fit(X_train, y_train)

    print("Evaluating on test set...")
    acc = vqc.score(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
