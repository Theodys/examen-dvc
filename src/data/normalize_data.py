"""
Script de normalisation des données d'entraînement et de test.
Utilise StandardScaler pour mettre les features à la même échelle.
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"


def main():
    X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv"))

    print(f"Normalisation des données ({X_train.shape[1]} features)...")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    X_train_scaled.to_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_PATH, "X_test_scaled.csv"), index=False)

    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

    print("Normalisation terminée.")
    print(f"Fichiers sauvegardés dans {PROCESSED_PATH}/")


if __name__ == "__main__":
    main()
