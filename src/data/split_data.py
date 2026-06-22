"""
Script de split des données en ensembles d'entraînement et de test.
Variable cible : silica_concentrate (dernière colonne).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
RAW_PATH = os.path.join("data", "raw_data", "raw.csv")
PROCESSED_PATH = os.path.join("data", "processed_data")

TARGET = "silica_concentrate"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    if not os.path.exists(RAW_PATH):
        print("Téléchargement des données...")
        df = pd.read_csv(DATA_URL)
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
    else:
        df = pd.read_csv(RAW_PATH)

    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    X = df.drop(columns=[TARGET])
    y = df[[TARGET]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)

    print(f"Split effectué : {len(X_train)} train / {len(X_test)} test")
    print(f"Fichiers sauvegardés dans {PROCESSED_PATH}/")


if __name__ == "__main__":
    main()
