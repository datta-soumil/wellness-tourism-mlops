"""Clean the source data and create reproducible train/test split artifacts."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]


def main() -> None:
    data = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    # Standardize a spelling variant before fitting the categorical encoder.
    data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})
    data = data.drop(columns=DROP_COLUMNS, errors="ignore")
    if TARGET not in data:
        raise ValueError(f"{TARGET} is not present in the dataset.")

    features = data.drop(columns=TARGET)
    target = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(ARTIFACT_DIR / "X_train.csv", index=False)
    X_test.to_csv(ARTIFACT_DIR / "X_test.csv", index=False)
    y_train.to_frame(TARGET).to_csv(ARTIFACT_DIR / "y_train.csv", index=False)
    y_test.to_frame(TARGET).to_csv(ARTIFACT_DIR / "y_test.csv", index=False)
    print(f"Saved split artifacts to {ARTIFACT_DIR}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Features used ({len(features.columns)}): {', '.join(features.columns)}")


if __name__ == "__main__":
    main()
