"""Validate and summarize the tourism source data before pipeline use."""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing = EXPECTED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if data["ProdTaken"].isna().any():
        raise ValueError("ProdTaken contains missing values.")

    print("Dataset registration successful")
    print(f"Path: {DATA_PATH}")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print("Target distribution:")
    print(data["ProdTaken"].value_counts(normalize=True).sort_index().rename("proportion"))
    print("Missing values (non-zero only):")
    print(data.isna().sum().loc[lambda s: s.gt(0)].sort_values(ascending=False))


if __name__ == "__main__":
    main()
