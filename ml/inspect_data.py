
from pathlib import Path
import pandas as pd

# Find the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"

# Find CSV files in the raw data folder.
csv_files = list(RAW_FOLDER.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Put your dataset inside data/raw/."
    )

if len(csv_files) > 1:
    raise ValueError(
        "More than one CSV found. Keep only the intended "
        "dataset in data/raw/ for this stage."
    )

# Load the dataset.
file_path = csv_files[0]
df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET INSPECTION REPORT")
print("=" * 60)

print("\nFile:", file_path.name)
print("\nDataset dimensions (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nNumerical summary:")
print(df.describe(include="all").T)

# Save a report that we can share after Stage 1.
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_FOLDER / "stage01_dataset_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Dataset file: {file_path.name}\n")
    file.write(f"Shape: {df.shape}\n")
    file.write(f"Columns: {df.columns.tolist()}\n\n")
    file.write("Data types:\n")
    file.write(df.dtypes.to_string())
    file.write("\n\nMissing values:\n")
    file.write(df.isnull().sum().to_string())
    file.write("\n\nDuplicate rows:\n")
    file.write(str(df.duplicated().sum()))
    file.write("\n\nSummary statistics:\n")
    file.write(df.describe(include="all").T.to_string())

print("\nReport saved to outputs/stage01_dataset_report.txt")