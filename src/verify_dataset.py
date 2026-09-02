import pandas as pd

from dataset_utils import DATASET_PATH


# -----------------------------------------
# Load dataset
# -----------------------------------------

df = pd.read_csv(DATASET_PATH)


print("----------------------------------")
print("Dataset Verification")
print("----------------------------------")


# -----------------------------------------
# Dataset shape
# -----------------------------------------

print("\nDataset shape:")
print(df.shape)


# -----------------------------------------
# Number of samples
# -----------------------------------------

print("\nTotal samples:")
print(len(df))


# -----------------------------------------
# Number of features
# -----------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column.startswith("f")
]

print("\nNumber of features:")
print(len(feature_columns))


# -----------------------------------------
# Labels
# -----------------------------------------

print("\nLabel distribution:")
print(df["label"].value_counts())


# -----------------------------------------
# Missing values
# -----------------------------------------

print("\nMissing values:")

missing_values = df.isnull().sum().sum()

print(missing_values)


# -----------------------------------------
# Check numeric features
# -----------------------------------------

print("\nChecking feature data types...")

non_numeric_features = []

for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        non_numeric_features.append(column)


if len(non_numeric_features) == 0:

    print("All 63 features are numeric.")

else:

    print(
        "Non-numeric features:",
        non_numeric_features
    )


# -----------------------------------------
# Duplicate rows
# -----------------------------------------

print("\nDuplicate samples:")

duplicates = df.duplicated().sum()

print(duplicates)


# -----------------------------------------
# Final summary
# -----------------------------------------

print("\n----------------------------------")
print("Verification Complete")
print("----------------------------------")