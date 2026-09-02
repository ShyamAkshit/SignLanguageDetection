import pandas as pd

from dataset_utils import DATASET_PATH


# -----------------------------------------
# Load dataset
# -----------------------------------------

df = pd.read_csv(DATASET_PATH)


print("----------------------------------")
print("Dataset Quality Inspection")
print("----------------------------------")


# -----------------------------------------
# 1. Dataset shape
# -----------------------------------------

print("\n1. Dataset Shape")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# -----------------------------------------
# 2. Feature and label separation
# -----------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column.startswith("f")
]

X = df[feature_columns]

y = df["label"]


print("\n2. Dataset Components")

print("Number of features:", X.shape[1])
print("Number of labels:", y.shape[0])


# -----------------------------------------
# 3. Class distribution
# -----------------------------------------

print("\n3. Class Distribution")

print(y.value_counts().sort_index())


# -----------------------------------------
# 4. Missing values
# -----------------------------------------

print("\n4. Missing Values")

total_missing = df.isnull().sum().sum()

print("Total missing values:", total_missing)


# -----------------------------------------
# 5. Data types
# -----------------------------------------

print("\n5. Data Types")

non_numeric = []

for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):

        non_numeric.append(column)


if len(non_numeric) == 0:

    print("All 63 features are numeric.")

else:

    print(
        "Non-numeric features:",
        non_numeric
    )


# -----------------------------------------
# 6. Feature minimum
# -----------------------------------------

print("\n6. Feature Minimum Values")

print(
    X.min().head(10)
)


# -----------------------------------------
# 7. Feature maximum
# -----------------------------------------

print("\n7. Feature Maximum Values")

print(
    X.max().head(10)
)


# -----------------------------------------
# 8. Feature mean
# -----------------------------------------

print("\n8. Feature Mean Values")

print(
    X.mean().head(10)
)


# -----------------------------------------
# 9. Feature standard deviation
# -----------------------------------------

print("\n9. Feature Standard Deviation")

print(
    X.std().head(10)
)


# -----------------------------------------
# 10. Duplicate samples
# -----------------------------------------

print("\n10. Duplicate Samples")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# -----------------------------------------
# 11. Constant features
# -----------------------------------------

print("\n11. Constant Features")

constant_features = []

for column in feature_columns:

    if X[column].nunique() <= 1:

        constant_features.append(column)


if len(constant_features) == 0:

    print("No constant features found.")

else:

    print(
        "Constant features:",
        constant_features
    )


# -----------------------------------------
# Final summary
# -----------------------------------------

print("\n----------------------------------")
print("Quality Inspection Complete")
print("----------------------------------")