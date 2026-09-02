import pandas as pd

from dataset_utils import DATASET_PATH


# -----------------------------------------
# Load dataset
# -----------------------------------------

df = pd.read_csv(DATASET_PATH)


# -----------------------------------------
# Select feature columns
# -----------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column.startswith("f")
]


X = df[feature_columns]


# -----------------------------------------
# Find constant features
# -----------------------------------------

constant_features = []

for column in X.columns:

    unique_values = X[column].nunique()

    if unique_values <= 1:

        constant_features.append(column)


# -----------------------------------------
# Display results
# -----------------------------------------

print("----------------------------------")
print("Constant Feature Detection")
print("----------------------------------")

print("\nTotal features:")
print(X.shape[1])

print("\nConstant features:")
print(constant_features)

print("\nNumber of constant features:")
print(len(constant_features))

print("\nFeatures after removal:")
print(
    X.shape[1] - len(constant_features)
)

print("----------------------------------")