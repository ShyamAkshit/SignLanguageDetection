import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset_utils import DATASET_PATH


def prepare_dataset_v2(test_size=0.20, random_state=42):

    print("=" * 60)
    print("V2 DATASET PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------

    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Original shape: {df.shape}")

    # --------------------------------------------------
    # 2. BASIC VALIDATION
    # --------------------------------------------------

    feature_columns = [
        column for column in df.columns
        if column.startswith("f")
    ]

    required_columns = feature_columns + ["label", "hand"]

    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 feature columns, "
            f"but found {len(feature_columns)}."
        )

    if list(df.columns) != required_columns:
        raise ValueError(
            "Dataset column structure is incorrect."
        )

    if df.isna().sum().sum() != 0:
        raise ValueError("Dataset contains missing values.")

    if df.duplicated().sum() != 0:
        raise ValueError("Dataset contains duplicate rows.")

    print(f"Feature count: {len(feature_columns)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print(f"Hands: {sorted(df['hand'].unique())}")

    # --------------------------------------------------
    # 3. VERIFY CLASS DISTRIBUTION
    # --------------------------------------------------

    print("\nClass distribution:")

    class_counts = df["label"].value_counts().sort_index()
    print(class_counts)

    # SPACE must exist
    if "SPACE" not in df["label"].unique():
        raise ValueError("SPACE class is missing from V2 dataset.")

    # --------------------------------------------------
    # 4. SEPARATE FEATURES / LABEL / HAND
    # --------------------------------------------------

    X = df[feature_columns].copy()
    y = df["label"].copy()
    hand = df["hand"].copy()

    print("\nBefore constant feature removal:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------
    # 5. REMOVE CONSTANT FEATURES
    # --------------------------------------------------

    constant_features = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]

    print("\nConstant features:")
    print(constant_features)

    X = X.drop(columns=constant_features)

    print(f"Features after removal: {X.shape[1]}")
    print(f"X shape after preprocessing: {X.shape}")

    # --------------------------------------------------
    # 6. ENCODE LABELS
    # --------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print("\nLabel encoding:")
    print(f"Number of encoded classes: {len(label_encoder.classes_)}")
    print("Classes:")
    print(list(label_encoder.classes_))

    # --------------------------------------------------
    # 7. CREATE STRATIFICATION GROUP
    # --------------------------------------------------
    # Keep both label AND hand balanced in train/test.

    stratify_group = (
        y.astype(str) + "_" + hand.astype(str)
    )

    # --------------------------------------------------
    # 8. TRAIN / TEST SPLIT
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        hand_train,
        hand_test
    ) = train_test_split(
        X,
        y_encoded,
        hand,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_group
    )

    print("\nTrain/Test Split:")
    print(f"Train X: {X_train.shape}")
    print(f"Test X:  {X_test.shape}")
    print(f"Train y: {y_train.shape}")
    print(f"Test y:  {y_test.shape}")

    # --------------------------------------------------
    # 9. HAND DISTRIBUTION
    # --------------------------------------------------

    print("\nTraining hand distribution:")
    print(hand_train.value_counts())

    print("\nTesting hand distribution:")
    print(hand_test.value_counts())

    # --------------------------------------------------
    # 10. SPACE DISTRIBUTION
    # --------------------------------------------------

    train_space = (
        y_train ==
        label_encoder.transform(["SPACE"])[0]
    ).sum()

    test_space = (
        y_test ==
        label_encoder.transform(["SPACE"])[0]
    ).sum()

    print("\nSPACE distribution:")
    print(f"Train SPACE: {train_space}")
    print(f"Test SPACE:  {test_space}")

    # --------------------------------------------------
    # 11. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("V2 PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nOriginal samples: {len(df)}")
    print(f"Final features: {X.shape[1]}")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Removed features: {constant_features}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        hand_train,
        hand_test,
        label_encoder,
        constant_features
    )


if __name__ == "__main__":

    prepare_dataset_v2()