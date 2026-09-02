import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset_utils import DATASET_PATH


# =========================================
# PREPARE DATASET
# =========================================

def prepare_dataset(
    test_size=0.20,
    random_state=42
):

    # -------------------------------------
    # Load dataset
    # -------------------------------------

    df = pd.read_csv(DATASET_PATH)


    # -------------------------------------
    # Separate feature columns
    # -------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("f")
    ]


    # -------------------------------------
    # Separate X and y
    # -------------------------------------

    X = df[feature_columns].copy()

    y = df["label"].copy()


    # -------------------------------------
    # Find constant features
    # -------------------------------------

    constant_features = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]


    # -------------------------------------
    # Remove constant features
    # -------------------------------------

    X = X.drop(
        columns=constant_features
    )


    # -------------------------------------
    # Encode labels
    # -------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)


    # -------------------------------------
    # Train/Test Split
    # -------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y_encoded,

        test_size=test_size,

        random_state=random_state,

        stratify=y_encoded
    )


    # -------------------------------------
    # Return processed data
    # -------------------------------------

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        constant_features
    )


# =========================================
# TEST PIPELINE
# =========================================

if __name__ == "__main__":

    print("----------------------------------")
    print("Phase 4 - Final Preprocessing")
    print("----------------------------------")


    # -------------------------------------
    # Run preprocessing
    # -------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        constant_features
    ) = prepare_dataset()


    # -------------------------------------
    # Display results
    # -------------------------------------

    print("\nConstant features removed:")

    print(constant_features)


    print("\nNumber of final features:")

    print(X_train.shape[1])


    print("\nNumber of classes:")

    print(len(label_encoder.classes_))


    print("\nTraining feature shape:")

    print(X_train.shape)


    print("\nTesting feature shape:")

    print(X_test.shape)


    print("\nTraining label shape:")

    print(y_train.shape)


    print("\nTesting label shape:")

    print(y_test.shape)


    print("\nClasses:")

    print(label_encoder.classes_)


    print("\n----------------------------------")
    print("Final Preprocessing Complete")
    print("----------------------------------")