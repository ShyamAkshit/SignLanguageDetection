import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.dataset_utils import DATASET_PATH


def prepare_dataset_v2(test_size=0.20, random_state=42):

    # Load V2 dataset
    df = pd.read_csv(DATASET_PATH)

    # Identify landmark feature columns
    feature_columns = [column for column in df.columns if column.startswith("f")]

    # Separate features and labels
    X = df[feature_columns]
    y = df["label"]

    # Keep hand information for analysis
    hand = df["hand"]

    # Detect constant features automatically
    constant_features = X.columns[X.nunique() <= 1].tolist()

    # Remove constant features
    X = X.drop(columns=constant_features)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create a combined stratification key using sign + hand
    stratify_key = y.astype(str) + "_" + hand.astype(str)

    # Stratified train/test split
    X_train, X_test, y_train, y_test, hand_train, hand_test = train_test_split(
        X,
        y_encoded,
        hand,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key
)


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

    print("----------------------------------")
    print("Testing V2 Preprocessing Pipeline")
    print("----------------------------------")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        hand_train,
        hand_test,
        label_encoder,
        constant_features
    ) = prepare_dataset_v2()

    print("\nDataset preprocessing complete.")

    print("\nOriginal features:", 63)
    print("Constant features:", constant_features)
    print("Final features:", X_train.shape[1])

    print("\nTraining data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("\nTraining labels:", y_train.shape)
    print("Testing labels:", y_test.shape)

    print("\nTraining hand distribution:")
    print(hand_train.value_counts().to_dict())

    print("\nTesting hand distribution:")
    print(hand_test.value_counts().to_dict())

    print("\nNumber of classes:", len(label_encoder.classes_))

    print("\nClasses:")
    print(label_encoder.classes_)

    print("----------------------------------")