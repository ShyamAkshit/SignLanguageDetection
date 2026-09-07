import csv
import os

# New version 2 dataset
DATASET_PATH = "data/sign_landmarks_v2.csv"


def create_dataset():
    """
    Create the V2 dataset CSV file if it does not already exist.
    """

    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATASET_PATH):
        return

    # 63 landmark features
    feature_names = [f"f{i}" for i in range(63)]

    # Add label and physical hand information
    header = feature_names + ["label", "hand"]

    with open(DATASET_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    print("New V2 dataset created successfully!")
    print("Dataset:", DATASET_PATH)
    print("Number of columns:", len(header))


def save_sample(features, label, hand):
    """
    Save one hand-landmark sample to the V2 dataset.
    """

    # Check feature count
    if len(features) != 63:
        print(
            "ERROR: Expected 63 features, "
            f"but received {len(features)}."
        )
        return

    # Check hand value
    if hand not in ["left", "right"]:
        print("ERROR: Hand must be 'left' or 'right'.")
        return

    # Make sure dataset exists
    create_dataset()

    # Save sample
    with open(DATASET_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(features + [label, hand])

    print(
        f"Sample saved successfully! "
        f"Label: {label}, Hand: {hand}"
    )