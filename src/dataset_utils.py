import csv
import os


# -----------------------------------------
# Dataset location
# -----------------------------------------

DATASET_PATH = "data/sign_landmarks.csv"


# -----------------------------------------
# Create dataset if it doesn't exist
# -----------------------------------------

def create_dataset():

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # If dataset already exists, don't overwrite it
    if os.path.exists(DATASET_PATH):
        return

    # Create feature names
    feature_names = [
        f"f{i}" for i in range(63)
    ]

    # Add label column
    header = feature_names + ["label"]

    # Create CSV file
    with open(
        DATASET_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

    print("Dataset created successfully!")
    print("Number of columns:", len(header))


# -----------------------------------------
# Save one sample
# -----------------------------------------

def save_sample(features, label):

    # Check feature count
    if len(features) != 63:

        print(
            "ERROR: Expected 63 features, "
            f"but received {len(features)}."
        )

        return

    # Make sure dataset exists
    create_dataset()

    # Append sample
    with open(
        DATASET_PATH,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            features + [label]
        )

    print(
        f"Sample saved successfully! "
        f"Label: {label}"
    )