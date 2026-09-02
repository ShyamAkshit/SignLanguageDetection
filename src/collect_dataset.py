import cv2
import mediapipe as mp
import csv
import os
import time

from feature_extraction import extract_landmarks
from dataset_utils import save_sample, create_dataset


# -----------------------------------------
# Configuration
# -----------------------------------------

TARGET_SAMPLES = 100

CAMERA_INDEX = 1

# Time between samples
SAMPLE_INTERVAL = 0.15

DATASET_PATH = "data/sign_landmarks.csv"


# -----------------------------------------
# Ask user for label
# -----------------------------------------

LABEL = input(
    "Enter the sign/letter you want to collect: "
).strip().upper()


# -----------------------------------------
# Validate label
# -----------------------------------------

if len(LABEL) != 1 or not LABEL.isalnum():

    print(
        "Invalid label. "
        "Please enter a single alphanumeric character such as A, B , 1, 0 etc."
    )

    exit()


# -----------------------------------------
# Count existing samples
# -----------------------------------------

def count_existing_samples(label):

    if not os.path.exists(DATASET_PATH):
        return 0

    count = 0

    with open(
        DATASET_PATH,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["label"] == label:
                count += 1

    return count


# -----------------------------------------
# Create dataset if required
# -----------------------------------------

create_dataset()


existing_samples = count_existing_samples(LABEL)


print()
print("----------------------------------")
print("Sign Language Dataset Collection")
print("----------------------------------")
print(f"Selected label: {LABEL}")
print(f"Existing samples: {existing_samples}")
print(f"Target samples: {TARGET_SAMPLES}")


# -----------------------------------------
# Check if target already reached
# -----------------------------------------

if existing_samples >= TARGET_SAMPLES:

    print()
    print(
        f"{LABEL} already has "
        f"{existing_samples} samples."
    )

    print("Nothing more to collect.")

    exit()


remaining_samples = TARGET_SAMPLES - existing_samples


print(f"Samples remaining: {remaining_samples}")
print()
print("Starting camera...")
print("Show the selected sign and move naturally.")
print("Press Q to stop collection.")
print()


# -----------------------------------------
# MediaPipe setup
# -----------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


MODEL_PATH = "models/hand_landmarker.task"


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1
)


# -----------------------------------------
# Open webcam
# -----------------------------------------

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not camera.isOpened():

    print("Could not open camera.")

    exit()


# -----------------------------------------
# Variables
# -----------------------------------------

timestamp = 0

last_sample_time = 0

current_samples = existing_samples


# -----------------------------------------
# Start MediaPipe
# -----------------------------------------

with HandLandmarker.create_from_options(options) as detector:

    while True:

        # ---------------------------------
        # Read webcam frame
        # ---------------------------------

        success, frame = camera.read()

        if not success:

            print("Could not read camera frame.")

            break


        # ---------------------------------
        # Flip image
        # ---------------------------------

        frame = cv2.flip(frame, 1)


        # ---------------------------------
        # Convert BGR → RGB
        # ---------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ---------------------------------
        # Create MediaPipe image
        # ---------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ---------------------------------
        # Timestamp
        # ---------------------------------

        timestamp += 1


        # ---------------------------------
        # Detect hand
        # ---------------------------------

        results = detector.detect_for_video(
            mp_image,
            timestamp
        )


        # ---------------------------------
        # Check hand detection
        # ---------------------------------

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]


            # ---------------------------------
            # Extract features
            # ---------------------------------

            features = extract_landmarks(hand)


            # ---------------------------------
            # Draw landmarks
            # ---------------------------------

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # ---------------------------------
            # Automatic sample collection
            # ---------------------------------

            current_time = time.time()


            if (
                current_time - last_sample_time
                >= SAMPLE_INTERVAL
            ):

                if current_samples < TARGET_SAMPLES:

                    save_sample(
                        features,
                        LABEL
                    )

                    current_samples += 1

                    last_sample_time = current_time


            # ---------------------------------
            # Detection status
            # ---------------------------------

            cv2.putText(
                frame,
                "Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        else:

            cv2.putText(
                frame,
                "No Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # ---------------------------------
        # Display selected label
        # ---------------------------------

        cv2.putText(
            frame,
            f"Label: {LABEL}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Display progress
        # ---------------------------------

        cv2.putText(
            frame,
            f"Samples: {current_samples}/{TARGET_SAMPLES}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Instructions
        # ---------------------------------

        cv2.putText(
            frame,
            "Move hand naturally",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to stop",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Show webcam
        # ---------------------------------

        cv2.imshow(
            "Sign Language Dataset Collection",
            frame
        )


        # ---------------------------------
        # Keyboard input
        # ---------------------------------

        key = cv2.waitKey(1) & 0xFF


        # Q → Stop
        if key == ord("q"):

            print()
            print("Collection stopped by user.")

            break


        # ---------------------------------
        # Target reached
        # ---------------------------------

        if current_samples >= TARGET_SAMPLES:

            print()
            print("----------------------------------")
            print("Target reached!")
            print(
                f"{LABEL}: "
                f"{current_samples} samples"
            )
            print("----------------------------------")

            break


# -----------------------------------------
# Cleanup
# -----------------------------------------

camera.release()

cv2.destroyAllWindows()

print("Dataset collection closed.")