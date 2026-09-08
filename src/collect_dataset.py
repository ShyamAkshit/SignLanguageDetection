import cv2
import time
import csv
import os
import mediapipe as mp

from feature_extraction import extract_landmarks
from dataset_utils import save_sample, DATASET_PATH


# ============================================================
# Configuration
# ============================================================

TARGET_SAMPLES = 200
CAMERA_INDEX = 1
SAMPLE_INTERVAL = 0.15

# Hand-detection safety settings
MIN_HAND_SCORE = 0.80
STABLE_FRAMES_REQUIRED = 5
FRAME_MARGIN = 0.05


# ============================================================
# MediaPipe setup
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# Count existing samples for label + hand
# ============================================================

def count_existing_samples(label, hand):
    """
    Count existing samples for a specific sign and hand.
    """

    if not os.path.exists(DATASET_PATH):
        return 0

    count = 0

    with open(DATASET_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["label"] == label and row["hand"] == hand:
                count += 1

    return count


# ============================================================
# Check whether all landmarks are inside safe camera area
# ============================================================

def hand_inside_safe_area(hand_landmarks):
    """
    Return True only if all 21 landmarks are safely inside
    the camera frame.

    FRAME_MARGIN = 0.05 means the hand must stay at least
    5% away from every edge of the frame.
    """

    min_x = FRAME_MARGIN
    max_x = 1.0 - FRAME_MARGIN

    min_y = FRAME_MARGIN
    max_y = 1.0 - FRAME_MARGIN

    for landmark in hand_landmarks:

        if landmark.x < min_x:
            return False

        if landmark.x > max_x:
            return False

        if landmark.y < min_y:
            return False

        if landmark.y > max_y:
            return False

    return True


# ============================================================
# User input
# ============================================================

print("----------------------------------")
print("Sign Language Dataset V2 Collector")
print("----------------------------------")

print("\nAvailable hands:")
print("1. left")
print("2. right")

while True:

    hand_choice = input("\nSelect hand (1/2): ").strip()

    if hand_choice == "1":
        selected_hand = "left"
        break

    elif hand_choice == "2":
        selected_hand = "right"
        break

    else:
        print("Invalid choice. Please enter 1 or 2.")


print("\nAvailable signs:")
print("0-9 and A-Z")

valid_labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

while True:

    selected_label = input(
        "Enter sign label: "
    ).strip().upper()

    if selected_label in valid_labels:
        break

    print(
        "Invalid label. Please enter one character "
        "from 0-9 or A-Z."
    )


# ============================================================
# Check existing samples
# ============================================================

existing_samples = count_existing_samples(
    selected_label,
    selected_hand
)

remaining_samples = TARGET_SAMPLES - existing_samples


print("\n----------------------------------")
print("Collection Settings")
print("----------------------------------")
print(f"Hand: {selected_hand}")
print(f"Label: {selected_label}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Existing samples: {existing_samples}")
print(f"Remaining samples: {max(remaining_samples, 0)}")
print("----------------------------------")


if existing_samples >= TARGET_SAMPLES:

    print(
        "\nTarget already reached for this "
        "sign and hand."
    )

    print("No additional samples will be collected.")

    exit()


# ============================================================
# Create MediaPipe detector
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


# ============================================================
# Start detector
# ============================================================

with HandLandmarker.create_from_options(options) as detector:

    # Camera index 0 for this PC
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print("\nERROR: Could not open camera.")
        exit()


    print("\nCamera opened successfully.")

    print(
        f"Show your {selected_hand} hand."
    )

    print(
        "Keep the entire hand visible "
        "inside the camera frame."
    )

    print("Press Q to stop collection.")

    print("----------------------------------")


    # ========================================================
    # Collection variables
    # ========================================================

    sample_count = existing_samples

    last_sample_time = 0

    frame_timestamp_ms = 0

    stable_hand_frames = 0

    hand_is_ready = False


    # ========================================================
    # Main collection loop
    # ========================================================

    while sample_count < TARGET_SAMPLES:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read frame.")
            break


        # IMPORTANT:
        # Do NOT flip the frame before MediaPipe.
        #
        # This keeps handedness correct:
        #
        # Physical left  -> MediaPipe left
        # Physical right -> MediaPipe right

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        frame_timestamp_ms += 33


        results = detector.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )


        detected_hand = None
        hand_score = 0.0

        status = "NO HAND"


        # ====================================================
        # Hand detected
        # ====================================================

        if results.hand_landmarks:

            hand_landmarks = results.hand_landmarks[0]


            detected_hand = (
                results.handedness[0][0]
                .category_name
                .lower()
            )


            hand_score = (
                results.handedness[0][0]
                .score
            )


            # =================================================
            # Draw landmarks
            # =================================================

            for landmark in hand_landmarks:

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # Draw hand connections
            # =================================================

            connections = (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            )


            for connection in connections:

                start = hand_landmarks[
                    connection.start
                ]

                end = hand_landmarks[
                    connection.end
                ]


                start_point = (
                    int(
                        start.x *
                        frame.shape[1]
                    ),
                    int(
                        start.y *
                        frame.shape[0]
                    )
                )


                end_point = (
                    int(
                        end.x *
                        frame.shape[1]
                    ),
                    int(
                        end.y *
                        frame.shape[0]
                    )
                )


                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )


            # =================================================
            # Safety check 1:
            # Handedness confidence
            # =================================================

            confidence_ok = (
                hand_score >= MIN_HAND_SCORE
            )


            # =================================================
            # Safety check 2:
            # Hand inside frame
            # =================================================

            position_ok = (
                hand_inside_safe_area(
                    hand_landmarks
                )
            )


            # =================================================
            # Safety check 3:
            # Correct hand
            # =================================================

            hand_ok = (
                detected_hand == selected_hand
            )


            # =================================================
            # Determine whether frame is valid
            # =================================================

            frame_is_valid = (
                confidence_ok
                and position_ok
                and hand_ok
            )


            # =================================================
            # Stable-hand counter
            # =================================================

            if frame_is_valid:

                stable_hand_frames += 1

            else:

                # Any invalid frame resets stability.
                stable_hand_frames = 0
                hand_is_ready = False


            # =================================================
            # Check stability requirement
            # =================================================

            if (
                stable_hand_frames
                >= STABLE_FRAMES_REQUIRED
            ):

                hand_is_ready = True


            # =================================================
            # Sample collection
            # =================================================

            if hand_is_ready:

                current_time = time.time()


                if (
                    current_time
                    - last_sample_time
                    >= SAMPLE_INTERVAL
                ):

                    features = extract_landmarks(
                        hand_landmarks
                    )


                    save_sample(
                        features,
                        selected_label,
                        selected_hand
                    )


                    sample_count += 1


                    last_sample_time = (
                        current_time
                    )


                    status = "SAMPLE SAVED"


            # =================================================
            # Display status
            # =================================================

            if not confidence_ok:

                status = "LOW CONFIDENCE"

            elif not position_ok:

                status = "HAND TOO CLOSE TO EDGE"

            elif not hand_ok:

                status = "WRONG HAND"

            elif not hand_is_ready:

                status = (
                    f"STABILIZING "
                    f"{stable_hand_frames}/"
                    f"{STABLE_FRAMES_REQUIRED}"
                )

            else:

                status = "READY"


        # ====================================================
        # No hand detected
        # ====================================================

        else:

            stable_hand_frames = 0
            hand_is_ready = False
            last_sample_time = 0

            status = "NO HAND"


        # ====================================================
        # Display detected hand
        # ====================================================

        if detected_hand:

            hand_text = (
                f"Detected: {detected_hand} "
                f"({hand_score:.2f})"
            )

            cv2.putText(
                frame,
                hand_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # Display status
        # ====================================================

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Display collection information
        # ====================================================

        progress_text = (
            f"Sign: {selected_label} | "
            f"Hand: {selected_hand} | "
            f"Samples: "
            f"{sample_count}/{TARGET_SAMPLES}"
        )


        cv2.putText(
            frame,
            progress_text,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Show camera
        # ====================================================

        cv2.imshow(
            "Sign Language Dataset Collector V2",
            frame
        )


        # ====================================================
        # Exit
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            print(
                "\nCollection stopped by user."
            )

            break


    # ========================================================
    # Cleanup
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# Final message
# ============================================================

print("\n----------------------------------")
print("Collection session complete.")
print(
    f"Final samples for "
    f"{selected_label} / "
    f"{selected_hand}: "
    f"{sample_count}/{TARGET_SAMPLES}"
)
print("----------------------------------")