import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_landmarks
from realtime_features import prepare_realtime_features


# =========================================
# 1. FILE PATHS
# =========================================

MODEL_PATH = "models/hand_landmarker.task"

SVM_PATH = "models/sign_language_svm_v2.joblib"

LABEL_ENCODER_PATH = "models/label_encoder_v2.joblib"


# =========================================
# 2. SETTINGS
# =========================================

CAMERA_INDEX = 0

PREDICTION_HISTORY_SIZE = 10

STABLE_FRAMES_REQUIRED = 7


# =========================================
# 3. LOAD MODEL
# =========================================

print("----------------------------------")
print("Real-Time Sign Language Stitching V1")
print("----------------------------------")

print("\nLoading V2 SVM model...")

model = joblib.load(SVM_PATH)

print("V2 SVM model loaded successfully.")


print("\nLoading V2 label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print("V2 label encoder loaded successfully.")


# =========================================
# 4. OPEN CAMERA
# =========================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("\nERROR: Could not open camera.")

    exit()


print("\nCamera opened successfully.")

print("Press Q to quit.")
print("Press C to clear stitched text.")


# =========================================
# 5. MEDIAPIPE SETUP
# =========================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1
)


detector = HandLandmarker.create_from_options(
    options
)


# =========================================
# 6. HAND CONNECTIONS
# =========================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


# =========================================
# 7. STATE VARIABLES
# =========================================

frame_timestamp = 0

prediction_history = []

current_stable_label = None

stable_frame_count = 0

last_added_label = None

stitched_text = ""


# =========================================
# 8. MAIN LOOP
# =========================================

while True:

    success, frame = camera.read()


    if not success:

        print("ERROR: Could not read frame.")

        break


    # -------------------------------------
    # Convert BGR → RGB
    # -------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -------------------------------------
    # Create MediaPipe image
    # -------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # -------------------------------------
    # Timestamp
    # -------------------------------------

    frame_timestamp += 1


    # -------------------------------------
    # Detect hand
    # -------------------------------------

    results = detector.detect_for_video(

        mp_image,

        frame_timestamp
    )


    # =====================================
    # HAND DETECTED
    # =====================================

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]


        # ---------------------------------
        # Extract 63 features
        # ---------------------------------

        features = extract_landmarks(
            hand
        )


        # ---------------------------------
        # Convert 63 → 60
        # ---------------------------------

        processed_features = (
            prepare_realtime_features(
                features
            )
        )


        # ---------------------------------
        # Create DataFrame
        # ---------------------------------

        feature_names = [
            f"f{i}"
            for i in range(3, 63)
        ]


        input_data = pd.DataFrame(

            [processed_features],

            columns=feature_names
        )


        # ---------------------------------
        # Predict
        # ---------------------------------

        prediction = model.predict(
            input_data
        )


        predicted_class_id = prediction[0]


        # =================================
        # PREDICTION HISTORY
        # =================================

        prediction_history.append(
            predicted_class_id
        )


        if len(prediction_history) > PREDICTION_HISTORY_SIZE:

            prediction_history.pop(0)


        # ---------------------------------
        # Majority vote
        # ---------------------------------

        prediction_counts = np.bincount(

            prediction_history,

            minlength=len(
                label_encoder.classes_
            )
        )


        stable_class_id = np.argmax(
            prediction_counts
        )


        stable_label = (
            label_encoder.inverse_transform(
                [stable_class_id]
            )[0]
        )


        # =================================
        # STABLE LETTER DETECTION
        # =================================

        if stable_label == current_stable_label:

            stable_frame_count += 1

        else:

            current_stable_label = stable_label

            stable_frame_count = 1


        # =================================
        # ADD LETTER
        # =================================

        if (
            stable_frame_count >= STABLE_FRAMES_REQUIRED
            and
            stable_label != last_added_label
            and
            stable_label != "SPACE"
        ):

            stitched_text += stable_label

            last_added_label = stable_label

            print(
                f"Added: {stable_label}"
            )

        # =================================
        # DRAW LANDMARKS
        # =================================

        for landmark in hand:

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

                5,

                (0, 255, 0),

                -1
            )


        # ---------------------------------
        # Draw connections
        # ---------------------------------

        for start, end in HAND_CONNECTIONS:

            start_point = hand[start]

            end_point = hand[end]


            start_x = int(
                start_point.x *
                frame.shape[1]
            )

            start_y = int(
                start_point.y *
                frame.shape[0]
            )


            end_x = int(
                end_point.x *
                frame.shape[1]
            )

            end_y = int(
                end_point.y *
                frame.shape[0]
            )


            cv2.line(

                frame,

                (start_x, start_y),

                (end_x, end_y),

                (0, 255, 0),

                2
            )


        # =================================
        # DISPLAY CURRENT PREDICTION
        # =================================

        cv2.putText(

            frame,

            f"Prediction: {stable_label}",

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 255, 0),

            2
        )


        # =================================
        # DISPLAY STABILITY
        # =================================

        cv2.putText(

            frame,

            f"Stable: {stable_frame_count}/"
            f"{STABLE_FRAMES_REQUIRED}",

            (30, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


    else:

        # =================================
        # NO HAND
        # =================================

        prediction_history.clear()

        current_stable_label = None

        stable_frame_count = 0

        last_added_label = None


        cv2.putText(

            frame,

            "No hand detected",

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 255),

            2
        )


    # =====================================
    # DISPLAY STITCHED TEXT
    # =====================================

    cv2.putText(

        frame,

        f"Text: {stitched_text}",

        (30, 125),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (255, 255, 255),

        2
    )


    # =====================================
    # DISPLAY FRAME
    # =====================================

    cv2.imshow(

        "Sign Language Stitching V1",

        frame
    )


    # =====================================
    # KEYBOARD CONTROLS
    # =====================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    if key == ord("c"):

        stitched_text = ""

        print("\nStitched text cleared.")


# =========================================
# CLEANUP
# =========================================

detector.close()

camera.release()

cv2.destroyAllWindows()

print("\nProgram closed successfully.")