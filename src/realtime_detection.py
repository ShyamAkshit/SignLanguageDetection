import cv2
import mediapipe as mp
import joblib
import numpy as np

from feature_extraction import extract_landmarks
from realtime_features import prepare_realtime_features


# =========================================
# 1. FILE PATHS
# =========================================

MODEL_PATH = "models/hand_landmarker.task"

SVM_PATH = "models/sign_language_svm.joblib"

LABEL_ENCODER_PATH = "models/label_encoder.joblib"


# =========================================
# 2. LOAD SAVED MACHINE LEARNING MODEL
# =========================================

print("----------------------------------")
print("Real-Time Sign Language Detection")
print("----------------------------------")

print("\nLoading SVM model...")

model = joblib.load(SVM_PATH)

print("SVM model loaded successfully.")


print("\nLoading label encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print("Label encoder loaded successfully.")


# =========================================
# 3. OPEN WEBCAM
# =========================================

camera = cv2.VideoCapture(
    1,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("\nERROR: Could not open the camera.")

    exit()


print("\nCamera opened successfully.")
print("Press Q to close the program.")


# =========================================
# 4. MEDIAPIPE HAND LANDMARKER SETUP
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
# 5. HAND CONNECTIONS
# =========================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


# =========================================
# 6. FRAME TIMESTAMP
# =========================================

frame_timestamp = 0


# =========================================
# 7. MAIN CAMERA LOOP
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
    # Increase timestamp
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
    # 8. PROCESS DETECTED HAND
    # =====================================

    if results.hand_landmarks:

        for hand in results.hand_landmarks:


            # ---------------------------------
            # Extract 63 normalized features
            # ---------------------------------

            features = extract_landmarks(hand)


            # ---------------------------------
            # Convert 63 → 60 features
            # ---------------------------------

            processed_features = (
                prepare_realtime_features(
                    features
                )
            )


            # ---------------------------------
            # Convert to NumPy array
            # ---------------------------------

            input_data = np.array(
                processed_features,
                dtype=float
            ).reshape(1, -1)


            # ---------------------------------
            # Predict sign
            # ---------------------------------

            prediction = model.predict(
                input_data
            )


            # ---------------------------------
            # Convert class ID → sign
            # ---------------------------------

            predicted_class_id = prediction[0]

            predicted_label = (
                label_encoder.inverse_transform(
                    [predicted_class_id]
                )[0]
            )


            # =================================
            # 9. DRAW LANDMARKS
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
            # 10. DISPLAY PREDICTION
            # =================================

            cv2.putText(

                frame,

                f"Prediction: {predicted_label}",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.2,

                (0, 255, 0),

                3
            )


    else:

        # =================================
        # NO HAND DETECTED
        # =================================

        cv2.putText(

            frame,

            "No hand detected",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 255),

            2
        )


    # =====================================
    # 11. DISPLAY FRAME
    # =====================================

    cv2.imshow(

        "Sign Language Detection",

        frame
    )


    # -------------------------------------
    # Press Q to exit
    # -------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================
# 12. CLEANUP
# =========================================

detector.close()

camera.release()

cv2.destroyAllWindows()

print("\nProgram closed successfully.")