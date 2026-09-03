import joblib
import numpy as np

from realtime_features import prepare_realtime_features


# =========================================
# FILE PATHS
# =========================================

MODEL_PATH = "models/sign_language_svm.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"


# =========================================
# LOAD SAVED MODEL
# =========================================

print("----------------------------------")
print("Real-Time Inference Test")
print("----------------------------------")

print("\nLoading saved model...")

model = joblib.load(MODEL_PATH)

print("Saved SVM loaded successfully.")


# =========================================
# LOAD LABEL ENCODER
# =========================================

print("\nLoading label encoder...")

label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Label encoder loaded successfully.")


# =========================================
# CREATE TEST FEATURES
# =========================================

# Temporary 63-feature vector
# This is only for testing the pipeline.

test_features = list(range(63))


print("\nOriginal feature count:")
print(len(test_features))


# =========================================
# PREPARE FEATURES
# =========================================

processed_features = prepare_realtime_features(
    test_features
)


print("\nProcessed feature count:")
print(len(processed_features))


# =========================================
# CONVERT TO NUMPY ARRAY
# =========================================

input_data = np.array(
    processed_features,
    dtype=float
).reshape(1, -1)


print("\nModel input shape:")
print(input_data.shape)


# =========================================
# PREDICT
# =========================================

prediction = model.predict(input_data)


# =========================================
# CONVERT CLASS ID TO LABEL
# =========================================

predicted_class_id = prediction[0]

predicted_label = label_encoder.inverse_transform(
    [predicted_class_id]
)[0]


# =========================================
# DISPLAY RESULT
# =========================================

print("\n----------------------------------")
print("Prediction Result")
print("----------------------------------")

print("Predicted class ID:")
print(predicted_class_id)

print("\nPredicted sign:")
print(predicted_label)

print("----------------------------------")
print("Real-Time Inference Test Complete")
print("----------------------------------")