import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("SAVE FINAL V2 MODEL")
print("----------------------------------")


# ============================================================
# 1. PREPARE DATASET
# ============================================================

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


print("\nDataset prepared.")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Number of classes:", len(label_encoder.classes_))


# ============================================================
# 2. CREATE FINAL V2 SVM
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=300,
        gamma="scale"
    ))
])


print("\nFinal V2 model configuration:")
print("Model: SVM")
print("Kernel: RBF")
print("C: 300")
print("Gamma: scale")


# ============================================================
# 3. TRAIN FINAL MODEL
# ============================================================

print("\n----------------------------------")
print("TRAINING FINAL V2 MODEL")
print("----------------------------------")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 4. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 5. SAVE MODEL
# ============================================================

model_path = "models/sign_language_svm_v2.joblib"
encoder_path = "models/label_encoder_v2.joblib"


joblib.dump(
    model,
    model_path
)

joblib.dump(
    label_encoder,
    encoder_path
)


# ============================================================
# 6. DISPLAY FILE INFORMATION
# ============================================================

print("\n----------------------------------")
print("FILES SAVED")
print("----------------------------------")

print("Model:", model_path)
print("Label encoder:", encoder_path)

print("\nModel file exists:", os.path.exists(model_path))
print("Encoder file exists:", os.path.exists(encoder_path))


# ============================================================
# 7. DISPLAY CLASS INFORMATION
# ============================================================

print("\n----------------------------------")
print("MODEL INFORMATION")
print("----------------------------------")

print("Number of classes:", len(label_encoder.classes_))

print("Classes:")

print(
    list(label_encoder.classes_)
)


print("\n==============================================")
print("FINAL V2 MODEL SAVED SUCCESSFULLY")
print("==============================================")