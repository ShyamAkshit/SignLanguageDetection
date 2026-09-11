import time
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess_dataset_v2 import prepare_dataset_v2


MODEL_PATH = "models/sign_language_svm_v2.joblib"
ENCODER_PATH = "models/label_encoder_v2.joblib"


print("=" * 60)
print("TRAINING V2 SVM")
print("=" * 60)


# --------------------------------------------------
# 1. PREPARE DATASET
# --------------------------------------------------

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


print("\n" + "-" * 60)
print("DATASET SUMMARY")
print("-" * 60)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Features         : {X_train.shape[1]}")
print(f"Classes          : {len(label_encoder.classes_)}")

print("\nClasses:")
print(list(label_encoder.classes_))


# --------------------------------------------------
# 2. CREATE SVM PIPELINE
# --------------------------------------------------

print("\n" + "-" * 60)
print("CREATING SVM")
print("-" * 60)

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=300,
            gamma="scale"
        )
    )
])

print("Scaler: StandardScaler")
print("Kernel: RBF")
print("C: 300")
print("Gamma: scale")


# --------------------------------------------------
# 3. TRAIN
# --------------------------------------------------

print("\n" + "-" * 60)
print("TRAINING MODEL")
print("-" * 60)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed.")
print(f"Training time: {training_time:.4f} seconds")


# --------------------------------------------------
# 4. TRAINING ACCURACY
# --------------------------------------------------

train_accuracy = model.score(X_train, y_train)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")


# --------------------------------------------------
# 5. TEST ACCURACY
# --------------------------------------------------

start_prediction = time.time()

test_accuracy = model.score(X_test, y_test)

prediction_time = time.time() - start_prediction

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Prediction time: {prediction_time:.4f} seconds")


# --------------------------------------------------
# 6. SAVE MODEL
# --------------------------------------------------

print("\n" + "-" * 60)
print("SAVING MODEL")
print("-" * 60)

joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"Model saved:")
print(MODEL_PATH)

print(f"\nLabel encoder saved:")
print(ENCODER_PATH)


# --------------------------------------------------
# 7. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("V2 SVM TRAINING COMPLETE")
print("=" * 60)

print(f"\nTraining accuracy : {train_accuracy * 100:.2f}%")
print(f"Test accuracy     : {test_accuracy * 100:.2f}%")
print(f"Training time     : {training_time:.4f} seconds")
print(f"Classes           : {len(label_encoder.classes_)}")
print(f"Features          : {X_train.shape[1]}")

print("\nModel:")
print("StandardScaler + RBF SVM")
print("C = 300")
print("gamma = scale")

print("\nSaved files:")
print(MODEL_PATH)
print(ENCODER_PATH)