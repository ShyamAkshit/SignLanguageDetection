import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 TUNED SVM EVALUATION")
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
# 2. CREATE TUNED SVM
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=300,
        gamma="scale"
    ))
])


print("\nModel configuration:")
print("Kernel: RBF")
print("C: 300")
print("Gamma: scale")


# ============================================================
# 3. TRAIN
# ============================================================

print("\n----------------------------------")
print("TRAINING TUNED SVM")
print("----------------------------------")

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("Training complete.")
print(f"Training time: {training_time:.4f} seconds")


# ============================================================
# 4. PREDICT
# ============================================================

print("\n----------------------------------")
print("TESTING TUNED SVM")
print("----------------------------------")

start_time = time.time()

predictions = model.predict(X_test)

prediction_time = time.time() - start_time

print(f"Prediction time: {prediction_time:.4f} seconds")


# ============================================================
# 5. OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

correct = (y_test == predictions).sum()
incorrect = (y_test != predictions).sum()


print("\n----------------------------------")
print("OVERALL RESULTS")
print("----------------------------------")

print(f"Testing accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")


# ============================================================
# 6. HAND-SPECIFIC ACCURACY
# ============================================================

left_mask = hand_test.to_numpy() == "left"
right_mask = hand_test.to_numpy() == "right"


left_accuracy = accuracy_score(
    y_test[left_mask],
    predictions[left_mask]
)

right_accuracy = accuracy_score(
    y_test[right_mask],
    predictions[right_mask]
)


print("\n----------------------------------")
print("HAND-SPECIFIC RESULTS")
print("----------------------------------")

print(
    f"Left-hand accuracy:  "
    f"{left_accuracy * 100:.2f}%"
)

print(
    f"Right-hand accuracy: "
    f"{right_accuracy * 100:.2f}%"
)

print(
    f"Left-hand errors: "
    f"{(y_test[left_mask] != predictions[left_mask]).sum()}"
)

print(
    f"Right-hand errors: "
    f"{(y_test[right_mask] != predictions[right_mask]).sum()}"
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n----------------------------------")
print("CLASSIFICATION REPORT")
print("----------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 8. PREDICTION SPEED
# ============================================================

predictions_per_second = (
    len(X_test) / prediction_time
)


print("----------------------------------")
print("PERFORMANCE")
print("----------------------------------")

print(
    f"Training time: "
    f"{training_time:.4f} seconds"
)

print(
    f"Prediction time: "
    f"{prediction_time:.4f} seconds"
)

print(
    f"Predictions/second: "
    f"{predictions_per_second:.0f}"
)


print("\n==============================================")
print("TUNED SVM EVALUATION COMPLETE")
print("==============================================")