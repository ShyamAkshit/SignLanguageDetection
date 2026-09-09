import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 MODEL COMPARISON")
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
# 2. DEFINE MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM Baseline": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


# ============================================================
# 3. STORAGE FOR RESULTS
# ============================================================

results = []


# ============================================================
# 4. TRAIN AND TEST EACH MODEL
# ============================================================

for model_name, model in models.items():

    print("\n----------------------------------")
    print(f"Testing: {model_name}")
    print("----------------------------------")

    # -----------------------------
    # Training
    # -----------------------------

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"Training complete: {training_time:.4f} seconds")


    # -----------------------------
    # Prediction
    # -----------------------------

    start_time = time.time()

    predictions = model.predict(X_test)

    prediction_time = time.time() - start_time

    print(f"Prediction complete: {prediction_time:.4f} seconds")


    # -----------------------------
    # Overall accuracy
    # -----------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # -----------------------------
    # Left-hand accuracy
    # -----------------------------

    left_mask = hand_test.to_numpy() == "left"

    left_accuracy = accuracy_score(
        y_test[left_mask],
        predictions[left_mask]
    )


    # -----------------------------
    # Right-hand accuracy
    # -----------------------------

    right_mask = hand_test.to_numpy() == "right"

    right_accuracy = accuracy_score(
        y_test[right_mask],
        predictions[right_mask]
    )


    # -----------------------------
    # Prediction speed
    # -----------------------------

    predictions_per_second = (
        len(X_test) / prediction_time
    )


    # -----------------------------
    # Store results
    # -----------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Left Accuracy": left_accuracy * 100,
        "Right Accuracy": right_accuracy * 100,
        "Training Time": training_time,
        "Prediction Time": prediction_time,
        "Predictions/sec": predictions_per_second
    })


# ============================================================
# 5. DISPLAY COMPARISON
# ============================================================

print("\n\n==============================================")
print("V2 MODEL COMPARISON RESULTS")
print("==============================================")

print(
    f"{'Model':<18}"
    f"{'Accuracy':>12}"
    f"{'Left':>12}"
    f"{'Right':>12}"
    f"{'Train(s)':>12}"
    f"{'Predict(s)':>14}"
    f"{'Pred/sec':>14}"
)

print("-" * 94)


for result in results:

    print(
        f"{result['Model']:<18}"
        f"{result['Accuracy']:>11.2f}%"
        f"{result['Left Accuracy']:>11.2f}%"
        f"{result['Right Accuracy']:>11.2f}%"
        f"{result['Training Time']:>12.4f}"
        f"{result['Prediction Time']:>14.4f}"
        f"{result['Predictions/sec']:>14.0f}"
    )


# ============================================================
# 6. FIND BEST MODEL
# ============================================================

best_model = max(
    results,
    key=lambda x: x["Accuracy"]
)


print("\n==============================================")
print("BEST V2 MODEL")
print("==============================================")

print("Model:", best_model["Model"])
print(f"Accuracy: {best_model['Accuracy']:.2f}%")
print(f"Left-hand accuracy: {best_model['Left Accuracy']:.2f}%")
print(f"Right-hand accuracy: {best_model['Right Accuracy']:.2f}%")

print("==============================================")