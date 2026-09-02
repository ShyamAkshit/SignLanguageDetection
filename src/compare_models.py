import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("Final Model Comparison")
print("----------------------------------")


# Load dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


# Define models
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

    "Tuned SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=100,
            gamma=0.01
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


results = []


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Training time
    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    end_train = time.perf_counter()

    training_time = end_train - start_train


    # Test accuracy
    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_predictions)


    # Prediction speed
    start_prediction = time.perf_counter()

    model.predict(X_test)

    end_prediction = time.perf_counter()

    prediction_time = end_prediction - start_prediction

    predictions_per_second = len(X_test) / prediction_time


    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "Training Time": training_time,
        "Prediction Time": prediction_time,
        "Predictions/sec": predictions_per_second
    })


# Print results
print("\n----------------------------------")
print("Final Comparison")
print("----------------------------------")

print(
    f"{'Model':<20}"
    f"{'Accuracy':>12}"
    f"{'Train(s)':>12}"
    f"{'Predict(s)':>14}"
    f"{'Pred/sec':>14}"
)

print("-" * 72)


for result in results:

    print(
        f"{result['Model']:<20}"
        f"{result['Accuracy']:>11.2f}%"
        f"{result['Training Time']:>12.4f}"
        f"{result['Prediction Time']:>14.4f}"
        f"{result['Predictions/sec']:>14.2f}"
    )


print("-" * 72)


# Find best model by accuracy
best_model = max(results, key=lambda x: x["Accuracy"])

print("\nBest Model by Test Accuracy:")
print(f"{best_model['Model']}")

print(f"Accuracy: {best_model['Accuracy']:.2f}%")

print("----------------------------------")