import time

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("SVM Hyperparameter Tuning")
print("----------------------------------")


# Load preprocessed dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")


# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])


# Hyperparameter combinations
param_grid = {
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}


print("\nStarting Grid Search...")
print("Testing 12 parameter combinations")
print("Using 5-fold cross-validation")


# Create Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# Start timing
start_time = time.time()

grid_search.fit(X_train, y_train)

end_time = time.time()

search_time = end_time - start_time


print("\n----------------------------------")
print("Grid Search Complete")
print("----------------------------------")

print(f"Search Time: {search_time:.2f} seconds")


# Best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation Accuracy: "
      f"{grid_search.best_score_ * 100:.2f}%")


# Test best model
best_model = grid_search.best_estimator_

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)


print("\n----------------------------------")
print("Final Tuned SVM Results")
print("----------------------------------")

print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")


# Classification report
print("\n----------------------------------")
print("Classification Report")
print("----------------------------------")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_
    )
)

print("----------------------------------")