import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("KNN Training")
print("----------------------------------")


# Load preprocessed dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")


# Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5
    ))
])


# Train KNN
print("\nTraining KNN...")

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()

training_time = end_time - start_time


print("Training complete.")


# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)


print("\n----------------------------------")
print("KNN Results")
print("----------------------------------")

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")
print(f"Training Time:     {training_time:.2f} seconds")


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