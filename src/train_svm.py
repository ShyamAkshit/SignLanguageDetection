import time

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("SVM Training")
print("----------------------------------")


# Load preprocessed dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")


# Create feature scaler
scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)


print("\nFeature scaling complete.")


# Create SVM model
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)


# Train SVM
print("\nTraining SVM...")

start_time = time.time()

model.fit(X_train_scaled, y_train)

end_time = time.time()

training_time = end_time - start_time


print("Training complete.")


# Predictions
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)


# Accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)


print("\n----------------------------------")
print("SVM Results")
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

# Create confusion matrix
cm = confusion_matrix(y_test, test_predictions)

print("\n----------------------------------")
print("Misclassified Samples")
print("----------------------------------")

misclassified = 0

for actual, predicted in zip(y_test, test_predictions):

    if actual != predicted:

        actual_label = label_encoder.inverse_transform([actual])[0]
        predicted_label = label_encoder.inverse_transform([predicted])[0]

        print(f"Actual: {actual_label} -> Predicted: {predicted_label}")

        misclassified += 1

print("----------------------------------")
print(f"Total Misclassified: {misclassified}")
print("----------------------------------")

