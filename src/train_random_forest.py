import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("Random Forest Training")
print("----------------------------------")


# Load preprocessed dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
print("\nTraining Random Forest...")

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()


training_time = end_time - start_time


print("Training complete.")


# Make predictions on training data
train_predictions = model.predict(X_train)

# Make predictions on testing data
test_predictions = model.predict(X_test)


# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)


print("\n----------------------------------")
print("Random Forest Results")
print("----------------------------------")

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")
print(f"Training Time:     {training_time:.2f} seconds")

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
print("Confusion Matrix")
print("----------------------------------")

print(cm)

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

# Plot confusion matrix
cm = confusion_matrix(y_test, test_predictions)

plt.figure(figsize=(16, 14))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_,
    rotation=90
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.colorbar()

plt.tight_layout()

plt.show()

print("----------------------------------")
