import joblib

from sklearn.metrics import accuracy_score

from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("Testing Saved Model")
print("----------------------------------")


# Load test data
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


# Load saved model
model = joblib.load("models/sign_language_svm.joblib")

# Load saved label encoder
saved_label_encoder = joblib.load("models/label_encoder.joblib")


print("Saved model loaded successfully.")
print("Saved label encoder loaded successfully.")


# Predict test data
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)


print("\n----------------------------------")
print("Saved Model Results")
print("----------------------------------")

print(f"Testing Accuracy: {accuracy * 100:.2f}%")

print(f"Number of classes: {len(saved_label_encoder.classes_)}")

print("\nClasses:")
print(saved_label_encoder.classes_)

print("----------------------------------")