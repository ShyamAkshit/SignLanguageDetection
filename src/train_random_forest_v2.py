import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 Random Forest Training")
print("----------------------------------")


# Prepare V2 dataset
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


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
print("\nTraining Random Forest...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training complete.")


# Make predictions
predictions = model.predict(X_test)


# Calculate overall accuracy
accuracy = accuracy_score(y_test, predictions)


print("\n----------------------------------")
print("V2 Random Forest Results")
print("----------------------------------")

print(f"Training time: {training_time:.4f} seconds")
print(f"Testing accuracy: {accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


# Calculate left-hand accuracy
left_mask = hand_test.to_numpy() == "left"

left_accuracy = accuracy_score(
    y_test[left_mask],
    predictions[left_mask]
)


# Calculate right-hand accuracy
right_mask = hand_test.to_numpy() == "right"

right_accuracy = accuracy_score(
    y_test[right_mask],
    predictions[right_mask]
)


print("----------------------------------")
print("Hand-Specific Accuracy")
print("----------------------------------")

print(f"Left-hand accuracy:  {left_accuracy * 100:.2f}%")
print(f"Right-hand accuracy: {right_accuracy * 100:.2f}%")

print("----------------------------------")