import time

from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 KNN Baseline Training")
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


# Create KNN pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5
    ))
])


# Train model
print("\nTraining KNN...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training complete.")


# Make predictions
predictions = model.predict(X_test)


# Overall accuracy
accuracy = accuracy_score(y_test, predictions)


print("\n----------------------------------")
print("V2 KNN Baseline Results")
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


# Left-hand accuracy
left_mask = hand_test.to_numpy() == "left"

left_accuracy = accuracy_score(
    y_test[left_mask],
    predictions[left_mask]
)


# Right-hand accuracy
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