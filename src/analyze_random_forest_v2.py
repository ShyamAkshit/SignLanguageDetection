import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 RANDOM FOREST ERROR ANALYSIS")
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
# 2. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 3. PREDICT TEST DATA
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n----------------------------------")
print("OVERALL RESULTS")
print("----------------------------------")

print(f"Testing accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {(y_test == predictions).sum()}")
print(f"Incorrect predictions: {(y_test != predictions).sum()}")


# ============================================================
# 4. CLASSIFICATION REPORT
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
# 5. FIND ALL MISCLASSIFICATIONS
# ============================================================

print("\n----------------------------------")
print("MISCLASSIFICATIONS")
print("----------------------------------")


# Convert numeric class IDs back to actual labels

actual_labels = label_encoder.inverse_transform(y_test)
predicted_labels = label_encoder.inverse_transform(predictions)


errors = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predicted_labels,
    "Hand": hand_test.to_numpy()
})


errors = errors[
    errors["Actual"] != errors["Predicted"]
]


print(f"\nTotal errors: {len(errors)}")

print("\nIndividual errors:")

for index, row in errors.iterrows():

    print(
        f"Actual: {row['Actual']:>2}  "
        f"Predicted: {row['Predicted']:>2}  "
        f"Hand: {row['Hand']}"
    )


# ============================================================
# 6. ERROR COUNT BY ACTUAL CLASS
# ============================================================

print("\n----------------------------------")
print("ERRORS BY ACTUAL CLASS")
print("----------------------------------")


errors_by_class = (
    errors["Actual"]
    .value_counts()
    .sort_index()
)


for label, count in errors_by_class.items():

    print(
        f"{label:>2} : {count} error(s)"
    )


# ============================================================
# 7. ERROR COUNT BY HAND
# ============================================================

print("\n----------------------------------")
print("ERRORS BY HAND")
print("----------------------------------")


errors_by_hand = (
    errors["Hand"]
    .value_counts()
)


for hand, count in errors_by_hand.items():

    print(
        f"{hand.capitalize():>5} : {count} error(s)"
    )


# ============================================================
# 8. MOST COMMON CONFUSIONS
# ============================================================

print("\n----------------------------------")
print("MOST COMMON CONFUSIONS")
print("----------------------------------")


confusion_pairs = (
    errors
    .groupby(["Actual", "Predicted"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)


for _, row in confusion_pairs.iterrows():

    print(
        f"{row['Actual']:>2} -> "
        f"{row['Predicted']:>2} : "
        f"{row['Count']} time(s)"
    )


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\n----------------------------------")
print("CONFUSION MATRIX")
print("----------------------------------")

labels = list(range(len(label_encoder.classes_)))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm_df)


# ============================================================
# 10. HAND-SPECIFIC ACCURACY
# ============================================================

print("\n----------------------------------")
print("HAND-SPECIFIC RESULTS")
print("----------------------------------")


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


print(
    f"Left-hand accuracy:  "
    f"{left_accuracy * 100:.2f}%"
)

print(
    f"Right-hand accuracy: "
    f"{right_accuracy * 100:.2f}%"
)


# ============================================================
# 11. HAND-SPECIFIC ERRORS
# ============================================================

left_errors = (
    (y_test[left_mask] != predictions[left_mask])
    .sum()
)

right_errors = (
    (y_test[right_mask] != predictions[right_mask])
    .sum()
)


print("\nLeft-hand errors:", left_errors)
print("Right-hand errors:", right_errors)


print("\n==============================================")
print("ERROR ANALYSIS COMPLETE")
print("==============================================")