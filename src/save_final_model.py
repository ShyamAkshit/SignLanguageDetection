import os
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocess_dataset import prepare_dataset


print("----------------------------------")
print("Saving Final SVM Model")
print("----------------------------------")


# Load preprocessed dataset
X_train, X_test, y_train, y_test, label_encoder, constant_features = prepare_dataset()


# Create final tuned SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=100,
        gamma=0.01
    ))
])


print("\nTraining final tuned SVM...")

# Train final model on the complete training set
model.fit(X_train, y_train)

print("Training complete.")


# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)


# Save trained model
model_path = "models/sign_language_svm.joblib"

joblib.dump(model, model_path)


# Save label encoder
label_encoder_path = "models/label_encoder.joblib"

joblib.dump(label_encoder, label_encoder_path)


print("\n----------------------------------")
print("Model Saving Complete")
print("----------------------------------")

print(f"SVM model saved to:")
print(model_path)

print("\nLabel encoder saved to:")
print(label_encoder_path)

print("----------------------------------")