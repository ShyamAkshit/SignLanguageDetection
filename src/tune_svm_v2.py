import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from preprocess_dataset_v2 import prepare_dataset_v2


print("----------------------------------")
print("V2 SVM OPTIMIZATION")
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
# 2. CREATE SVM PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf"
    ))
])


# ============================================================
# 3. DEFINE HYPERPARAMETER SEARCH
# ============================================================

param_grid = {
    "svm__C": [
        1,
        10,
        100,
        300
    ],

    "svm__gamma": [
        "scale",
        0.001,
        0.01,
        0.1
    ]
}


print("\nHyperparameter search space:")

for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# 5. RUN SEARCH
# ============================================================

print("\n----------------------------------")
print("STARTING GRID SEARCH")
print("----------------------------------")

start_time = time.time()

grid_search.fit(
    X_train,
    y_train
)

search_time = time.time() - start_time


print("\nGrid search complete.")
print(f"Search time: {search_time:.2f} seconds")


# ============================================================
# 6. BEST PARAMETERS
# ============================================================

print("\n----------------------------------")
print("BEST PARAMETERS")
print("----------------------------------")

print("Best parameters:")
print(grid_search.best_params_)

print(
    f"\nBest cross-validation accuracy: "
    f"{grid_search.best_score_ * 100:.2f}%"
)


# ============================================================
# 7. TOP 10 CONFIGURATIONS
# ============================================================

print("\n----------------------------------")
print("TOP 10 CONFIGURATIONS")
print("----------------------------------")


results = grid_search.cv_results_

sorted_indices = results["rank_test_score"].argsort()


for rank, index in enumerate(sorted_indices[:10], start=1):

    score = results["mean_test_score"][index]

    params = results["params"][index]

    print(f"\nRank {rank}")

    print(
        f"CV Accuracy: {score * 100:.2f}%"
    )

    print(
        f"Parameters: {params}"
    )


print("\n==============================================")
print("SVM OPTIMIZATION COMPLETE")
print("==============================================")