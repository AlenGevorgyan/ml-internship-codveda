"""
train_rf.py
-----------
Task 1: Advanced Random Forest Classifier

Grid-searches n_estimators / max_depth, cross-validates, reports
precision/recall/f1, and dumps a feature importance chart + confusion
matrix into outputs/.

Run from inside random_forest_classifier/src/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from data_loader import get_processed_data

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# grid to search over -- tried a wider grid first but it took forever
# on my laptop so trimmed it down to the values that were actually
# moving the CV score
PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
}

RANDOM_STATE = 42


def tune_model(X_train, y_train):
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\nbest params found:", grid_search.best_params_)
    print("best CV f1 (weighted):", round(grid_search.best_score_, 4))

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- test set performance ---")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}")
    print(f"f1 score : {f1:.4f}")
    print("\nfull classification report:")
    print(classification_report(y_test, y_pred))

    # cross-val score on the full training set too, just as a sanity
    # check against overfitting to the grid search's own folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print("5-fold CV f1 (refit model):", np.round(cv_scores, 4))
    print("mean:", round(cv_scores.mean(), 4), "std:", round(cv_scores.std(), 4))

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("saved confusion matrix ->", cm_path)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print("saved feature importance chart ->", fi_path)

    return imp_df


def main():
    data = get_processed_data()

    best_model, best_params = tune_model(data["X_train"], data["y_train"])

    metrics = evaluate_model(
        best_model,
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
    )

    imp_df = plot_feature_importance(best_model, data["feature_names"])
    print("\ntop features:\n", imp_df.head(10).to_string(index=False))

    return best_model, metrics, imp_df


if __name__ == "__main__":
    main()
