"""Evaluation harness for Intent Classification.

Compares Majority-Class baseline, Simple TF-IDF + Logistic Regression, and Tuned Model
against the 200-example Golden Evaluation Set (data/golden_set.csv).
Generates classification_metrics.json, classification_report.csv, and confusion_matrix.png.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    GOLDEN_SET_PATH,
    HISTORICAL_PAIRS_PATH,
    RESULTS_DIR,
    INTENT_CLASSES,
)
from src.preprocessing import clean_text


def run_classifier_evaluation():
    """Run full comparative evaluation across baselines on the golden set."""
    print("=" * 70)
    print("EVALUATING INTENT CLASSIFIERS ON GOLDEN EVALUATION SET")
    print("=" * 70)

    # 1. Load Datasets
    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    df_train = pd.read_csv(HISTORICAL_PAIRS_PATH)
    
    df_train = df_train[df_train["customer_text"].notna() & df_train["intent"].notna()]
    X_train = df_train["customer_text"].astype(str)
    y_train = df_train["intent"].astype(str)

    X_gold = df_golden["customer_text"].astype(str)
    y_gold = df_golden["gold_intent"].astype(str)

    print(f"Loaded {len(X_train)} training rows, {len(X_gold)} golden evaluation rows.")

    # 2. Baseline 1: Majority Class Dummy Classifier
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train, y_train)
    y_pred_maj = majority_clf.predict(X_gold)

    # 3. Baseline 2: Simple TF-IDF (1-gram) + Logistic Regression
    vec_simple = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
    X_train_simple = vec_simple.fit_transform(X_train)
    X_gold_simple = vec_simple.transform(X_gold)
    
    simple_logreg = LogisticRegression(max_iter=500, random_state=42)
    simple_logreg.fit(X_train_simple, y_train)
    y_pred_simple = simple_logreg.predict(X_gold_simple)

    # 4. Model 3 / Final: Tuned TF-IDF (1,2-grams, sublinear tf) + Balanced Logistic Regression
    vec_tuned = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=15000,
    )
    X_train_tuned = vec_tuned.fit_transform(X_train)
    X_gold_tuned = vec_tuned.transform(X_gold)

    tuned_logreg = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    tuned_logreg.fit(X_train_tuned, y_train)
    y_pred_tuned = tuned_logreg.predict(X_gold_tuned)

    # 5. Compute Metrics for all models
    def calc_metrics(y_true, y_pred):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        }

    metrics = {
        "golden_set_size": len(X_gold),
        "baseline_1_majority": calc_metrics(y_gold, y_pred_maj),
        "baseline_2_simple_tfidf_logreg": calc_metrics(y_gold, y_pred_simple),
        "final_tuned_tfidf_logreg": calc_metrics(y_gold, y_pred_tuned),
    }

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "classification_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    # Print comparative table
    print("\n" + "=" * 75)
    print(f"{'Model':<35} | {'Acc':<7} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Macro Prec':<10}")
    print("-" * 75)
    for model_name, m in [
        ("Baseline 1: Majority Class", metrics["baseline_1_majority"]),
        ("Baseline 2: Simple TF-IDF + LogReg", metrics["baseline_2_simple_tfidf_logreg"]),
        ("Final: Tuned TF-IDF + LogReg", metrics["final_tuned_tfidf_logreg"]),
    ]:
        print(f"{model_name:<35} | {m['accuracy']:<7.4f} | {m['macro_f1']:<9.4f} | {m['weighted_f1']:<11.4f} | {m['macro_precision']:<10.4f}")
    print("=" * 75)

    # 6. Detailed Classification Report for Final Model
    report_dict = classification_report(y_gold, y_pred_tuned, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    report_path = RESULTS_DIR / "classification_report.csv"
    df_report.to_csv(report_path)
    print(f"\nSaved detailed classification report -> {report_path}")

    # 7. Generate and Save Confusion Matrix
    classes = sorted(list(set(y_gold) | set(y_pred_tuned)))
    cm = confusion_matrix(y_gold, y_pred_tuned, labels=classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Final Intent Classifier (Golden Set)")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Gold Intent")
    plt.xlabel("Predicted Intent")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot -> {cm_path}")

    return metrics


if __name__ == "__main__":
    run_classifier_evaluation()
