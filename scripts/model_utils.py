import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, brier_score_loss

def optimize_threshold(y_true, y_proba, metric='f1'):
    thresholds = np.linspace(0.1, 0.9, 1000)
    scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds)
        elif metric == 'precision':
            score = precision_score(y_true, preds)
        elif metric == 'recall':
            score = recall_score(y_true, preds)
        elif metric == 'brier':
            score = brier_score_loss(y_true, preds)
        elif metric == 'roc':
            score = roc_auc_score(y_true, preds)
        else:
            raise ValueError("Métrica no soportada.")
        scores.append(score)

    best_index = np.argmax(scores)
    best_threshold = thresholds[best_index]
    best_score = scores[best_index]

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, scores, label=f'{metric}')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.2f}')
    plt.title(f'Threshold Optimization ({metric})')
    plt.xlabel('Threshold')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold, best_score



def rolling_test(
    pipeline,
    X,
    y,
    test_window=50,
    step=25,
    min_train_size=250,
    verbose=True
):
    accuracies, f1_scores, roc_auc_scores = [], [], []
    y_true_all, y_proba_all = [], []

    for start in range(min_train_size, len(X) - test_window + 1, step):
        X_train, y_train_split = X.iloc[:start], y.iloc[:start]
        X_test, y_test_split = X.iloc[start:start + test_window], y.iloc[start:start + test_window]

        if len(np.unique(y_train_split)) < 2:
            continue

        pipeline.fit(X_train, y_train_split)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracies.append(accuracy_score(y_test_split, y_pred))
        f1_scores.append(f1_score(y_test_split, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test_split, y_proba))

        y_true_all.extend(y_test_split)
        y_proba_all.extend(y_proba)

    if verbose:
        print(">>> Rolling Test (threshold 0.5):")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
        print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")
        print(f"Mean ROC AUC Score: {np.mean(roc_auc_scores):.4f}")

    return {
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "roc_auc_scores": roc_auc_scores,
        "y_true": np.array(y_true_all),
        "y_proba": np.array(y_proba_all)
    }