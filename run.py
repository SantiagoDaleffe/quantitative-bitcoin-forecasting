# run.py

import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

X_real_test = pd.read_csv("data/X_real_test.csv", parse_dates=["date"], index_col="date")
y_real_test = pd.read_csv("data/y_real_test.csv", parse_dates=["date"], index_col="date")

with open("models/feature_list.json", "r") as f:
    feature_list = json.load(f)

X_real_test = X_real_test[feature_list]

pipeline = joblib.load("models/final_mlp_pipeline.joblib")

best_thresh = float(open("models/best_threshold.txt").read().strip())

y_real_proba = pipeline.predict_proba(X_real_test)[:, 1]
y_real_pred = (y_real_proba >= best_thresh).astype(int)

accuracy = accuracy_score(y_real_test, y_real_pred)
f1 = f1_score(y_real_test, y_real_pred)
roc_auc = roc_auc_score(y_real_test, y_real_proba)
brier = brier_score_loss(y_real_test, y_real_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print(f"Brier Score: {brier:.4f}")


plt.figure(figsize=(8, 4))
sns.histplot(y_real_proba[y_real_test.values.ravel() == 0], label="Class 0", kde=True, color="steelblue", stat="density")
sns.histplot(y_real_proba[y_real_test.values.ravel() == 1], label="Class 1", kde=True, color="salmon", stat="density")
plt.axvline(best_thresh, color="black", linestyle="--", label=f"Threshold = {best_thresh}")
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()