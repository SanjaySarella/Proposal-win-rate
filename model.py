import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)

df = pd.read_csv("contracts_features.csv")

feature_cols = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded"
]

X = df[feature_cols]
y = df["won_definitive"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTarget distribution in test set:")
print(y_test.value_counts())

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=15,
    min_samples_leaf=6,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Delivery Order", "Definitive Contract"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=["Delivery Order", "Definitive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Contract Type Prediction")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# Feature importance
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importances["feature"], importances["importance"], color="#2E75B6")
plt.xlabel("Importance")
plt.title("Feature Importance — Contract Win Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")

# SHAP
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
else:
    shap_vals_class1 = shap_values[:, :, 1] if len(
        np.array(shap_values).shape) == 3 else shap_values

shap.summary_plot(
    shap_vals_class1,
    X_test,
    feature_names=feature_cols,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png")

# Save everything
joblib.dump(model, "contract_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
print("\nModel saved to contract_model.pkl")
print("Explainer saved to shap_explainer.pkl")

# Save test predictions for agent
test_data = X_test.copy()
test_data["won_definitive"] = y_test.values
test_data["win_probability"] = y_proba
test_data["Recipient Name"] = df.loc[X_test.index, "Recipient Name"].values
test_data["Awarding Agency"] = df.loc[X_test.index, "Awarding Agency"].values
test_data["Award Amount"] = df.loc[X_test.index, "Award Amount"].values
test_data["Contract Award Type"] = df.loc[X_test.index, "Contract Award Type"].values
test_data.to_csv("test_predictions.csv", index=False)
print("Saved test_predictions.csv")
