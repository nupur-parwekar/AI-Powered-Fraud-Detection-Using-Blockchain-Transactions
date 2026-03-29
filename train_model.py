import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 1: Load the preprocessed data
X_train = joblib.load("X_train.pkl")
X_test  = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print("Training data loaded successfully!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 2: Create the model
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,    # reproducible results
    class_weight="balanced"  # handles any remaining imbalance
)

# Step 3: Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Training done!")

# Step 4: Test the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # fraud probability (0.0 to 1.0)

# Step 5: Print results
print("\n--- Model Results ---")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 3))

# Step 6: Show top features
feature_names = ["amount", "hour", "merchant_category", "is_foreign", "customer_age"]
importances = model.feature_importances_
print("\n--- Feature Importance (what the AI focuses on) ---")
for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {round(score * 100, 1)}%")

# Step 7: Save the model
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl — ready for the API!")
