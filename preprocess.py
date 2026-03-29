import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load the data
df = pd.read_csv("transactions.csv")
print("Original data shape:", df.shape)

# Step 2: Drop columns the model doesn't need
df = df.drop(columns=["transaction_id"])

# Step 3: Encode text columns into numbers
le = LabelEncoder()
df["merchant_category"] = le.fit_transform(df["merchant_category"])

# Step 4: Separate features (X) and label (y)
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

print("Fraud cases before SMOTE:", y.sum())

# Step 5: Apply SMOTE to fix imbalance
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Fraud cases after SMOTE:", y_resampled.sum())
print("Total rows after SMOTE:", len(X_resampled))

# Step 6: Scale the numbers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Step 7: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 8: Save everything for the next step
joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Preprocessing done! All files saved.")
