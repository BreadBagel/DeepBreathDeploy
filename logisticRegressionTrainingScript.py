import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report



CSV_PATH = r"C:\Users\User\Downloads\synthetic_pneumonia_dataset_8000_transformed.csv"
df = pd.read_csv(CSV_PATH)

#Split into X,Y
FEATURE_COLS = [
    'cough',
    'fever',
    'chest_retractions',
    'nasal_flaring',
    'lethargy',
    'grunting',
    'cyanosis',
    'dry_cough',
    'wheezing',
    'nocturnal_cough',
    'productive_cough',
    'rr',
]
X = df[FEATURE_COLS].astype(float)   # ensure float dtype
y = df['pneumonia'].astype(int)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#Fit LogisticRegression
model = LogisticRegression(
    solver='liblinear',        # small dataset, binary
    class_weight='balanced',   # because ~60/40 split
    max_iter=1000,
)
model.fit(X_train, y_train)

#Evaluate on held‐out test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Test Set Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#5-fold CV AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"5-Fold ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#Save the fusion model
OUT_PATH = 'FINALSTABLELogisticsReg1.pkl'
joblib.dump(model, OUT_PATH)
print(f"\nSaved trained fusion model to {OUT_PATH}")
