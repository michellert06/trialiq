import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATABASE_URL = "postgresql://localhost/trialiq"

def load_data():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM trials", engine)
    return df

def prepare_features(df):
    valid_statuses = ["COMPLETED", "TERMINATED", "WITHDRAWN"]
    df = df[df["status"].isin(valid_statuses)].copy()
    df["failed"] = df["status"].isin(["TERMINATED", "WITHDRAWN"]).astype(int)

    # Duration feature
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["duration_days"] = (df["completion_date"] - df["start_date"]).dt.days

    # Fill missing categoricals
    for col in ["phase", "study_type", "lead_sponsor", "sponsor_class", 
                "fda_regulated_drug", "sex", "accepts_healthy"]:
        df[col] = df[col].fillna("UNKNOWN")

    # Fill missing numerics with median
    for col in ["enrollment_count", "num_sites", "num_collaborators", "duration_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    features = [
        "phase", "study_type", "sponsor_class",
        "fda_regulated_drug", "sex", "accepts_healthy",
        "enrollment_count", "num_sites", "num_collaborators", "duration_days"
    ]

    X = df[features].copy()
    y = df["failed"]
    return X, y


def build_pipeline(model):
    categorical = ["phase", "study_type", "sponsor_class", 
                   "fda_regulated_drug", "sex", "accepts_healthy"]
    numerical = ["enrollment_count", "num_sites", "num_collaborators", "duration_days"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numerical)
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


def evaluate(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["completed", "failed"]))
    print(f"ROC AUC: {roc_auc_score(y_test, probs):.3f}")

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    print(f"Total trials used: {len(X)}")
    print(f"Failure rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_reg = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    log_reg.fit(X_train, y_train)
    evaluate("Logistic Regression", log_reg, X_test, y_test)

    rf = build_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)

import joblib
import os

# Save the best model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/trial_failure_model.pkl")
print("\nModel saved to models/trial_failure_model.pkl")
