# src/train.py

# 1️⃣ Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib

# 2️⃣ Load processed data
# Make sure you have run Task 3 and Task 4 to create features and proxy target
df = pd.read_csv("data/processed/processed_data.csv")

# Features and target
X = df.drop(columns=["is_high_risk", "CustomerId"])
y = df["is_high_risk"]

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Start MLflow experiment
mlflow.set_experiment("BNPL_Credit_Risk_Model")

with mlflow.start_run(run_name="LogReg_RandomForest"):

    # 5️⃣ Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {"C": [0.01, 0.1, 1, 10]}
    
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="roc_auc")
    lr_grid.fit(X_train, y_train)
    
    lr_best = lr_grid.best_estimator_
    
    # Log LR model
    mlflow.sklearn.log_model(lr_best, "logreg_model")
    mlflow.log_params(lr_grid.best_params_)
    
    # 6️⃣ Model 2: Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }
    
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="roc_auc")
    rf_grid.fit(X_train, y_train)
    
    rf_best = rf_grid.best_estimator_
    
    # Log RF model
    mlflow.sklearn.log_model(rf_best, "rf_model")
    mlflow.log_params(rf_grid.best_params_)

    # 7️⃣ Evaluate models
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        return metrics
    
    lr_metrics = evaluate_model(lr_best, X_test, y_test)
    rf_metrics = evaluate_model(rf_best, X_test, y_test)
    
    # Log metrics
    mlflow.log_metrics({f"lr_{k}": v for k, v in lr_metrics.items()})
    mlflow.log_metrics({f"rf_{k}": v for k, v in rf_metrics.items()})

    print("Logistic Regression Metrics:", lr_metrics)
    print("Random Forest Metrics:", rf_metrics)

    # 8️⃣ Save best model locally
    joblib.dump(rf_best, "models/rf_best_model.pkl")  # Example: save RF as best
