import mlflow
import pandas as pd
import os
import numpy as np
import warnings
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load data from command line or default path
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed/cleaned_training.csv")
    data = pd.read_csv(file_path)

    # Tentukan kolom target
    target_col = 'Credit_Score'  # Ganti sesuai kebutuhanmu, bisa juga 'SeriousDlqin2yrs' jika itu yang digunakan
    if target_col not in data.columns:
        print("Kolom target tidak ditemukan!")
        sys.exit(1)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling optional untuk Gradient Boosting
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_example = pd.DataFrame(X_train[:5])

    # Ambil parameter dari command-line atau pakai default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    with mlflow.start_run(run_name="GradientBoosting_CreditApproval"):
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Logging model & metric
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_metric("accuracy", accuracy)

        print(f"Model trained. Accuracy: {accuracy:.4f}")
