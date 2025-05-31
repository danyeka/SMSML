import mlflow
import pandas as pd
import os
import numpy as np
import warnings
import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Set MLflow tracking URI to use local mlruns directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set or create experiment
    experiment_name = "Credit Approval Basic Models V1"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)

    # Load data from command line or default path
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed/cleaned_training.csv")
    data = pd.read_csv(file_path)

    # Tentukan kolom target
    target_col = 'SeriousDlqin2yrs'  # Ganti sesuai kebutuhanmu, bisa juga 'SeriousDlqin2yrs' jika itu yang digunakan
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

    # Scaling penting untuk Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_example = pd.DataFrame(X_train_scaled[:5], columns=X_train.columns)

    # Ambil parameter dari command-line atau pakai default
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    with mlflow.start_run(run_name="LogisticRegression_CreditApproval"):
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        auc_roc = roc_auc_score(y_test, pred_proba, average='weighted')

        # Logging model & metrics
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)

        print(f"Model trained with the following metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
