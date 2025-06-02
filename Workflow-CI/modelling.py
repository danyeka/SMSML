import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Set MLflow tracking URI for Docker compatibility
    # Use environment variable if available, otherwise use local file system
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(tracking_uri)

    # Read the csv file
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_training.csv")
    data = pd.read_csv(file_path)

    # The predicted column is "SeriousDlqin2yrs"
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("SeriousDlqin2yrs", axis=1),
        data["SeriousDlqin2yrs"],
        random_state=42,
        test_size=0.2
    )
    
    # Scale the features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_example = X_train[0:5]
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    with mlflow.start_run():
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Log parameters
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Log metrics
        accuracy = model.score(X_test_scaled, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
