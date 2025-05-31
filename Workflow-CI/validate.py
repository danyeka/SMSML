import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def validate_model():
    """
    Validate the MLflow model by loading and testing it with sample data
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:../mlruns")
    
    # Model path - using direct path to artifacts
    logged_model = '../mlartifacts/250495177937610801/36b6d9f92bb440488cdf061e6c97fca5/artifacts/logistic_regression_model'
    
    try:
        # Load model as a PyFuncModel
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model loaded successfully from: {logged_model}")
        
        # Load sample data for prediction
        data_path = os.path.join("MLProject", "cleaned_training.csv")
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            print(f"Data loaded from: {data_path}")
            print(f"Data shape: {data.shape}")
            
            # Remove target column if it exists
            target_col = 'SeriousDlqin2yrs'
            if target_col in data.columns:
                X = data.drop(columns=[target_col])
                y = data[target_col]
                print(f"Target column '{target_col}' found and removed for prediction")
            else:
                X = data
                print("No target column found, using all columns for prediction")
            
            # Take a small sample for validation (first 5 rows)
            sample_data = X.head(5)
            print(f"\nSample data for prediction:")
            print(sample_data)
            
            # Predict on the sample data
            predictions = loaded_model.predict(sample_data)
            print(f"\nPredictions:")
            print(predictions)
            
            # If we have target values, show them for comparison
            if target_col in data.columns:
                actual_values = y.head(5)
                print(f"\nActual values (for comparison):")
                print(actual_values.values)
            
            print("\n✅ Model validation completed successfully!")
            return True
            
        else:
            print(f"❌ Data file not found at: {data_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error during model validation: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting MLflow model validation...")
    print("=" * 50)
    
    success = validate_model()
    
    print("=" * 50)
    if success:
        print("🎉 Validation completed successfully!")
    else:
        print("💥 Validation failed!")