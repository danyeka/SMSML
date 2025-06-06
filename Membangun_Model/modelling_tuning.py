import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (cocok untuk ML pipeline)
import numpy as np
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')
import dagshub
import os

dagshub.init(repo_owner='danyeka', repo_name='credit-approval', mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# To:
mlflow.set_tracking_uri("file:./mlruns")

# Create a new MLflow Experiment
mlflow.set_experiment("Credit Approval Hyperparameter Tuning")

# Load data
print("Loading data...")
try:
    train_data = pd.read_csv('processed/cleaned_training.csv')
    test_data = pd.read_csv('processed/cleaned_testing.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data files exist in the 'processed' directory")
    exit(1)

# Identify target column
target_candidates = ['SeriousDlqin2yrs']
target_column = None

for col in target_candidates:
    if col in train_data.columns:
        target_column = col
        break

if target_column is None:
    print("Available columns:", list(train_data.columns))
    target_column = train_data.columns[-1]
    print(f"Using '{target_column}' as target column")

print(f"Target column: {target_column}")

# Prepare data
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

# Check if test data has target column
if target_column in test_data.columns:
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
else:
    # Split training data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Create input example for MLflow
input_example = X_train.iloc[0:5]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Logistic Regression Random Search
print("\n Logistic Regression - Random Search Tuning...")

with mlflow.start_run(run_name="LR_RandomSearch_Tuning"):
    # Parameter distributions yang lebih fokus
    lr_param_dist = {
        'C': uniform(0.1, 10),  # Range yang lebih sempit
        'penalty': ['l2'],  # Hanya l2 penalty
        'solver': ['lbfgs'],  # Solver yang lebih cepat
        'max_iter': [500]  # Iterasi yang lebih sedikit
    }
    
    # Random search
    lr_random = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        lr_param_dist,
        n_iter=5,  # Kurangi jumlah iterasi
        cv=3,      # Kurangi fold CV
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    lr_random.fit(X_train_scaled, y_train)
    
    # Best model
    lr_best_model = lr_random.best_estimator_
    lr_best_accuracy = lr_best_model.score(X_test_scaled, y_test)
    
    # Generate predictions for classification report
    predictions = lr_best_model.predict(X_test_scaled)
    
    # Log results
    mlflow.log_params(lr_random.best_params_)
    mlflow.log_metric("best_cv_score", lr_random.best_score_)
    mlflow.log_metric("test_accuracy", lr_best_accuracy)
    
    # Log model with proper input example
    mlflow.sklearn.log_model(
        sk_model=lr_best_model,
        artifact_path="model",
        input_example=pd.DataFrame(X_train_scaled[:5], columns=X_train.columns)
    )
    
    print(f"Best LR Parameters: {lr_random.best_params_}")
    print(f"Best CV Score: {lr_random.best_score_:.4f}")
    print(f"Test Accuracy: {lr_best_accuracy:.4f}")

# Tuned models results
print("\n" + "="*60)
print("TUNED MODELS")
print("="*60)

tuned_results = {'Logistic Regression': lr_best_accuracy}

# Find best model
best_tuned_name = max(tuned_results, key=tuned_results.get)
best_tuned_accuracy = tuned_results[best_tuned_name]

print(f"\nBest Tuned Model: {best_tuned_name}")
print(f"Best Accuracy: {best_tuned_accuracy:.4f}")

# Detailed classification report
print(f"\nDetailed Classification Report ({best_tuned_name}):")
print("-" * 60)
print(classification_report(y_test, predictions))

print("\n" + "="*60)
print("HYPERPARAMETER TUNING COMPLETED!")
print("Check MLflow UI at http://127.0.0.1:5000/ for detailed results")
print("="*60)