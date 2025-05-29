import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Credit Approval Basic Models V1")

# Load data
print("Loading data...")
train_data = pd.read_csv('Membangun_Model/processed/cleaned_training.csv')
test_data = pd.read_csv('Membangun_Model/processed/cleaned_testing.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Identify target column for Credit Approval (binary classification)
target_candidates = ['SeriousDlqin2yrs', 'target', 'approval', 'approved', 'credit_approval', 'default']
target_column = None

for col in target_candidates:
    if col in train_data.columns:
        target_column = col
        break

if target_column is None:
    print("Available columns:", list(train_data.columns))
    target_column = train_data.columns[0]  # Try first column for credit approval
    print(f"Using '{target_column}' as target column")

print(f"Target column: {target_column}")
print(f"Target distribution (Credit Approval):")
target_counts = train_data[target_column].value_counts()
print(target_counts)

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

# Create input example for MLflow
input_example = X_train.iloc[0:5]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*60)

# Train Logistic Regression with MLflow tracking
with mlflow.start_run(run_name="Logistic Regression"):
    # Initialize and train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_precision = precision_score(y_test, lr_predictions)
    lr_recall = recall_score(y_test, lr_predictions)
    lr_f1 = f1_score(y_test, lr_predictions)
    lr_auc_roc = roc_auc_score(y_test, lr_proba)
    
    # Log parameters
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": lr_accuracy,
        "precision": lr_precision,
        "recall": lr_recall,
        "f1_score": lr_f1,
        "auc_roc": lr_auc_roc
    })
    
    # Log model
    mlflow.sklearn.log_model(
        lr_model,
        "logistic_regression_model",
        input_example=input_example
    )
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall: {lr_recall:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    print(f"AUC-ROC: {lr_auc_roc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, lr_predictions,
                              target_names=['Rejected (0)', 'Approved (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, lr_predictions)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (Correct Rejections): {cm[0,0]}")
    print(f"False Positives (Wrong Approvals): {cm[0,1]}")
    print(f"False Negatives (Wrong Rejections): {cm[1,0]}")
    print(f"True Positives (Correct Approvals): {cm[1,1]}")

    # Log confusion matrix as an artifact
print("\n" + "="*60)
print("CREDIT APPROVAL MODELING COMPLETED!")
print("Check MLflow UI at http://127.0.0.1:5000/ for detailed results")
print("="*60)