import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Credit Approval Basic Models V1")

# Load data
print("Loading data...")
train_data = pd.read_csv('processed/cleaned_training.csv')
test_data = pd.read_csv('processed/cleaned_testing.csv')

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
    # Assume first column or last column as target
    target_column = train_data.columns[0]  # Try first column for credit approval
    print(f"Using '{target_column}' as target column")

print(f"Target column: {target_column}")
print(f"Target distribution (Credit Approval):")
target_counts = train_data[target_column].value_counts()
print(target_counts)
print(f"Class balance - 0 (Rejected): {target_counts.get(0, 0)}, 1 (Approved): {target_counts.get(1, 0)}")

# Check for class imbalance
total_samples = len(train_data)
approval_rate = target_counts.get(1, 0) / total_samples
print(f"Credit Approval Rate: {approval_rate:.2%}")

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
        stratify=y_train  # Maintain class balance in split
    )

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Training set approval rate: {y_train.mean():.2%}")
print(f"Test set approval rate: {y_test.mean():.2%}")

# Create input example for MLflow
input_example = X_train.iloc[0:5]

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store models and results
models_results = {}

print("\n" + "="*60)
print("TRAINING CREDIT APPROVAL MODELS")
print("="*60)

# 1. Logistic Regression

# 

# Compare all models
print("\n" + "="*60)
print("CREDIT APPROVAL MODELS COMPARISON")
print("="*60)

# Sort models by AUC-ROC (better metric for binary classification)
sorted_models_auc = sorted(models_results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)
sorted_models_f1 = sorted(models_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)

print("Ranking by AUC-ROC Score:")
print("-" * 40)
for i, (name, results) in enumerate(sorted_models_auc, 1):
    print(f"{i}. {name}: AUC={results['auc_roc']:.4f}, Acc={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")

print("\nRanking by F1-Score:")
print("-" * 40)
for i, (name, results) in enumerate(sorted_models_f1, 1):
    print(f"{i}. {name}: F1={results['f1_score']:.4f}, AUC={results['auc_roc']:.4f}, Acc={results['accuracy']:.4f}")

# Best model based on AUC-ROC
best_model_name, best_model_results = sorted_models_auc[0]
print(f"\nBest Model for Credit Approval: {best_model_name}")
print(f"AUC-ROC: {best_model_results['auc_roc']:.4f}")
print(f"Accuracy: {best_model_results['accuracy']:.4f}")
print(f"Precision: {best_model_results['precision']:.4f}")
print(f"Recall: {best_model_results['recall']:.4f}")
print(f"F1-Score: {best_model_results['f1_score']:.4f}")

# Feature importance for tree-based models
if 'Random Forest' in models_results or 'XGBoost' in models_results:
    print(f"\nTop 10 Feature Importance for Credit Approval:")
    print("-" * 50)
    
    if 'XGBoost' in models_results:
        xgb_model = models_results['XGBoost']['model']
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("XGBoost Feature Importance:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    elif 'Random Forest' in models_results:
        rf_model = models_results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Random Forest Feature Importance:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Detailed report for best model
print(f"\nDetailed Classification Report for {best_model_name}:")
print("-" * 60)
print(classification_report(y_test, best_model_results['predictions'], 
                          target_names=['Rejected (0)', 'Approved (1)']))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_model_results['predictions'])
print(f"\nConfusion Matrix for {best_model_name}:")
print("-" * 30)
print(f"True Negatives (Correct Rejections): {cm[0,0]}")
print(f"False Positives (Wrong Approvals): {cm[0,1]}")
print(f"False Negatives (Wrong Rejections): {cm[1,0]}")
print(f"True Positives (Correct Approvals): {cm[1,1]}")

# Business Impact Analysis
total_predictions = len(y_test)
approval_predictions = best_model_results['predictions'].sum()
actual_approvals = y_test.sum()

print(f"\nBusiness Impact Analysis:")
print("-" * 30)
print(f"Total Applications: {total_predictions}")
print(f"Model Approved: {approval_predictions} ({approval_predictions/total_predictions:.1%})")
print(f"Actually Approved: {actual_approvals} ({actual_approvals/total_predictions:.1%})")
print(f"False Positive Rate (Risk): {cm[0,1]/(cm[0,0]+cm[0,1]):.2%}")
print(f"False Negative Rate (Lost Opportunity): {cm[1,0]/(cm[1,0]+cm[1,1]):.2%}")

print("\n" + "="*60)
print("CREDIT APPROVAL MODELING COMPLETED!")
print("Check MLflow UI at http://127.0.0.1:5000/ for detailed results")
print("="*60)