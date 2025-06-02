import os
import time
import logging
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import mlflow.sklearn
from datetime import datetime
import threading
import psutil
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML inference requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'ML inference request latency')
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made', ['prediction_class'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
MODEL_PRECISION = Gauge('ml_model_precision', 'Current model precision')
MODEL_RECALL = Gauge('ml_model_recall', 'Current model recall')
MODEL_F1_SCORE = Gauge('ml_model_f1_score', 'Current model F1 score')
MODEL_AUC_ROC = Gauge('ml_model_auc_roc', 'Current model AUC-ROC')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')
MODEL_LOAD_TIME = Gauge('ml_model_load_time_seconds', 'Time taken to load the model')
ERROR_COUNT = Counter('ml_errors_total', 'Total ML inference errors', ['error_type'])
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Number of active connections')
THROUGHPUT = Gauge('ml_throughput_requests_per_second', 'Requests per second throughput')

# Global variables
model = None
model_version = None
model_loaded_at = None
request_times = []
active_requests = 0

def load_model():
    """Load the trained model from MLflow"""
    global model, model_version, model_loaded_at
    
    start_time = time.time()
    try:
        # Try to load from MLflow first
        try:
            model_uri = "models:/credit_scoring_model/latest"
            model = mlflow.sklearn.load_model(model_uri)
            model_version = "mlflow_latest"
            logger.info(f"Model loaded from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")
            # Fallback to local model file
            model_path = "../Workflow-CI/mlruns/0/*/artifacts/model/model.pkl"
            import glob
            model_files = glob.glob(model_path)
            if model_files:
                with open(model_files[0], 'rb') as f:
                    model = pickle.load(f)
                model_version = "local_latest"
                logger.info(f"Model loaded from local file: {model_files[0]}")
            else:
                raise Exception("No model file found")
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        model_loaded_at = datetime.now()
        
        # Set initial model metrics (these would be updated with real evaluation)
        MODEL_ACCURACY.set(0.85)  # Example values
        MODEL_PRECISION.set(0.82)
        MODEL_RECALL.set(0.78)
        MODEL_F1_SCORE.set(0.80)
        MODEL_AUC_ROC.set(0.88)
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ERROR_COUNT.labels(error_type='model_load_error').inc()
        raise

def update_system_metrics():
    """Update system resource metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory_percent)
        DISK_USAGE.set(disk_percent)
        
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")
        ERROR_COUNT.labels(error_type='system_metrics_error').inc()

def update_throughput():
    """Update throughput metrics"""
    global request_times
    current_time = time.time()
    # Keep only requests from the last minute
    request_times = [t for t in request_times if current_time - t < 60]
    throughput = len(request_times) / 60.0  # requests per second
    THROUGHPUT.set(throughput)

def system_monitor():
    """Background thread to monitor system metrics"""
    while True:
        update_system_metrics()
        update_throughput()
        time.sleep(10)  # Update every 10 seconds

@app.before_request
def before_request():
    global active_requests
    active_requests += 1
    ACTIVE_CONNECTIONS.set(active_requests)
    request.start_time = time.time()

@app.after_request
def after_request(response):
    global active_requests, request_times
    active_requests -= 1
    ACTIVE_CONNECTIONS.set(active_requests)
    
    # Record request time
    request_times.append(time.time())
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()
    
    if hasattr(request, 'start_time'):
        REQUEST_LATENCY.observe(time.time() - request.start_time)
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_version': model_version,
        'model_loaded_at': model_loaded_at.isoformat() if model_loaded_at else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            ERROR_COUNT.labels(error_type='model_not_loaded').inc()
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        if not data:
            ERROR_COUNT.labels(error_type='invalid_input').inc()
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to DataFrame
        if isinstance(data, dict):
            if 'instances' in data:
                df = pd.DataFrame(data['instances'])
            else:
                df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            ERROR_COUNT.labels(error_type='invalid_input_format').inc()
            return jsonify({'error': 'Invalid input format'}), 400
        
        # Expected features
        expected_features = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        
        # Validate features
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            ERROR_COUNT.labels(error_type='missing_features').inc()
            return jsonify({
                'error': f'Missing features: {list(missing_features)}'
            }), 400
        
        # Select and order features
        df = df[expected_features]
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = None
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df)
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Update prediction metrics
        for pred in predictions:
            PREDICTION_COUNT.labels(prediction_class=str(pred)).inc()
        
        # Prepare response
        response_data = {
            'predictions': predictions.tolist(),
            'model_version': model_version,
            'timestamp': datetime.now().isoformat()
        }
        
        if probabilities is not None:
            response_data['probabilities'] = probabilities.tolist()
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_version': model_version,
        'model_loaded_at': model_loaded_at.isoformat() if model_loaded_at else None,
        'model_type': str(type(model).__name__),
        'features': [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Start system monitoring thread
    monitor_thread = threading.Thread(target=system_monitor, daemon=True)
    monitor_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)