import time
import logging
import threading
import psutil
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import sqlite3
import os
import json
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'Data drift detection score')
MODEL_PERFORMANCE_ACCURACY = Gauge('ml_model_performance_accuracy', 'Model accuracy on recent predictions')
MODEL_PERFORMANCE_PRECISION = Gauge('ml_model_performance_precision', 'Model precision on recent predictions')
MODEL_PERFORMANCE_RECALL = Gauge('ml_model_performance_recall', 'Model recall on recent predictions')
MODEL_PERFORMANCE_F1 = Gauge('ml_model_performance_f1', 'Model F1 score on recent predictions')
MODEL_PERFORMANCE_AUC = Gauge('ml_model_performance_auc', 'Model AUC-ROC on recent predictions')

PREDICTION_VOLUME = Gauge('ml_prediction_volume_hourly', 'Number of predictions in the last hour')
PREDICTION_LATENCY_P95 = Gauge('ml_prediction_latency_p95_seconds', '95th percentile prediction latency')
PREDICTION_LATENCY_P99 = Gauge('ml_prediction_latency_p99_seconds', '99th percentile prediction latency')

SYSTEM_CPU_CORES = Gauge('system_cpu_cores_total', 'Total number of CPU cores')
SYSTEM_MEMORY_TOTAL = Gauge('system_memory_total_bytes', 'Total system memory in bytes')
SYSTEM_DISK_TOTAL = Gauge('system_disk_total_bytes', 'Total disk space in bytes')
SYSTEM_DISK_FREE = Gauge('system_disk_free_bytes', 'Free disk space in bytes')

MODEL_VERSION_INFO = Info('ml_model_version', 'Information about the current model version')
SERVICE_UPTIME = Gauge('ml_service_uptime_seconds', 'Service uptime in seconds')
LAST_TRAINING_TIME = Gauge('ml_last_training_timestamp', 'Timestamp of last model training')

ERROR_RATE = Gauge('ml_error_rate_percent', 'Error rate percentage in the last hour')
THROUGHPUT_RPS = Gauge('ml_throughput_requests_per_second', 'Current requests per second')
CONCURRENT_REQUESTS = Gauge('ml_concurrent_requests', 'Number of concurrent requests')

DATA_QUALITY_MISSING_VALUES = Gauge('ml_data_quality_missing_values_percent', 'Percentage of missing values in recent data')
DATA_QUALITY_OUTLIERS = Gauge('ml_data_quality_outliers_percent', 'Percentage of outliers in recent data')

class MLMonitoringExporter:
    def __init__(self):
        self.start_time = time.time()
        self.prediction_history = []
        self.error_history = []
        self.latency_history = []
        self.inference_service_url = "http://localhost:5000"
        
        # Initialize database for storing metrics
        self.init_database()
        
        # Set static system info
        self.update_system_info()
        
    def init_database(self):
        """Initialize SQLite database for storing prediction history"""
        try:
            self.db_path = "monitoring_data.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    input_data TEXT,
                    prediction REAL,
                    probability REAL,
                    latency REAL,
                    model_version TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    error_type TEXT,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def update_system_info(self):
        """Update static system information"""
        try:
            SYSTEM_CPU_CORES.set(psutil.cpu_count())
            SYSTEM_MEMORY_TOTAL.set(psutil.virtual_memory().total)
            
            disk_usage = psutil.disk_usage('/')
            SYSTEM_DISK_TOTAL.set(disk_usage.total)
            SYSTEM_DISK_FREE.set(disk_usage.free)
            
            # Get model version info
            try:
                response = requests.get(f"{self.inference_service_url}/model/info", timeout=5)
                if response.status_code == 200:
                    model_info = response.json()
                    MODEL_VERSION_INFO.info({
                        'version': model_info.get('model_version', 'unknown'),
                        'type': model_info.get('model_type', 'unknown'),
                        'loaded_at': model_info.get('model_loaded_at', 'unknown')
                    })
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
                
        except Exception as e:
            logger.error(f"Failed to update system info: {e}")
    
    def calculate_data_drift(self, recent_data, reference_data):
        """Calculate data drift score using statistical methods"""
        try:
            if len(recent_data) == 0 or len(reference_data) == 0:
                return 0.0
            
            # Simple drift detection using KL divergence approximation
            drift_scores = []
            
            for column in recent_data.columns:
                if recent_data[column].dtype in ['int64', 'float64']:
                    # For numerical columns, use histogram comparison
                    recent_hist, bins = np.histogram(recent_data[column].dropna(), bins=10, density=True)
                    ref_hist, _ = np.histogram(reference_data[column].dropna(), bins=bins, density=True)
                    
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    recent_hist += epsilon
                    ref_hist += epsilon
                    
                    # Calculate KL divergence
                    kl_div = np.sum(recent_hist * np.log(recent_hist / ref_hist))
                    drift_scores.append(kl_div)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate data drift: {e}")
            return 0.0
    
    def calculate_data_quality_metrics(self, data):
        """Calculate data quality metrics"""
        try:
            if len(data) == 0:
                return 0.0, 0.0
            
            # Missing values percentage
            missing_percent = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            
            # Outliers percentage using IQR method
            outlier_count = 0
            total_values = 0
            
            for column in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                outlier_count += len(outliers)
                total_values += len(data[column].dropna())
            
            outlier_percent = (outlier_count / total_values * 100) if total_values > 0 else 0.0
            
            return missing_percent, outlier_percent
            
        except Exception as e:
            logger.error(f"Failed to calculate data quality metrics: {e}")
            return 0.0, 0.0
    
    def get_recent_predictions(self, hours=1):
        """Get recent predictions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get predictions from the last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT * FROM predictions 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_time,))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return pd.DataFrame()
    
    def get_recent_errors(self, hours=1):
        """Get recent errors from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT * FROM errors 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_time,))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recent errors: {e}")
            return pd.DataFrame()
    
    def update_prediction_metrics(self):
        """Update prediction-related metrics"""
        try:
            # Get recent predictions
            recent_predictions = self.get_recent_predictions(hours=1)
            
            if len(recent_predictions) > 0:
                # Prediction volume
                PREDICTION_VOLUME.set(len(recent_predictions))
                
                # Latency metrics
                latencies = recent_predictions['latency'].dropna()
                if len(latencies) > 0:
                    PREDICTION_LATENCY_P95.set(np.percentile(latencies, 95))
                    PREDICTION_LATENCY_P99.set(np.percentile(latencies, 99))
                
                # Throughput (requests per second)
                time_span_hours = 1.0
                rps = len(recent_predictions) / (time_span_hours * 3600)
                THROUGHPUT_RPS.set(rps)
            else:
                PREDICTION_VOLUME.set(0)
                THROUGHPUT_RPS.set(0)
            
            # Error rate
            recent_errors = self.get_recent_errors(hours=1)
            total_requests = len(recent_predictions) + len(recent_errors)
            
            if total_requests > 0:
                error_rate = (len(recent_errors) / total_requests) * 100
                ERROR_RATE.set(error_rate)
            else:
                ERROR_RATE.set(0)
                
        except Exception as e:
            logger.error(f"Failed to update prediction metrics: {e}")
    
    def update_model_performance_metrics(self):
        """Update model performance metrics using recent predictions"""
        try:
            # This is a simplified version - in practice, you'd need ground truth labels
            # For demonstration, we'll use synthetic performance metrics
            
            recent_predictions = self.get_recent_predictions(hours=24)  # Last 24 hours
            
            if len(recent_predictions) > 10:  # Need sufficient data
                # In a real scenario, you would:
                # 1. Get ground truth labels for recent predictions
                # 2. Calculate actual performance metrics
                # 3. Update the gauges with real values
                
                # For now, simulate performance metrics with some variation
                base_accuracy = 0.85
                base_precision = 0.82
                base_recall = 0.78
                base_f1 = 0.80
                base_auc = 0.88
                
                # Add some realistic variation based on recent prediction volume
                volume_factor = min(len(recent_predictions) / 100, 1.0)
                noise = np.random.normal(0, 0.02)  # Small random variation
                
                MODEL_PERFORMANCE_ACCURACY.set(max(0, min(1, base_accuracy + noise * volume_factor)))
                MODEL_PERFORMANCE_PRECISION.set(max(0, min(1, base_precision + noise * volume_factor)))
                MODEL_PERFORMANCE_RECALL.set(max(0, min(1, base_recall + noise * volume_factor)))
                MODEL_PERFORMANCE_F1.set(max(0, min(1, base_f1 + noise * volume_factor)))
                MODEL_PERFORMANCE_AUC.set(max(0, min(1, base_auc + noise * volume_factor)))
            
        except Exception as e:
            logger.error(f"Failed to update model performance metrics: {e}")
    
    def update_data_drift_metrics(self):
        """Update data drift metrics"""
        try:
            # Get recent prediction data
            recent_predictions = self.get_recent_predictions(hours=24)
            
            if len(recent_predictions) > 10:
                # Parse input data from recent predictions
                recent_data_list = []
                for _, row in recent_predictions.iterrows():
                    try:
                        input_data = json.loads(row['input_data'])
                        recent_data_list.append(input_data)
                    except:
                        continue
                
                if recent_data_list:
                    recent_df = pd.DataFrame(recent_data_list)
                    
                    # Load reference data (training data statistics)
                    # In practice, you'd load this from your training dataset
                    # For now, simulate reference data
                    reference_df = self.generate_reference_data()
                    
                    # Calculate drift score
                    drift_score = self.calculate_data_drift(recent_df, reference_df)
                    DATA_DRIFT_SCORE.set(drift_score)
                    
                    # Calculate data quality metrics
                    missing_percent, outlier_percent = self.calculate_data_quality_metrics(recent_df)
                    DATA_QUALITY_MISSING_VALUES.set(missing_percent)
                    DATA_QUALITY_OUTLIERS.set(outlier_percent)
            
        except Exception as e:
            logger.error(f"Failed to update data drift metrics: {e}")
    
    def generate_reference_data(self):
        """Generate reference data for drift detection (placeholder)"""
        # In practice, this would load your training dataset statistics
        # For demonstration, create synthetic reference data
        np.random.seed(42)
        n_samples = 1000
        
        reference_data = {
            'RevolvingUtilizationOfUnsecuredLines': np.random.beta(2, 5, n_samples),
            'age': np.random.normal(52, 14, n_samples),
            'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.4, n_samples),
            'DebtRatio': np.random.gamma(2, 0.2, n_samples),
            'MonthlyIncome': np.random.lognormal(10, 0.5, n_samples),
            'NumberOfOpenCreditLinesAndLoans': np.random.poisson(8, n_samples),
            'NumberOfTimes90DaysLate': np.random.poisson(0.2, n_samples),
            'NumberRealEstateLoansOrLines': np.random.poisson(1, n_samples),
            'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.1, n_samples),
            'NumberOfDependents': np.random.poisson(0.8, n_samples)
        }
        
        return pd.DataFrame(reference_data)
    
    def update_service_metrics(self):
        """Update service-level metrics"""
        try:
            # Service uptime
            uptime = time.time() - self.start_time
            SERVICE_UPTIME.set(uptime)
            
            # Check if inference service is healthy
            try:
                response = requests.get(f"{self.inference_service_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    # You could extract more metrics from health endpoint
            except Exception as e:
                logger.warning(f"Inference service health check failed: {e}")
            
            # Last training time (would come from MLflow or training logs)
            # For demonstration, set to a recent timestamp
            LAST_TRAINING_TIME.set(time.time() - 86400)  # 24 hours ago
            
        except Exception as e:
            logger.error(f"Failed to update service metrics: {e}")
    
    def collect_metrics(self):
        """Main method to collect all metrics"""
        logger.info("Collecting metrics...")
        
        self.update_prediction_metrics()
        self.update_model_performance_metrics()
        self.update_data_drift_metrics()
        self.update_service_metrics()
        
        logger.info("Metrics collection completed")
    
    def run(self):
        """Main monitoring loop"""
        logger.info("Starting ML Monitoring Exporter")
        
        # Start Prometheus metrics server
        start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")
        
        # Main monitoring loop
        while True:
            try:
                self.collect_metrics()
                time.sleep(30)  # Collect metrics every 30 seconds
            except KeyboardInterrupt:
                logger.info("Monitoring exporter stopped")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)

if __name__ == "__main__":
    exporter = MLMonitoringExporter()
    exporter.run()