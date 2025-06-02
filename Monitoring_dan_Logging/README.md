# Sistem Monitoring dan Logging ML

Sistem monitoring dan logging lengkap untuk model machine learning Credit Scoring dengan menggunakan Prometheus, Grafana, dan Alertmanager.

## 📋 Struktur Folder

```
Monitoring_dan_Logging/
├── 1.bukti_serving/                    # Screenshots bukti serving model
├── 2.prometheus.yml                    # Konfigurasi Prometheus
├── 3.prometheus_exporter.py            # Custom metrics exporter
├── 4.bukti_monitoring_Prometheus/      # Screenshots monitoring Prometheus
├── 5.bukti_monitoring_Grafana/         # Screenshots monitoring Grafana
├── 6.bukti_alerting_Grafana/          # Screenshots alerting Grafana
├── 7.inference.py                     # Service inference dengan metrics
├── alert_rules.yml                    # Rules untuk alerting
├── alertmanager.yml                   # Konfigurasi Alertmanager
├── docker-compose.yml                 # Orchestration semua services
├── Dockerfile.exporter                # Docker image untuk exporter
├── requirements.txt                   # Python dependencies
└── README.md                          # Dokumentasi ini
```

## 🚀 Quick Start

### 1. Persiapan Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Atau menggunakan Docker Compose (Recommended)
docker-compose up -d
```

### 2. Menjalankan Services

#### Opsi A: Manual (untuk development)

```bash
# Terminal 1: Jalankan inference service
python inference.py

# Terminal 2: Jalankan prometheus exporter
python prometheus_exporter.py

# Terminal 3: Jalankan Prometheus
prometheus --config.file=prometheus.yml

# Terminal 4: Jalankan Grafana
# Download dan install Grafana, lalu akses http://localhost:3000
```

#### Opsi B: Docker Compose (Recommended)

```bash
# Jalankan semua services sekaligus
docker-compose up -d

# Cek status services
docker-compose ps

# Lihat logs
docker-compose logs -f
```

### 3. Akses Services

- **ML Inference API**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Alertmanager**: http://localhost:9093
- **Custom Exporter**: http://localhost:8000/metrics

## 📊 Metrics yang Dimonitor

### Model Performance Metrics
- `ml_model_accuracy` - Akurasi model
- `ml_model_precision` - Precision model
- `ml_model_recall` - Recall model
- `ml_model_f1_score` - F1 score model
- `ml_model_auc_roc` - AUC-ROC score

### System Metrics
- `system_cpu_usage_percent` - Penggunaan CPU
- `system_memory_usage_percent` - Penggunaan memory
- `system_disk_usage_percent` - Penggunaan disk
- `ml_service_uptime_seconds` - Uptime service

### Request Metrics
- `ml_requests_total` - Total requests
- `ml_request_duration_seconds` - Latency requests
- `ml_predictions_total` - Total prediksi per kelas
- `ml_error_rate_percent` - Error rate
- `ml_throughput_requests_per_second` - Throughput

### Data Quality Metrics
- `ml_data_drift_score` - Skor data drift
- `ml_data_quality_missing_values_percent` - Persentase missing values
- `ml_data_quality_outliers_percent` - Persentase outliers

## 🔔 Alerting Rules

### Critical Alerts
- Model accuracy < 65%
- Service down
- Disk usage > 90%
- Error rate > 15%

### Warning Alerts
- Model accuracy < 75%
- CPU usage > 80%
- Memory usage > 85%
- Error rate > 5%
- Data drift detected

## 📈 Grafana Dashboards

### Dashboard 1: Model Performance
- Model accuracy, precision, recall, F1-score
- Prediction distribution
- Model version info

### Dashboard 2: System Health
- CPU, Memory, Disk usage
- Service uptime
- Request latency dan throughput

### Dashboard 3: Data Quality
- Data drift monitoring
- Missing values tracking
- Outlier detection

### Dashboard 4: Business Metrics
- Prediction volume
- Risk distribution
- Error analysis

## 🧪 Testing Inference Service

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.766127,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.802982,
    "MonthlyIncome": 9120,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "RevolvingUtilizationOfUnsecuredLines": 0.766127,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.802982,
        "MonthlyIncome": 9120,
        "NumberOfOpenCreditLinesAndLoans": 13,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 6,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
      }
    ]
  }'
```

### Get Metrics
```bash
curl http://localhost:5000/metrics
curl http://localhost:8000/metrics
```

## 🔧 Konfigurasi

### Environment Variables

```bash
# Inference Service
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_VERSION=latest
SERVICE_PORT=5000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EXPORTER_PORT=8000

# Alerting
ALERT_EMAIL=alerts@company.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Grafana Setup

1. Login ke Grafana (admin/admin123)
2. Add Prometheus data source: http://prometheus:9090
3. Import dashboard dari file JSON atau buat manual
4. Setup notification channels untuk alerting

## 📝 Logging

Semua services menggunakan structured logging dengan format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "ml-inference",
  "message": "Prediction completed",
  "request_id": "req-123",
  "latency_ms": 45,
  "prediction": 0
}
```

## 🚨 Troubleshooting

### Service tidak bisa diakses
```bash
# Cek status container
docker-compose ps

# Cek logs
docker-compose logs ml-inference
docker-compose logs prometheus
docker-compose logs grafana
```

### Model tidak bisa dimuat
```bash
# Cek apakah model file ada
ls -la ../Workflow-CI/mlruns/

# Cek MLflow tracking
curl http://localhost:5001/api/2.0/mlflow/experiments/list
```

### Metrics tidak muncul di Prometheus
```bash
# Cek endpoint metrics
curl http://localhost:5000/metrics
curl http://localhost:8000/metrics

# Cek konfigurasi Prometheus
cat prometheus.yml
```

## 📊 Penilaian Kriteria

### Basic (2 pts)
- ✅ Serving model melalui inference.py
- ✅ Monitoring dengan Prometheus (3+ metrics)
- ✅ Monitoring dengan Grafana (sama dengan Prometheus)

### Skilled (3 pts)
- ✅ Monitoring dengan Grafana (5+ metrics)
- ✅ Satu alerting menggunakan Grafana

### Advanced (4 pts)
- ✅ Monitoring dengan Grafana (10+ metrics)
- ✅ Tiga alerting menggunakan Grafana

## 🔗 Links

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📞 Support

Untuk pertanyaan atau issues, silakan buat issue di repository ini atau hubungi tim ML Engineering.