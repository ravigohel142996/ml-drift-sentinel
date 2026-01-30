# ML Drift Sentinel 🛡️

**Production-Grade Machine Learning Monitoring System**

Detect silent ML model failures caused by data drift, feature distribution shift, and prediction confidence decay — even when ground-truth labels are delayed or unavailable.

## 🎯 Overview

ML Drift Sentinel is an enterprise-level AI reliability platform designed for production ML systems. It provides comprehensive monitoring, alerting, and explainability for detecting model degradation without requiring immediate ground truth labels.

### Key Features

- **Data Drift Detection**: Statistical tests (KS test, PSI, Jensen-Shannon) for distribution comparison
- **Confidence Monitoring**: Track prediction confidence decay and instability
- **Performance Proxy**: Estimate reliability without labels using entropy analysis
- **Explainability**: Identify and explain which features are drifting and why
- **Alert Engine**: Rule-based, interpretable, and auditable alerts
- **Executive Dashboard**: Clean, professional Streamlit UI for monitoring

## 🏗️ Architecture

```
ml-drift-sentinel/
│
├── app.py                     # Streamlit dashboard (monitoring UI)
├── requirements.txt           # Python dependencies
├── README.md
│
├── data/
│   ├── baseline_data.csv      # Training / reference distribution
│   └── live_data.csv          # Production data
│
├── core/
│   ├── drift_detector.py      # Data & feature drift detection
│   ├── confidence_monitor.py  # Prediction confidence tracking
│   └── performance_proxy.py   # Accuracy estimation without labels
│
├── explain/
│   └── feature_shift.py       # Feature attribution & drift explanations
│
└── utils/
    ├── metrics.py             # Statistical metrics and tests
    └── alerts.py              # Alert generation engine
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ravigohel142996/ml-drift-sentinel.git
cd ml-drift-sentinel

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Pages

### 1. Overview
- System health at a glance
- Key metrics: drift status, confidence, reliability, alerts
- Recent alerts and recommendations

### 2. Data Drift
- Detailed drift analysis for all features
- Distribution comparisons (baseline vs live)
- Top drifting features with explanations
- PSI, JS divergence, and KS test results

### 3. Confidence Stability
- Prediction confidence trends over time
- Confidence decline detection
- Instability analysis
- Performance proxy comparison

### 4. Risk Alerts
- Active alerts by severity
- Actionable recommendations
- Alert timeline and history
- Root cause analysis

## 🔬 Technical Details

### Data Drift Detection

Uses three complementary statistical tests:

1. **Kolmogorov-Smirnov Test**: Non-parametric distribution comparison
2. **Population Stability Index (PSI)**: Industry standard from credit risk
   - PSI < 0.1: No significant change
   - 0.1 ≤ PSI < 0.25: Moderate change
   - PSI ≥ 0.25: Significant change
3. **Jensen-Shannon Divergence**: Symmetric KL divergence (0-1 range)

### Confidence Monitoring

- Rolling average confidence tracking
- Trend analysis with linear regression
- Variance-based instability detection
- Confidence decline alerts (>10% drop)

### Performance Proxy

Estimates model reliability without ground truth using:
- Prediction entropy analysis
- Confidence distribution statistics
- Temporal variance detection
- Risk score calculation

### Alert Engine

Rule-based alerts with clear thresholds:
- **RETRAIN_REQUIRED** (Critical): Multiple high-severity drifts
- **RETRAIN_RECOMMENDED** (High): Moderate drift detected
- **DATA_PIPELINE_ANOMALY** (High): Upstream data issues
- **CONFIDENCE_COLLAPSE** (Critical): Significant confidence drop
- **CONFIDENCE_UNSTABLE** (High): High variance in predictions
- **LOW_RELIABILITY_ESTIMATE** (High): Poor proxy metrics

## 📈 Usage Example

```python
from core.drift_detector import DriftDetector
from core.confidence_monitor import ConfidenceMonitor
from core.performance_proxy import PerformanceProxy
from utils.alerts import AlertEngine
import pandas as pd

# Load data
baseline = pd.read_csv('data/baseline_data.csv')
live = pd.read_csv('data/live_data.csv')

# Detect drift
detector = DriftDetector(baseline)
drift_results = detector.detect_drift(live)
summary = detector.get_drift_summary(drift_results)

# Monitor confidence
monitor = ConfidenceMonitor()
monitor.add_predictions(predictions)
conf_metrics = monitor.get_confidence_metrics()

# Generate alerts
alert_engine = AlertEngine()
alerts = alert_engine.check_drift_alerts(drift_results)
```

## 🎓 Design Principles

1. **Production-Safe**: Conservative thresholds to minimize false positives
2. **Explainable**: Every metric and alert has clear reasoning
3. **Auditable**: All decisions are logged with full context
4. **Modular**: Clean separation of concerns
5. **Extensible**: Easy to add new metrics or alert types

## ⚙️ Configuration

### Alert Thresholds

Edit thresholds in `utils/alerts.py`:

```python
# Drift alerts
high_threshold = 3  # Number of high-drift features for critical alert
medium_threshold = 5  # Number of medium+ drift features for warning

# Confidence alerts
decline_threshold = 0.1  # 10% confidence drop
instability_threshold = 0.15  # Variance threshold

# Reliability alerts
reliability_threshold = 0.7  # Minimum acceptable reliability
```

### Drift Metrics

Adjust drift classification in `utils/metrics.py`:

```python
# PSI thresholds
psi_high = 0.25
psi_medium = 0.1

# JS divergence thresholds
js_high = 0.5
js_medium = 0.2

# KS statistic thresholds
ks_high = 0.3
ks_medium = 0.15
```

## 🔒 Security & Privacy

- No external API calls
- All data processed locally
- No sensitive information logged
- Audit trail for all alerts

## 🤝 Contributing

This is a production-grade system. When contributing:
- Maintain code quality and documentation
- Add tests for new features
- Follow existing patterns and conventions
- Update README for significant changes

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built using best practices from production ML systems at scale, incorporating techniques from:
- Credit risk modeling (PSI)
- Healthcare AI monitoring
- Financial ML systems
- Large-scale tech company ML platforms

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a monitoring and alerting system. It does not replace proper model validation with ground truth labels. Always validate model performance with actual labels when available.
