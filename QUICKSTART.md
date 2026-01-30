# ML Drift Sentinel - Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/ravigohel142996/ml-drift-sentinel.git
cd ml-drift-sentinel

# Install dependencies
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Using the API

```python
import pandas as pd
from core.drift_detector import DriftDetector
from core.confidence_monitor import ConfidenceMonitor
from core.performance_proxy import PerformanceProxy
from utils.alerts import AlertEngine

# Load your data
baseline = pd.read_csv('data/baseline_data.csv')
live = pd.read_csv('data/live_data.csv')

# Detect drift
detector = DriftDetector(baseline)
drift_results = detector.detect_drift(live)
summary = detector.get_drift_summary(drift_results)

print(f"High drift features: {summary['high_drift_count']}")
print(f"Average PSI: {summary['average_psi']:.4f}")

# Monitor confidence
monitor = ConfidenceMonitor()
monitor.add_predictions(your_predictions)
metrics = monitor.get_confidence_metrics()

print(f"Average confidence: {metrics['average_confidence']:.2%}")
print(f"Declining: {metrics['is_declining']}")

# Generate alerts
alert_engine = AlertEngine()
alerts = alert_engine.check_drift_alerts(drift_results)

for alert in alert_engine.alerts:
    print(f"[{alert.severity}] {alert.message}")
```

## Dashboard Pages

### 1. Overview
- System health metrics at a glance
- Drift status, confidence, reliability, active alerts
- Recent alerts and recommendations

### 2. Data Drift
- Detailed drift analysis for all features
- Distribution comparisons (baseline vs live)
- Statistical test results (KS, PSI, JS divergence)
- Top drifting features with explanations

### 3. Confidence Stability
- Prediction confidence trends over time
- Confidence decline and instability detection
- Trend analysis with R² scores
- Performance proxy comparison

### 4. Risk Alerts
- Active alerts by severity (Critical, High, Medium)
- Actionable recommendations
- Alert timeline and history
- Detailed alert information

## Key Metrics Explained

### Population Stability Index (PSI)
- **< 0.1**: No significant change
- **0.1 - 0.25**: Moderate change
- **≥ 0.25**: Significant change (retrain recommended)

### Jensen-Shannon Divergence
- **Range**: 0 to 1
- **< 0.2**: Low drift
- **0.2 - 0.5**: Medium drift
- **≥ 0.5**: High drift

### Confidence Metrics
- **Average Confidence**: Mean of max probabilities
- **Variance**: Prediction stability indicator
- **Declining**: >10% drop from baseline
- **Unstable**: High variance in recent predictions

## Alert Types

1. **RETRAIN_REQUIRED** (Critical) - Multiple high-severity drifts detected
2. **RETRAIN_RECOMMENDED** (High) - Moderate drift detected
3. **DATA_PIPELINE_ANOMALY** (High) - Upstream data issues suspected
4. **CONFIDENCE_COLLAPSE** (Critical) - Significant confidence drop
5. **CONFIDENCE_UNSTABLE** (High) - High variance in predictions
6. **LOW_RELIABILITY_ESTIMATE** (High) - Poor proxy metrics

## Configuration

### Adjusting Alert Thresholds

Edit `utils/alerts.py`:

```python
# Drift alerts
high_threshold = 3  # Features for critical alert
medium_threshold = 5  # Features for warning

# Confidence alerts
decline_threshold = 0.1  # 10% drop
instability_threshold = 0.15  # Variance threshold
```

### Adjusting Drift Classification

Edit `utils/metrics.py`:

```python
# PSI thresholds
psi_high = 0.25
psi_medium = 0.1

# JS divergence thresholds
js_high = 0.5
js_medium = 0.2
```

## Best Practices

1. **Baseline Data**: Use training or validation data as baseline
2. **Sample Size**: Minimum 100 samples for reliable drift detection
3. **Update Frequency**: Check drift daily or weekly depending on data volume
4. **Ground Truth**: Validate proxy metrics with actual labels when available
5. **Thresholds**: Start conservative and adjust based on false positive rate

## Troubleshooting

### "Data files not found"
- Ensure `baseline_data.csv` and `live_data.csv` exist in `data/` directory
- Check file paths are correct

### High false positive rate
- Increase drift thresholds in `utils/metrics.py`
- Increase alert thresholds in `utils/alerts.py`
- Check data quality and preprocessing

### Low sensitivity
- Decrease drift thresholds
- Use more bins for histogram comparison
- Check feature scaling

## Support

For issues or questions, please open an issue on GitHub.

## License

MIT License - see LICENSE file for details.
