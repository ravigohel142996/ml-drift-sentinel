"""
ML Drift Sentinel - Production-Grade ML Monitoring Dashboard
Executive dashboard for silent model failure detection
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.drift_detector import DriftDetector
from core.confidence_monitor import ConfidenceMonitor
from core.performance_proxy import PerformanceProxy
from explain.feature_shift import FeatureShiftAnalyzer
from utils.alerts import AlertEngine

# Page configuration
st.set_page_config(
    page_title="ML Drift Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    .alert-critical {
        background-color: #fee;
        padding: 15px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #d1ecf1;
        padding: 15px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load baseline and live data"""
    baseline = pd.read_csv('data/baseline_data.csv')
    live = pd.read_csv('data/live_data.csv')
    return baseline, live


@st.cache_data
def detect_drift(_baseline, _live):
    """Detect drift between baseline and live data"""
    detector = DriftDetector(_baseline)
    results = detector.detect_drift(_live)
    summary = detector.get_drift_summary(results)
    report = detector.generate_drift_report(results)
    return results, summary, report


def generate_sample_predictions(data, confidence_level='medium'):
    """
    Generate sample prediction probabilities
    In production, these would come from your model
    """
    n_samples = len(data)
    
    if confidence_level == 'high':
        # High confidence predictions
        probs = np.random.beta(8, 2, n_samples)
    elif confidence_level == 'medium':
        # Medium confidence predictions
        probs = np.random.beta(4, 4, n_samples)
    else:
        # Low confidence predictions (more drift)
        probs = np.random.beta(2, 2, n_samples)
    
    # Create binary probabilities
    predictions = np.column_stack([probs, 1 - probs])
    return predictions


def main():
    # Sidebar navigation
    st.sidebar.title("🛡️ ML Drift Sentinel")
    st.sidebar.markdown("### Production ML Monitoring")
    
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Overview", "📉 Data Drift", "🎯 Confidence Stability", "⚠️ Risk Alerts"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This system detects silent ML model failures caused by data drift, "
        "feature distribution shift, and prediction confidence decay — even when "
        "ground-truth labels are unavailable."
    )
    
    # Load data
    try:
        baseline_data, live_data = load_data()
    except FileNotFoundError:
        st.error("Data files not found. Please ensure baseline_data.csv and live_data.csv exist in the data/ directory.")
        return
    
    # Detect drift
    drift_results, drift_summary, drift_report = detect_drift(baseline_data, live_data)
    
    # Initialize components
    analyzer = FeatureShiftAnalyzer()
    alert_engine = AlertEngine()
    
    # Generate alerts
    drift_alerts = alert_engine.check_drift_alerts(drift_results)
    
    # Generate sample predictions for demonstration
    baseline_predictions = generate_sample_predictions(baseline_data, 'high')
    live_predictions = generate_sample_predictions(live_data, 'medium')
    
    # Confidence monitoring
    conf_monitor = ConfidenceMonitor()
    for i in range(len(baseline_predictions)):
        conf_monitor.add_predictions(baseline_predictions[i:i+1])
    for i in range(len(live_predictions)):
        conf_monitor.add_predictions(live_predictions[i:i+1])
    
    conf_metrics = conf_monitor.get_confidence_metrics()
    conf_alerts = alert_engine.check_confidence_alerts(conf_metrics)
    
    # Performance proxy
    proxy = PerformanceProxy()
    proxy_metrics = proxy.get_proxy_metrics(live_predictions)
    proxy_alerts = alert_engine.check_performance_proxy_alerts(proxy_metrics)
    
    # Render selected page
    if page == "📊 Overview":
        render_overview(drift_summary, conf_metrics, proxy_metrics, alert_engine)
    elif page == "📉 Data Drift":
        render_drift_page(drift_results, drift_summary, drift_report, baseline_data, live_data, analyzer)
    elif page == "🎯 Confidence Stability":
        render_confidence_page(conf_monitor, conf_metrics, baseline_predictions, live_predictions)
    elif page == "⚠️ Risk Alerts":
        render_alerts_page(alert_engine, drift_results, conf_metrics, proxy_metrics)


def render_overview(drift_summary, conf_metrics, proxy_metrics, alert_engine):
    """Render overview dashboard"""
    st.title("📊 System Overview")
    st.markdown("### ML Model Health Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drift_status = "🔴 HIGH" if drift_summary['high_drift_count'] >= 3 else \
                      "🟡 MEDIUM" if drift_summary['medium_drift_count'] >= 3 else "🟢 LOW"
        st.metric(
            "Drift Status",
            drift_status,
            f"{drift_summary['high_drift_count']} features"
        )
    
    with col2:
        conf_status = "🔴 UNSTABLE" if conf_metrics['is_unstable'] else "🟢 STABLE"
        st.metric(
            "Confidence",
            conf_status,
            f"{conf_metrics['average_confidence']:.2%}"
        )
    
    with col3:
        reliability_color = "🔴" if proxy_metrics['estimated_reliability'] < 0.7 else \
                           "🟡" if proxy_metrics['estimated_reliability'] < 0.8 else "🟢"
        st.metric(
            "Est. Reliability",
            f"{reliability_color} {proxy_metrics['estimated_reliability']:.2%}",
            f"Risk: {proxy_metrics['risk_level']}"
        )
    
    with col4:
        alert_count = len(alert_engine.alerts)
        alert_status = "🔴" if alert_count >= 3 else "🟡" if alert_count > 0 else "🟢"
        st.metric(
            "Active Alerts",
            f"{alert_status} {alert_count}",
            "alerts"
        )
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Drift Summary")
        st.markdown(f"- **High Drift Features**: {drift_summary['high_drift_count']}")
        st.markdown(f"- **Medium Drift Features**: {drift_summary['medium_drift_count']}")
        st.markdown(f"- **Average PSI**: {drift_summary['average_psi']:.3f}")
        
        if drift_summary['drift_detected']:
            st.warning("⚠️ Significant drift detected. Review Data Drift page for details.")
        else:
            st.success("✅ No significant drift detected.")
    
    with col2:
        st.markdown("### 🎯 Model Health")
        st.markdown(f"- **Average Confidence**: {conf_metrics['average_confidence']:.2%}")
        st.markdown(f"- **Confidence Variance**: {conf_metrics['variance']:.4f}")
        st.markdown(f"- **Estimated Reliability**: {proxy_metrics['estimated_reliability']:.2%}")
        
        if conf_metrics['is_declining']:
            st.warning(f"⚠️ Confidence declining by {conf_metrics['confidence_decline']:.1%}")
        else:
            st.success("✅ Confidence levels stable.")
    
    # Recent alerts
    if alert_engine.alerts:
        st.markdown("---")
        st.markdown("### 🚨 Recent Alerts")
        for alert in alert_engine.alerts[-5:]:  # Show last 5
            severity_color = {
                'CRITICAL': 'alert-critical',
                'HIGH': 'alert-high',
                'MEDIUM': 'alert-medium'
            }.get(alert.severity, 'alert-medium')
            
            st.markdown(
                f'<div class="{severity_color}"><strong>{alert.alert_type}</strong>: {alert.message}</div>',
                unsafe_allow_html=True
            )
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    recommendations = alert_engine.get_actionable_recommendations()
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ No immediate actions required. System operating normally.")


def render_drift_page(drift_results, drift_summary, drift_report, baseline_data, live_data, analyzer):
    """Render data drift analysis page"""
    st.title("📉 Data Drift Analysis")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Features", drift_summary['total_features'])
    with col2:
        st.metric("High Drift", drift_summary['high_drift_count'], 
                 delta=None, delta_color="inverse")
    with col3:
        st.metric("Medium Drift", drift_summary['medium_drift_count'])
    with col4:
        st.metric("Avg PSI", f"{drift_summary['average_psi']:.3f}")
    
    st.markdown("---")
    
    # Drift report table
    st.markdown("### Drift Detection Results")
    
    # Color code by severity
    def highlight_severity(row):
        if row['Severity'] == 'High':
            return ['background-color: #fee'] * len(row)
        elif row['Severity'] == 'Medium':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return [''] * len(row)
    
    styled_report = drift_report.style.apply(highlight_severity, axis=1)
    st.dataframe(styled_report, use_container_width=True)
    
    # Top drifted features
    st.markdown("---")
    st.markdown("### 🔍 Top Drifting Features")
    
    top_features = analyzer.rank_unstable_features(drift_results, top_n=5)
    
    for i, feat in enumerate(top_features, 1):
        with st.expander(f"{i}. {feat['feature']} ({feat['severity']} severity)"):
            explanation = analyzer.generate_explanation(feat)
            st.markdown(explanation)
            
            # Distribution comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Statistics:**")
                st.write(f"- PSI: {feat['psi']:.3f}")
                st.write(f"- JS Divergence: {feat['js_divergence']:.3f}")
                st.write(f"- KS Statistic: {feat['ks_statistic']:.3f}")
            
            with col2:
                st.markdown("**Distribution Shift:**")
                st.write(f"- Baseline Mean: {feat['baseline_mean']:.2f}")
                st.write(f"- Live Mean: {feat['live_mean']:.2f}")
                st.write(f"- Change: {feat['pct_change']:.1f}%")
            
            # Plot distributions
            fig = go.Figure()
            
            feature_name = feat['feature']
            fig.add_trace(go.Histogram(
                x=baseline_data[feature_name],
                name='Baseline',
                opacity=0.6,
                marker_color='blue'
            ))
            fig.add_trace(go.Histogram(
                x=live_data[feature_name],
                name='Live',
                opacity=0.6,
                marker_color='red'
            ))
            
            fig.update_layout(
                title=f'{feature_name} Distribution Comparison',
                xaxis_title=feature_name,
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_confidence_page(conf_monitor, conf_metrics, baseline_preds, live_preds):
    """Render confidence monitoring page"""
    st.title("🎯 Prediction Confidence Stability")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Confidence", f"{conf_metrics['average_confidence']:.2%}")
    with col2:
        st.metric("Variance", f"{conf_metrics['variance']:.4f}")
    with col3:
        status = "🔴 Yes" if conf_metrics['is_declining'] else "🟢 No"
        st.metric("Declining?", status)
    with col4:
        status = "🔴 Yes" if conf_metrics['is_unstable'] else "🟢 No"
        st.metric("Unstable?", status)
    
    st.markdown("---")
    
    # Confidence over time
    st.markdown("### Confidence Trend")
    
    if len(conf_monitor.confidence_history) > 0:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=conf_monitor.confidence_history,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2)
        ))
        
        # Add rolling average
        rolling_avg = conf_monitor.calculate_rolling_average()
        if len(rolling_avg) > 0:
            fig.add_trace(go.Scatter(
                y=rolling_avg,
                mode='lines',
                name='Rolling Average',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Prediction Confidence Over Time',
            xaxis_title='Batch',
            yaxis_title='Confidence',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.markdown("---")
    st.markdown("### 📊 Trend Analysis")
    
    trend = conf_monitor.analyze_trend()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend['trend'], "❓")
        st.metric("Trend", f"{trend_emoji} {trend['trend'].title()}")
    
    with col2:
        st.metric("Slope", f"{trend['slope']:.6f}")
    
    with col3:
        st.metric("R²", f"{trend['r_squared']:.3f}")
    
    # Performance proxy comparison
    st.markdown("---")
    st.markdown("### 🔬 Performance Proxy Analysis")
    
    proxy = PerformanceProxy()
    comparison = proxy.compare_baseline(baseline_preds, live_preds)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline (Training)**")
        st.write(f"- Reliability: {comparison['baseline_reliability']:.2%}")
        st.write(f"- Avg Confidence: {comparison['baseline_confidence']:.2%}")
        st.write(f"- Avg Entropy: {comparison['baseline_entropy']:.3f}")
    
    with col2:
        st.markdown("**Live (Production)**")
        st.write(f"- Reliability: {comparison['current_reliability']:.2%}")
        st.write(f"- Avg Confidence: {comparison['current_confidence']:.2%}")
        st.write(f"- Avg Entropy: {comparison['current_entropy']:.3f}")
    
    st.markdown("**Shifts:**")
    st.write(f"- Reliability Shift: {comparison['reliability_shift']:.2%}")
    st.write(f"- Confidence Shift: {comparison['confidence_shift']:.2%}")
    st.write(f"- Entropy Shift: {comparison['entropy_shift']:.3f}")
    
    if comparison['degradation_detected']:
        st.warning("⚠️ Performance degradation detected based on proxy metrics")
    else:
        st.success("✅ No significant performance degradation detected")


def render_alerts_page(alert_engine, drift_results, conf_metrics, proxy_metrics):
    """Render alerts and recommendations page"""
    st.title("⚠️ Risk Alerts & Recommendations")
    
    # Alert summary
    alert_summary = alert_engine.generate_alert_summary()
    
    if len(alert_engine.alerts) == 0:
        st.success("✅ No active alerts. System operating normally.")
        return
    
    # Count by severity
    severity_counts = alert_summary['Severity'].value_counts().to_dict()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Critical", severity_counts.get('CRITICAL', 0))
    with col2:
        st.metric("🟡 High", severity_counts.get('HIGH', 0))
    with col3:
        st.metric("🔵 Medium", severity_counts.get('MEDIUM', 0))
    
    st.markdown("---")
    
    # Show all alerts
    st.markdown("### Active Alerts")
    
    for alert in alert_engine.alerts:
        severity_color = {
            'CRITICAL': 'alert-critical',
            'HIGH': 'alert-high',
            'MEDIUM': 'alert-medium'
        }.get(alert.severity, 'alert-medium')
        
        with st.expander(f"{alert.severity}: {alert.alert_type}"):
            st.markdown(f"**Message:** {alert.message}")
            st.markdown(f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if 'recommendation' in alert.details:
                st.markdown(f"**Recommendation:** {alert.details['recommendation']}")
            
            if 'risk_level' in alert.details:
                st.markdown(f"**Risk Level:** {alert.details['risk_level']}")
            
            if 'affected_features' in alert.details:
                st.markdown(f"**Affected Features:** {', '.join(alert.details['affected_features'])}")
            
            if 'possible_causes' in alert.details:
                st.markdown("**Possible Causes:**")
                for cause in alert.details['possible_causes']:
                    st.markdown(f"- {cause}")
    
    # Actionable recommendations
    st.markdown("---")
    st.markdown("### 💡 Actionable Recommendations")
    
    recommendations = alert_engine.get_actionable_recommendations()
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"{rec}")
    else:
        st.info("No specific recommendations at this time.")
    
    # Alert timeline
    st.markdown("---")
    st.markdown("### Alert Timeline")
    
    if not alert_summary.empty:
        st.dataframe(alert_summary, use_container_width=True)


if __name__ == "__main__":
    main()
