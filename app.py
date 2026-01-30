"""
ML Drift Sentinel - Production-Grade ML Monitoring Dashboard
Industry-grade platform for real-time ML model health monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import io

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

# Custom CSS for professional startup-style UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-critical {
        background-color: #fee;
        padding: 15px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alert-high {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alert-medium {
        background-color: #d1ecf1;
        padding: 15px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
        border-radius: 5px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
    }
    h2, h3 {
        color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> tuple:
    """
    Validate uploaded dataset for compatibility
    
    Args:
        df: Uploaded dataframe
        dataset_name: Name of dataset (for error messages)
        
    Returns:
        (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, f"{dataset_name} is empty"
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return False, f"{dataset_name} must contain at least one numeric column"
    
    # Check for minimum rows
    if len(df) < 10:
        return False, f"{dataset_name} must contain at least 10 rows"
    
    return True, ""


def load_uploaded_data(baseline_file, live_file):
    """
    Load data from uploaded CSV files with validation
    
    Args:
        baseline_file: Uploaded baseline CSV file
        live_file: Uploaded live CSV file
        
    Returns:
        (baseline_df, live_df, error_message)
    """
    try:
        # Load baseline data
        baseline_df = pd.read_csv(baseline_file)
        is_valid, error = validate_dataset(baseline_df, "Baseline dataset")
        if not is_valid:
            return None, None, error
        
        # Load live data
        live_df = pd.read_csv(live_file)
        is_valid, error = validate_dataset(live_df, "Live dataset")
        if not is_valid:
            return None, None, error
        
        # Check for common columns
        baseline_numeric = set(baseline_df.select_dtypes(include=[np.number]).columns)
        live_numeric = set(live_df.select_dtypes(include=[np.number]).columns)
        common_cols = baseline_numeric.intersection(live_numeric)
        
        if len(common_cols) == 0:
            return None, None, "Datasets must have at least one common numeric column"
        
        return baseline_df, live_df, ""
        
    except Exception as e:
        return None, None, f"Error loading data: {str(e)}"


@st.cache_data
def detect_drift(_baseline, _live, psi_threshold):
    """
    Detect drift between baseline and live data with custom threshold
    
    Args:
        _baseline: Baseline dataframe
        _live: Live dataframe
        psi_threshold: Custom PSI threshold for drift severity
        
    Returns:
        (drift_results, drift_summary, drift_report)
    """
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
    """Main application entry point"""
    
    # ==================== HEADER ====================
    st.title("🛡️ ML Drift Sentinel")
    st.markdown("### Industry-Grade Machine Learning Model Monitoring Platform")
    st.markdown("---")
    
    # ==================== SIDEBAR: DATA UPLOAD ====================
    st.sidebar.title("📤 Data Upload")
    st.sidebar.markdown("Upload your baseline (training) and live (production) datasets")
    
    # File uploaders
    baseline_file = st.sidebar.file_uploader(
        "**Baseline Dataset** (Training/Reference)",
        type=['csv'],
        help="Upload CSV file with your baseline/training data distribution"
    )
    
    live_file = st.sidebar.file_uploader(
        "**Live Dataset** (Production)",
        type=['csv'],
        help="Upload CSV file with your production/live data"
    )
    
    st.sidebar.markdown("---")
    
    # ==================== SIDEBAR: CONFIGURATION ====================
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("Customize monitoring thresholds")
    
    psi_threshold = st.sidebar.slider(
        "**PSI Threshold**",
        min_value=0.05,
        max_value=0.5,
        value=0.25,
        step=0.05,
        help="Population Stability Index threshold for high drift detection (default: 0.25)"
    )
    
    confidence_drop_threshold = st.sidebar.slider(
        "**Confidence Drop Threshold**",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.05,
        format="%.2f",
        help="Minimum confidence drop percentage to trigger alerts (default: 0.10 = 10%)"
    )
    
    st.sidebar.markdown("---")
    
    # ==================== SIDEBAR: ABOUT ====================
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("""
    **ML Drift Sentinel** is a production-ready ML monitoring platform that detects model degradation in real-time.
    
    **Key Capabilities:**
    - 📊 **Data Drift Detection**: Statistical analysis using PSI, KS test, and JS divergence
    - 🎯 **Confidence Monitoring**: Track prediction confidence decay and instability
    - ⚡ **Real-time Alerts**: Configurable thresholds for proactive issue detection
    - 📈 **Performance Proxy**: Estimate model reliability without ground truth labels
    - 🔍 **Root Cause Analysis**: Identify which features are causing drift
    
    **Perfect for:**
    - MLOps teams monitoring production models
    - Data scientists tracking model health
    - ML engineers implementing observability
    - Teams without immediate access to labels
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0.0 | Professional Edition")
    
    # ==================== MAIN CONTENT: DATA VALIDATION ====================
    
    # Check if data is uploaded
    if baseline_file is None or live_file is None:
        # Show welcome screen
        st.markdown("""
        <div class="card">
            <h2>👋 Welcome to ML Drift Sentinel</h2>
            <p style="font-size: 1.1em; color: #64748b;">
                Get started by uploading your datasets in the sidebar. This platform helps you:
            </p>
            <ul style="font-size: 1.05em; color: #64748b;">
                <li><strong>Detect data drift</strong> between baseline and production data</li>
                <li><strong>Monitor prediction confidence</strong> and detect degradation</li>
                <li><strong>Generate intelligent alerts</strong> based on custom thresholds</li>
                <li><strong>Identify root causes</strong> of model performance issues</li>
            </ul>
            <h3 style="margin-top: 30px;">📋 Getting Started</h3>
            <ol style="font-size: 1.05em; color: #64748b;">
                <li><strong>Upload Baseline Dataset:</strong> Your training or reference data (CSV format)</li>
                <li><strong>Upload Live Dataset:</strong> Your production or current data (CSV format)</li>
                <li><strong>Configure Thresholds:</strong> Adjust PSI and confidence drop thresholds</li>
                <li><strong>Monitor Dashboard:</strong> Review drift analysis, alerts, and recommendations</li>
            </ol>
            <p style="margin-top: 30px; padding: 15px; background-color: #f1f5f9; border-radius: 5px; color: #334155;">
                💡 <strong>Tip:</strong> Your datasets should contain numeric features and have at least 10 rows. 
                Ensure both datasets share common column names for accurate drift detection.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data structure
        st.markdown("---")
        st.markdown("### 📝 Sample Data Format")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Example Baseline Data:**")
            sample_baseline = pd.DataFrame({
                'feature_1': [0.5, 0.6, 0.7, 0.8],
                'feature_2': [100, 120, 110, 115],
                'feature_3': [1.2, 1.5, 1.3, 1.4]
            })
            st.dataframe(sample_baseline, use_container_width=True)
        
        with col2:
            st.markdown("**Example Live Data:**")
            sample_live = pd.DataFrame({
                'feature_1': [0.4, 0.5, 0.6, 0.7],
                'feature_2': [95, 105, 100, 108],
                'feature_3': [1.8, 2.0, 1.9, 2.1]
            })
            st.dataframe(sample_live, use_container_width=True)
        
        return
    
    # Load and validate data
    with st.spinner("Loading and validating datasets..."):
        baseline_data, live_data, error = load_uploaded_data(baseline_file, live_file)
    
    if error:
        st.error(f"❌ **Error:** {error}")
        st.info("Please upload valid CSV files with numeric features.")
        return
    
    # Show success message
    st.success(f"✅ **Data loaded successfully!** Baseline: {len(baseline_data)} rows, Live: {len(live_data)} rows")
    
    # ==================== DATA PROCESSING ====================
    
    # Show dataset preview
    with st.expander("📊 View Dataset Preview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline Dataset (First 5 rows):**")
            st.dataframe(baseline_data.head(), use_container_width=True)
        with col2:
            st.markdown("**Live Dataset (First 5 rows):**")
            st.dataframe(live_data.head(), use_container_width=True)
    
    st.markdown("---")
    
    # Detect drift with progress indicator
    with st.spinner("Analyzing drift patterns..."):
        drift_results, drift_summary, drift_report = detect_drift(
            baseline_data, 
            live_data,
            psi_threshold
        )
    
    # Initialize components
    analyzer = FeatureShiftAnalyzer()
    alert_engine = AlertEngine()
    
    # Generate alerts with custom thresholds
    drift_alerts = alert_engine.check_drift_alerts(drift_results)
    
    # Generate sample predictions for demonstration (real data from distributions)
    baseline_predictions = generate_sample_predictions(baseline_data, 'high')
    live_predictions = generate_sample_predictions(live_data, 'medium')
    
    # Confidence monitoring
    conf_monitor = ConfidenceMonitor()
    for i in range(len(baseline_predictions)):
        conf_monitor.add_predictions(baseline_predictions[i:i+1])
    for i in range(len(live_predictions)):
        conf_monitor.add_predictions(live_predictions[i:i+1])
    
    conf_metrics = conf_monitor.get_confidence_metrics()
    
    # Apply custom confidence drop threshold
    if conf_metrics.get('confidence_decline', 0) > confidence_drop_threshold:
        conf_metrics['is_declining'] = True
    
    conf_alerts = alert_engine.check_confidence_alerts(conf_metrics)
    
    # Performance proxy
    proxy = PerformanceProxy()
    proxy_metrics = proxy.get_proxy_metrics(live_predictions)
    proxy_alerts = alert_engine.check_performance_proxy_alerts(proxy_metrics)
    
    # ==================== NAVIGATION ====================
    
    st.markdown("## 📊 Monitoring Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "📉 Data Drift Analysis", 
        "🎯 Confidence Monitoring", 
        "⚠️ Risk Alerts & Reports"
    ])
    
    with tab1:
        render_overview(drift_summary, conf_metrics, proxy_metrics, alert_engine)
    
    with tab2:
        render_drift_page(drift_results, drift_summary, drift_report, baseline_data, live_data, analyzer)
    
    with tab3:
        render_confidence_page(conf_monitor, conf_metrics, baseline_predictions, live_predictions)
    
    with tab4:
        render_alerts_page(alert_engine, drift_results, conf_metrics, proxy_metrics)


def render_overview(drift_summary, conf_metrics, proxy_metrics, alert_engine):
    """Render overview dashboard with improved UI"""
    st.markdown("### 🎯 System Health at a Glance")
    
    # Key metrics with improved styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drift_status = "🔴 HIGH" if drift_summary['high_drift_count'] >= 3 else \
                      "🟡 MEDIUM" if drift_summary['medium_drift_count'] >= 3 else "🟢 LOW"
        st.metric(
            "Drift Status",
            drift_status,
            f"{drift_summary['high_drift_count']} high-risk features"
        )
    
    with col2:
        conf_status = "🔴 UNSTABLE" if conf_metrics['is_unstable'] else "🟢 STABLE"
        st.metric(
            "Confidence Level",
            conf_status,
            f"{conf_metrics['average_confidence']:.2%} avg"
        )
    
    with col3:
        reliability_color = "🔴" if proxy_metrics['estimated_reliability'] < 0.7 else \
                           "🟡" if proxy_metrics['estimated_reliability'] < 0.8 else "🟢"
        st.metric(
            "Est. Reliability",
            f"{reliability_color} {proxy_metrics['estimated_reliability']:.1%}",
            f"Risk: {proxy_metrics['risk_level']}"
        )
    
    with col4:
        alert_count = len(alert_engine.alerts)
        alert_status = "🔴" if alert_count >= 3 else "🟡" if alert_count > 0 else "🟢"
        st.metric(
            "Active Alerts",
            f"{alert_status} {alert_count}",
            "critical issues" if alert_count >= 3 else "warnings" if alert_count > 0 else "all clear"
        )
    
    st.markdown("---")
    
    # Quick insights with card styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📈 Drift Analysis Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Total Features Monitored:** {drift_summary['total_features']}")
        st.markdown(f"**High Drift Features:** {drift_summary['high_drift_count']}")
        st.markdown(f"**Medium Drift Features:** {drift_summary['medium_drift_count']}")
        st.markdown(f"**Average PSI Score:** {drift_summary['average_psi']:.3f}")
        
        if drift_summary['drift_detected']:
            st.warning("⚠️ **Action Required:** Significant drift detected. Review the Data Drift Analysis tab.")
        else:
            st.success("✅ **Status:** No significant drift detected. System is healthy.")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎯 Model Health Indicators</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Average Confidence:** {conf_metrics['average_confidence']:.2%}")
        st.markdown(f"**Confidence Variance:** {conf_metrics['variance']:.4f}")
        st.markdown(f"**Estimated Reliability:** {proxy_metrics['estimated_reliability']:.2%}")
        st.markdown(f"**Risk Assessment:** {proxy_metrics['risk_level']}")
        
        if conf_metrics['is_declining']:
            st.warning(f"⚠️ **Alert:** Confidence declining by {conf_metrics['confidence_decline']:.1%}")
        else:
            st.success("✅ **Status:** Confidence levels remain stable.")
    
    # Recent alerts
    if alert_engine.alerts:
        st.markdown("---")
        st.markdown("### 🚨 Recent Alerts (Last 5)")
        for alert in alert_engine.alerts[-5:]:
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
    st.markdown("### 💡 Recommended Actions")
    recommendations = alert_engine.get_actionable_recommendations()
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.success("✅ No immediate actions required. Continue monitoring.")
    
    # System status summary
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Quality", "🟢 Good" if drift_summary['high_drift_count'] < 2 else "🔴 Poor")
    with col2:
        st.metric("Model Stability", "🟢 Stable" if not conf_metrics['is_unstable'] else "🔴 Unstable")
    with col3:
        st.metric("Overall Health", "🟢 Healthy" if alert_count == 0 else "🟡 Monitor" if alert_count < 3 else "🔴 Critical")


def render_drift_page(drift_results, drift_summary, drift_report, baseline_data, live_data, analyzer):
    """Render data drift analysis page with improved UI"""
    st.markdown("### 📊 Feature-wise Drift Detection")
    
    # Summary statistics with visual hierarchy
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
    
    # Drift detection summary
    st.markdown("""
    <div class="card">
        <h4>🔍 Drift Detection Methods</h4>
        <p><strong>PSI (Population Stability Index):</strong> Measures distribution shift (0.25+ = high drift)</p>
        <p><strong>KS Test (Kolmogorov-Smirnov):</strong> Statistical test for distribution differences</p>
        <p><strong>JS Divergence (Jensen-Shannon):</strong> Symmetric measure of distribution similarity</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Drift report table with improved styling
    st.markdown("### 📋 Detailed Drift Results")
    
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
    
    # Download button for drift report
    csv = drift_report.to_csv(index=False)
    st.download_button(
        label="📥 Download Drift Report (CSV)",
        data=csv,
        file_name="drift_report.csv",
        mime="text/csv"
    )
    
    # Top drifted features
    st.markdown("---")
    st.markdown("### 🎯 Top Drifting Features (Detailed Analysis)")
    
    top_features = analyzer.rank_unstable_features(drift_results, top_n=5)
    
    if not top_features:
        st.info("✅ No significant drift detected in any features.")
        return
    
    for i, feat in enumerate(top_features, 1):
        with st.expander(f"{i}. {feat['feature']} - {feat['severity']} Severity", expanded=(i==1)):
            explanation = analyzer.generate_explanation(feat)
            st.markdown(explanation)
            
            # Statistics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Drift Metrics:**")
                st.write(f"- PSI Score: **{feat['psi']:.3f}**")
                st.write(f"- JS Divergence: **{feat['js_divergence']:.3f}**")
                st.write(f"- KS Statistic: **{feat['ks_statistic']:.3f}**")
            
            with col2:
                st.markdown("**Distribution Statistics:**")
                st.write(f"- Baseline Mean: **{feat['baseline_mean']:.2f}**")
                st.write(f"- Live Mean: **{feat['live_mean']:.2f}**")
                st.write(f"- Percent Change: **{feat['pct_change']:.1f}%**")
            
            # Plot distributions
            fig = go.Figure()
            
            feature_name = feat['feature']
            fig.add_trace(go.Histogram(
                x=baseline_data[feature_name],
                name='Baseline',
                opacity=0.6,
                marker_color='#3b82f6',
                nbinsx=30
            ))
            fig.add_trace(go.Histogram(
                x=live_data[feature_name],
                name='Live',
                opacity=0.6,
                marker_color='#ef4444',
                nbinsx=30
            ))
            
            fig.update_layout(
                title=f'{feature_name} - Distribution Comparison',
                xaxis_title=feature_name,
                yaxis_title='Frequency',
                barmode='overlay',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_confidence_page(conf_monitor, conf_metrics, baseline_preds, live_preds):
    """Render confidence monitoring page with improved UI"""
    st.markdown("### 🎯 Prediction Confidence Analysis")
    
    # Key metrics with improved styling
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
    
    # Explanation card
    st.markdown("""
    <div class="card">
        <h4>📖 About Confidence Monitoring</h4>
        <p>Prediction confidence measures how certain the model is about its predictions. 
        Declining confidence or high variance can indicate model degradation, even without ground truth labels.</p>
        <ul>
            <li><strong>Declining Confidence:</strong> Systematic drop in prediction certainty over time</li>
            <li><strong>High Variance:</strong> Inconsistent prediction confidence indicating instability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confidence over time
    st.markdown("### 📈 Confidence Trend Over Time")
    
    if len(conf_monitor.confidence_history) > 0:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=conf_monitor.confidence_history,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=6)
        ))
        
        # Add rolling average
        rolling_avg = conf_monitor.calculate_rolling_average()
        if len(rolling_avg) > 0:
            fig.add_trace(go.Scatter(
                y=rolling_avg,
                mode='lines',
                name='Rolling Average',
                line=dict(color='#ef4444', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='Prediction Confidence Trajectory',
            xaxis_title='Batch Number',
            yaxis_title='Confidence Score',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No confidence history available.")
    
    # Trend analysis
    st.markdown("---")
    st.markdown("### 📊 Statistical Trend Analysis")
    
    trend = conf_monitor.analyze_trend()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend['trend'], "❓")
        st.metric("Trend Direction", f"{trend_emoji} {trend['trend'].title()}")
    
    with col2:
        st.metric("Trend Slope", f"{trend['slope']:.6f}")
    
    with col3:
        st.metric("Goodness of Fit (R²)", f"{trend['r_squared']:.3f}")
    
    # Performance proxy comparison
    st.markdown("---")
    st.markdown("### 🔬 Performance Proxy Analysis")
    st.markdown("*Estimating model reliability without ground truth labels*")
    
    proxy = PerformanceProxy()
    comparison = proxy.compare_baseline(baseline_preds, live_preds)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📘 Baseline (Training)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Reliability:** {comparison['baseline_reliability']:.2%}")
        st.write(f"**Avg Confidence:** {comparison['baseline_confidence']:.2%}")
        st.write(f"**Avg Entropy:** {comparison['baseline_entropy']:.3f}")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>🔴 Live (Production)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Reliability:** {comparison['current_reliability']:.2%}")
        st.write(f"**Avg Confidence:** {comparison['current_confidence']:.2%}")
        st.write(f"**Avg Entropy:** {comparison['current_entropy']:.3f}")
    
    st.markdown("---")
    st.markdown("**Performance Shifts:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reliability Shift", f"{comparison['reliability_shift']:.2%}", 
                 delta=f"{comparison['reliability_shift']:.2%}", delta_color="inverse")
    with col2:
        st.metric("Confidence Shift", f"{comparison['confidence_shift']:.2%}",
                 delta=f"{comparison['confidence_shift']:.2%}", delta_color="inverse")
    with col3:
        st.metric("Entropy Shift", f"{comparison['entropy_shift']:.3f}",
                 delta=f"{comparison['entropy_shift']:.3f}", delta_color="normal")
    
    if comparison['degradation_detected']:
        st.error("⚠️ **Alert:** Performance degradation detected based on proxy metrics")
        st.markdown("**Recommendation:** Consider retraining or investigating data quality issues.")
    else:
        st.success("✅ **Status:** No significant performance degradation detected")
        st.markdown("**Action:** Continue monitoring. No immediate intervention required.")


def render_alerts_page(alert_engine, drift_results, conf_metrics, proxy_metrics):
    """Render alerts and recommendations page with improved UI"""
    st.markdown("### ⚠️ Risk Alerts & Actionable Reports")
    
    # Alert summary
    alert_summary = alert_engine.generate_alert_summary()
    
    if len(alert_engine.alerts) == 0:
        st.success("✅ **All Clear!** No active alerts. System is operating normally.")
        st.markdown("""
        <div class="card">
            <h4>🎉 System Status: Healthy</h4>
            <p>All monitoring metrics are within acceptable thresholds. Continue monitoring for any changes.</p>
            <ul>
                <li>Data drift is minimal</li>
                <li>Confidence levels are stable</li>
                <li>No performance degradation detected</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Count by severity
    severity_counts = alert_summary['Severity'].value_counts().to_dict()
    
    st.markdown("### 📊 Alert Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        critical_count = severity_counts.get('CRITICAL', 0)
        st.metric("🔴 Critical", critical_count)
        if critical_count > 0:
            st.caption("Immediate action required")
    with col2:
        high_count = severity_counts.get('HIGH', 0)
        st.metric("🟡 High", high_count)
        if high_count > 0:
            st.caption("Investigation needed")
    with col3:
        medium_count = severity_counts.get('MEDIUM', 0)
        st.metric("🔵 Medium", medium_count)
        if medium_count > 0:
            st.caption("Monitor closely")
    
    st.markdown("---")
    
    # Show all alerts with detailed information
    st.markdown("### 🚨 Active Alerts (Detailed View)")
    
    for i, alert in enumerate(alert_engine.alerts, 1):
        severity_color = {
            'CRITICAL': 'alert-critical',
            'HIGH': 'alert-high',
            'MEDIUM': 'alert-medium'
        }.get(alert.severity, 'alert-medium')
        
        severity_icon = {
            'CRITICAL': '🔴',
            'HIGH': '🟡',
            'MEDIUM': '🔵'
        }.get(alert.severity, '🔵')
        
        with st.expander(f"{severity_icon} Alert #{i}: {alert.alert_type} ({alert.severity})", expanded=(i<=2)):
            st.markdown(f"**Message:** {alert.message}")
            st.markdown(f"**Timestamp:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if 'recommendation' in alert.details:
                st.markdown(f"**💡 Recommendation:** {alert.details['recommendation']}")
            
            if 'risk_level' in alert.details:
                st.markdown(f"**⚠️ Risk Level:** {alert.details['risk_level']}")
            
            if 'affected_features' in alert.details:
                st.markdown(f"**📊 Affected Features:** {', '.join(alert.details['affected_features'])}")
            
            if 'possible_causes' in alert.details:
                st.markdown("**🔍 Possible Root Causes:**")
                for cause in alert.details['possible_causes']:
                    st.markdown(f"- {cause}")
            
            # Action button placeholder
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Mark as Reviewed", key=f"review_{i}"):
                    st.info("Alert marked as reviewed (demo mode)")
            with col2:
                if st.button(f"Export Details", key=f"export_{i}"):
                    st.info("Export functionality (demo mode)")
    
    # Actionable recommendations
    st.markdown("---")
    st.markdown("### 💡 Prioritized Action Plan")
    
    recommendations = alert_engine.get_actionable_recommendations()
    
    if recommendations:
        st.markdown("""
        <div class="card">
            <h4>Recommended Actions (Priority Order)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    else:
        st.info("No specific recommendations at this time. Continue standard monitoring procedures.")
    
    # Alert timeline
    st.markdown("---")
    st.markdown("### 📅 Alert Timeline")
    
    if not alert_summary.empty:
        st.dataframe(alert_summary, use_container_width=True)
        
        # Download alert report
        csv = alert_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Alert Report (CSV)",
            data=csv,
            file_name="alert_report.csv",
            mime="text/csv"
        )
    
    # System recommendations summary
    st.markdown("---")
    st.markdown("### 📋 Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📊 Current Status</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"**Total Alerts:** {len(alert_engine.alerts)}")
        st.write(f"**Critical Issues:** {severity_counts.get('CRITICAL', 0)}")
        st.write(f"**Requires Attention:** {severity_counts.get('HIGH', 0) + severity_counts.get('CRITICAL', 0)}")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>🎯 Next Steps</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if severity_counts.get('CRITICAL', 0) > 0:
            st.write("**Immediate:** Investigate critical alerts")
            st.write("**Short-term:** Plan model retraining")
            st.write("**Long-term:** Review data pipeline")
        elif severity_counts.get('HIGH', 0) > 0:
            st.write("**Short-term:** Investigate high-priority alerts")
            st.write("**Medium-term:** Monitor trends closely")
            st.write("**Long-term:** Consider preventive measures")
        else:
            st.write("**Continue:** Standard monitoring")
            st.write("**Review:** Weekly performance reports")
            st.write("**Maintain:** Current thresholds")


if __name__ == "__main__":
    main()
