"""
Alert Engine for ML Model Monitoring
Rule-based, interpretable, and auditable alerts
"""
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


class Alert:
    """Alert data structure"""
    
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Dict,
        timestamp: Optional[datetime] = None
    ):
        """
        Args:
            alert_type: Type of alert (RETRAIN, DATA_ANOMALY, CONFIDENCE_UNSTABLE, etc.)
            severity: LOW, MEDIUM, HIGH, CRITICAL
            message: Human-readable alert message
            details: Additional context and metrics
            timestamp: Alert timestamp
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class AlertEngine:
    """
    Production-grade alert engine for ML monitoring
    
    Design principles:
    - Rule-based: Clear thresholds, no black boxes
    - Interpretable: Every alert has clear reasoning
    - Auditable: All alerts logged with context
    - Actionable: Clear recommendations for each alert type
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        
    def clear_alerts(self):
        """Clear all stored alerts"""
        self.alerts = []
    
    def check_drift_alerts(
        self,
        drift_results: Dict[str, Dict],
        high_threshold: int = 3,
        medium_threshold: int = 5
    ) -> List[Alert]:
        """
        Generate alerts based on data drift detection results
        
        Args:
            drift_results: Dictionary with feature names and their drift metrics
            high_threshold: Number of high-severity drifts to trigger critical alert
            medium_threshold: Number of medium+ severity drifts to trigger warning
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Count drift by severity
        high_drift_features = []
        medium_drift_features = []
        
        for feature, metrics in drift_results.items():
            severity = metrics.get('severity', 'Low')
            if severity == "High":
                high_drift_features.append(feature)
            elif severity == "Medium":
                medium_drift_features.append(feature)
        
        # Critical alert: Multiple high-severity drifts
        if len(high_drift_features) >= high_threshold:
            alert = Alert(
                alert_type="RETRAIN_REQUIRED",
                severity="CRITICAL",
                message=f"Model retrain strongly recommended: {len(high_drift_features)} features showing high drift",
                details={
                    "affected_features": high_drift_features,
                    "recommendation": "Immediate model retraining required",
                    "risk_level": "High - Model predictions may be unreliable"
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Warning alert: Multiple medium+ severity drifts
        elif len(high_drift_features) + len(medium_drift_features) >= medium_threshold:
            alert = Alert(
                alert_type="RETRAIN_RECOMMENDED",
                severity="HIGH",
                message=f"Model retrain recommended: {len(high_drift_features) + len(medium_drift_features)} features showing drift",
                details={
                    "high_drift_features": high_drift_features,
                    "medium_drift_features": medium_drift_features,
                    "recommendation": "Plan model retraining within next deployment cycle",
                    "risk_level": "Medium - Model performance may degrade"
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Data pipeline alert: Check for specific high-drift features
        if high_drift_features:
            alert = Alert(
                alert_type="DATA_PIPELINE_ANOMALY",
                severity="HIGH",
                message=f"Data pipeline anomaly detected in {len(high_drift_features)} features",
                details={
                    "affected_features": high_drift_features,
                    "recommendation": "Investigate upstream data pipeline for issues",
                    "possible_causes": [
                        "Data collection process changed",
                        "Feature engineering pipeline modified",
                        "Upstream data source quality issues",
                        "Population shift in production environment"
                    ]
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def check_confidence_alerts(
        self,
        confidence_metrics: Dict,
        decline_threshold: float = 0.1,
        instability_threshold: float = 0.15
    ) -> List[Alert]:
        """
        Generate alerts based on prediction confidence monitoring
        
        Args:
            confidence_metrics: Dictionary with confidence monitoring results
            decline_threshold: Threshold for confidence decline (10% drop)
            instability_threshold: Threshold for confidence variance
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        avg_confidence = confidence_metrics.get('average_confidence', 1.0)
        confidence_decline = confidence_metrics.get('confidence_decline', 0.0)
        is_unstable = confidence_metrics.get('is_unstable', False)
        variance = confidence_metrics.get('variance', 0.0)
        
        # Confidence collapse alert
        if confidence_decline > decline_threshold:
            alert = Alert(
                alert_type="CONFIDENCE_COLLAPSE",
                severity="CRITICAL",
                message=f"Model confidence dropped by {confidence_decline*100:.1f}%",
                details={
                    "current_confidence": avg_confidence,
                    "decline_percentage": confidence_decline * 100,
                    "recommendation": "Urgent investigation required - model may be encountering OOD data",
                    "risk_level": "Critical - Model reliability compromised"
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Confidence instability alert
        if is_unstable:
            alert = Alert(
                alert_type="CONFIDENCE_UNSTABLE",
                severity="HIGH",
                message=f"Model confidence is unstable (variance: {variance:.4f})",
                details={
                    "variance": variance,
                    "recommendation": "Monitor closely - model predictions becoming inconsistent",
                    "possible_causes": [
                        "Encountering edge cases",
                        "Data quality issues",
                        "Model approaching decision boundaries",
                        "Concept drift in progress"
                    ]
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def check_performance_proxy_alerts(
        self,
        proxy_metrics: Dict,
        reliability_threshold: float = 0.7
    ) -> List[Alert]:
        """
        Generate alerts based on performance proxy metrics
        
        Args:
            proxy_metrics: Dictionary with proxy metrics (without ground truth)
            reliability_threshold: Threshold for estimated reliability
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        estimated_reliability = proxy_metrics.get('estimated_reliability', 1.0)
        high_entropy_ratio = proxy_metrics.get('high_entropy_ratio', 0.0)
        
        # Low reliability alert
        if estimated_reliability < reliability_threshold:
            alert = Alert(
                alert_type="LOW_RELIABILITY_ESTIMATE",
                severity="HIGH",
                message=f"Estimated model reliability below threshold: {estimated_reliability:.2f}",
                details={
                    "estimated_reliability": estimated_reliability,
                    "threshold": reliability_threshold,
                    "high_entropy_ratio": high_entropy_ratio,
                    "recommendation": "Request ground truth labels for accuracy validation",
                    "note": "This is an estimate based on prediction patterns - not actual accuracy"
                }
            )
            alerts.append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def generate_alert_summary(self) -> pd.DataFrame:
        """
        Generate summary of all alerts for reporting
        
        Returns:
            DataFrame with alert summary
        """
        if not self.alerts:
            return pd.DataFrame(columns=['Timestamp', 'Type', 'Severity', 'Message'])
        
        alert_data = []
        for alert in self.alerts:
            alert_data.append({
                'Timestamp': alert.timestamp,
                'Type': alert.alert_type,
                'Severity': alert.severity,
                'Message': alert.message
            })
        
        return pd.DataFrame(alert_data)
    
    def get_actionable_recommendations(self) -> List[str]:
        """
        Get list of actionable recommendations based on current alerts
        
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        seen = set()
        
        # Priority order for recommendations - maintain order and avoid duplicates
        for alert in self.alerts:
            rec_text = alert.details.get('recommendation', '')
            if not rec_text or rec_text in seen:
                continue
            
            seen.add(rec_text)
            if alert.severity == "CRITICAL":
                recommendations.append((0, f"🔴 CRITICAL: {rec_text}"))
            elif alert.severity == "HIGH":
                recommendations.append((1, f"🟡 HIGH: {rec_text}"))
        
        # Sort by priority (0=CRITICAL, 1=HIGH) and return just the messages
        recommendations.sort(key=lambda x: x[0])
        return [rec[1] for rec in recommendations]
