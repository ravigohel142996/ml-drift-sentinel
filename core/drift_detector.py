"""
Data & Feature Drift Detection
Statistical drift detection for production ML systems
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    kolmogorov_smirnov_test,
    population_stability_index,
    jensen_shannon_divergence,
    classify_drift_severity
)


class DriftDetector:
    """
    Production-grade drift detector for ML monitoring
    
    Implements multiple statistical tests:
    - Kolmogorov-Smirnov test: Non-parametric distribution comparison
    - Population Stability Index (PSI): Industry standard from credit risk
    - Jensen-Shannon divergence: Symmetric KL divergence
    
    Design principles:
    - Multiple metrics for robustness
    - Conservative thresholds to reduce false positives
    - Detailed reporting for explainability
    """
    
    def __init__(self, baseline_data: pd.DataFrame):
        """
        Initialize drift detector with baseline (reference) data
        
        Args:
            baseline_data: Training or reference distribution data
        """
        self.baseline_data = baseline_data
        self.numeric_features = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_drift(self, live_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift between baseline and live data
        
        Args:
            live_data: Production or current data
            
        Returns:
            Dictionary with drift metrics for each feature
        """
        drift_results = {}
        
        for feature in self.numeric_features:
            if feature not in live_data.columns:
                continue
            
            baseline_values = self.baseline_data[feature].values
            live_values = live_data[feature].values
            
            # Calculate all three drift metrics
            ks_stat, ks_pvalue = kolmogorov_smirnov_test(baseline_values, live_values)
            psi = population_stability_index(baseline_values, live_values)
            js_div = jensen_shannon_divergence(baseline_values, live_values)
            
            # Classify drift severity
            severity = classify_drift_severity(psi, js_div, ks_stat)
            
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi': psi,
                'js_divergence': js_div,
                'severity': severity,
                'baseline_mean': np.nanmean(baseline_values),
                'live_mean': np.nanmean(live_values),
                'baseline_std': np.nanstd(baseline_values),
                'live_std': np.nanstd(live_values)
            }
        
        return drift_results
    
    def get_top_drifted_features(
        self,
        drift_results: Dict[str, Dict],
        top_n: int = 10
    ) -> List[Tuple[str, float, str]]:
        """
        Get top N features with highest drift
        
        Args:
            drift_results: Output from detect_drift()
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, combined_drift_score, severity) tuples
        """
        feature_scores = []
        
        for feature, metrics in drift_results.items():
            # Combined drift score: weighted average of normalized metrics
            # PSI and JS divergence are more reliable than KS statistic alone
            combined_score = (
                0.4 * metrics['psi'] +
                0.4 * metrics['js_divergence'] +
                0.2 * metrics['ks_statistic']
            )
            
            feature_scores.append((
                feature,
                combined_score,
                metrics['severity']
            ))
        
        # Sort by combined score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:top_n]
    
    def generate_drift_report(self, drift_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Generate detailed drift report as DataFrame
        
        Args:
            drift_results: Output from detect_drift()
            
        Returns:
            DataFrame with drift report
        """
        report_data = []
        
        for feature, metrics in drift_results.items():
            report_data.append({
                'Feature': feature,
                'Severity': metrics['severity'],
                'PSI': round(metrics['psi'], 4),
                'JS_Divergence': round(metrics['js_divergence'], 4),
                'KS_Statistic': round(metrics['ks_statistic'], 4),
                'KS_PValue': round(metrics['ks_pvalue'], 4),
                'Baseline_Mean': round(metrics['baseline_mean'], 4),
                'Live_Mean': round(metrics['live_mean'], 4),
                'Mean_Shift_%': round(
                    ((metrics['live_mean'] - metrics['baseline_mean']) / 
                     (abs(metrics['baseline_mean']) + 1e-10) * 100), 2
                )
            })
        
        df = pd.DataFrame(report_data)
        
        # Sort by severity and PSI
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        df['_severity_order'] = df['Severity'].map(severity_order)
        df = df.sort_values(['_severity_order', 'PSI'], ascending=[True, False])
        df = df.drop('_severity_order', axis=1)
        
        return df
    
    def get_drift_summary(self, drift_results: Dict[str, Dict]) -> Dict:
        """
        Get high-level drift summary statistics
        
        Args:
            drift_results: Output from detect_drift()
            
        Returns:
            Dictionary with summary statistics
        """
        total_features = len(drift_results)
        
        severity_counts = {
            'High': 0,
            'Medium': 0,
            'Low': 0
        }
        
        for metrics in drift_results.values():
            severity = metrics['severity']
            severity_counts[severity] += 1
        
        # Calculate average drift metrics
        avg_psi = np.mean([m['psi'] for m in drift_results.values()])
        avg_js = np.mean([m['js_divergence'] for m in drift_results.values()])
        avg_ks = np.mean([m['ks_statistic'] for m in drift_results.values()])
        
        return {
            'total_features': total_features,
            'high_drift_count': severity_counts['High'],
            'medium_drift_count': severity_counts['Medium'],
            'low_drift_count': severity_counts['Low'],
            'average_psi': avg_psi,
            'average_js_divergence': avg_js,
            'average_ks_statistic': avg_ks,
            'drift_detected': severity_counts['High'] > 0 or severity_counts['Medium'] > 0
        }
