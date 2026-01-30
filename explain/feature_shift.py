"""
Feature Shift Analysis and Explainability
Identify and explain why features are drifting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class FeatureShiftAnalyzer:
    """
    Analyze and explain feature drift
    
    Capabilities:
    - Rank features by drift magnitude
    - Identify shift direction (increase/decrease)
    - Provide human-readable explanations
    - Detect correlation between drifting features
    """
    
    def __init__(self):
        """Initialize feature shift analyzer"""
        pass
    
    def rank_unstable_features(
        self,
        drift_results: Dict[str, Dict],
        top_n: int = 10
    ) -> List[Dict]:
        """
        Rank top unstable features by drift severity
        
        Args:
            drift_results: Output from DriftDetector.detect_drift()
            top_n: Number of top features to return
            
        Returns:
            List of feature information dictionaries
        """
        feature_info = []
        
        for feature, metrics in drift_results.items():
            # Calculate combined drift score
            drift_score = (
                0.4 * metrics['psi'] +
                0.4 * metrics['js_divergence'] +
                0.2 * metrics['ks_statistic']
            )
            
            # Determine shift direction
            mean_shift = metrics['live_mean'] - metrics['baseline_mean']
            if abs(mean_shift) < 1e-6:
                shift_direction = 'stable'
            elif mean_shift > 0:
                shift_direction = 'increased'
            else:
                shift_direction = 'decreased'
            
            # Calculate percentage change
            baseline_mean = metrics['baseline_mean']
            if abs(baseline_mean) > 1e-6:
                pct_change = (mean_shift / abs(baseline_mean)) * 100
            else:
                pct_change = 0.0
            
            feature_info.append({
                'feature': feature,
                'drift_score': drift_score,
                'severity': metrics['severity'],
                'psi': metrics['psi'],
                'js_divergence': metrics['js_divergence'],
                'ks_statistic': metrics['ks_statistic'],
                'baseline_mean': metrics['baseline_mean'],
                'live_mean': metrics['live_mean'],
                'mean_shift': mean_shift,
                'pct_change': pct_change,
                'shift_direction': shift_direction,
                'baseline_std': metrics['baseline_std'],
                'live_std': metrics['live_std']
            })
        
        # Sort by drift score
        feature_info.sort(key=lambda x: x['drift_score'], reverse=True)
        
        return feature_info[:top_n]
    
    def generate_explanation(self, feature_info: Dict) -> str:
        """
        Generate human-readable explanation for a drifting feature
        
        Args:
            feature_info: Feature information from rank_unstable_features()
            
        Returns:
            Human-readable explanation string
        """
        feature = feature_info['feature']
        severity = feature_info['severity']
        psi = feature_info['psi']
        shift_direction = feature_info['shift_direction']
        pct_change = feature_info['pct_change']
        baseline_mean = feature_info['baseline_mean']
        live_mean = feature_info['live_mean']
        
        # Build explanation
        explanation = f"**{feature}** ({severity} severity drift)\n"
        
        # Describe the shift
        if shift_direction == 'increased':
            explanation += f"- Mean value increased by {abs(pct_change):.1f}% "
            explanation += f"(from {baseline_mean:.3f} to {live_mean:.3f})\n"
        elif shift_direction == 'decreased':
            explanation += f"- Mean value decreased by {abs(pct_change):.1f}% "
            explanation += f"(from {baseline_mean:.3f} to {live_mean:.3f})\n"
        else:
            explanation += f"- Mean value stable (from {baseline_mean:.3f} to {live_mean:.3f})\n"
        
        # Describe drift metrics
        explanation += f"- PSI: {psi:.3f} "
        if psi >= 0.25:
            explanation += "(significant population shift)"
        elif psi >= 0.1:
            explanation += "(moderate population shift)"
        else:
            explanation += "(minor population shift)"
        explanation += "\n"
        
        # Provide interpretation
        if severity == "High":
            explanation += "- **Action**: Investigate data pipeline and consider model retraining\n"
            explanation += "- **Possible causes**: Data collection changes, population shift, or upstream errors\n"
        elif severity == "Medium":
            explanation += "- **Action**: Monitor closely and plan retraining if drift continues\n"
            explanation += "- **Possible causes**: Natural distribution evolution or seasonal effects\n"
        else:
            explanation += "- **Action**: Continue monitoring\n"
        
        return explanation
    
    def identify_correlated_drift(
        self,
        baseline_data: pd.DataFrame,
        live_data: pd.DataFrame,
        drift_results: Dict[str, Dict],
        correlation_threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of features that are drifting together
        
        This can reveal systemic issues or related feature groups
        
        Args:
            baseline_data: Reference data
            live_data: Current data
            drift_results: Output from DriftDetector.detect_drift()
            correlation_threshold: Minimum correlation to report
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        # Get features with medium or high drift
        drifting_features = [
            f for f, m in drift_results.items()
            if m['severity'] in ['Medium', 'High']
        ]
        
        if len(drifting_features) < 2:
            return []
        
        # Calculate correlation matrix for drifting features
        baseline_subset = baseline_data[drifting_features]
        live_subset = live_data[drifting_features]
        
        baseline_corr = baseline_subset.corr()
        live_corr = live_subset.corr()
        
        # Find pairs with high correlation in both datasets
        correlated_pairs = []
        for i, feat1 in enumerate(drifting_features):
            for j in range(i + 1, len(drifting_features)):
                feat2 = drifting_features[j]
                
                baseline_corr_val = baseline_corr.loc[feat1, feat2]
                live_corr_val = live_corr.loc[feat1, feat2]
                
                # Check if highly correlated in both
                if (abs(baseline_corr_val) >= correlation_threshold and
                    abs(live_corr_val) >= correlation_threshold):
                    avg_corr = (baseline_corr_val + live_corr_val) / 2
                    correlated_pairs.append((feat1, feat2, avg_corr))
        
        return correlated_pairs
    
    def generate_drift_summary_report(
        self,
        drift_results: Dict[str, Dict],
        top_n: int = 5
    ) -> str:
        """
        Generate executive summary of drift analysis
        
        Args:
            drift_results: Output from DriftDetector.detect_drift()
            top_n: Number of top features to highlight
            
        Returns:
            Markdown-formatted summary report
        """
        # Get summary statistics
        total_features = len(drift_results)
        high_drift = sum(1 for m in drift_results.values() if m['severity'] == 'High')
        medium_drift = sum(1 for m in drift_results.values() if m['severity'] == 'Medium')
        low_drift = sum(1 for m in drift_results.values() if m['severity'] == 'Low')
        
        # Build report
        report = "# Data Drift Analysis Summary\n\n"
        report += f"**Total Features Analyzed**: {total_features}\n\n"
        report += f"- 🔴 High Severity Drift: {high_drift} features\n"
        report += f"- 🟡 Medium Severity Drift: {medium_drift} features\n"
        report += f"- 🟢 Low Severity Drift: {low_drift} features\n\n"
        
        # Overall assessment
        if high_drift >= 3:
            report += "## ⚠️ Critical Assessment\n"
            report += "Multiple features show high drift. **Immediate action required**.\n\n"
        elif high_drift > 0 or medium_drift >= 5:
            report += "## ⚠️ Warning Assessment\n"
            report += "Significant drift detected. **Review and plan retraining**.\n\n"
        else:
            report += "## ✅ Acceptable Assessment\n"
            report += "Drift levels are within acceptable ranges. Continue monitoring.\n\n"
        
        # Top drifting features
        top_features = self.rank_unstable_features(drift_results, top_n)
        
        if top_features:
            report += f"## Top {len(top_features)} Drifting Features\n\n"
            for i, feat in enumerate(top_features, 1):
                report += f"{i}. **{feat['feature']}** "
                report += f"({feat['severity']} severity, PSI: {feat['psi']:.3f})\n"
                report += f"   - Shift: {feat['shift_direction']} by {abs(feat['pct_change']):.1f}%\n"
        
        return report
    
    def create_drift_comparison_table(
        self,
        drift_results: Dict[str, Dict],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Create comparison table for top drifting features
        
        Args:
            drift_results: Output from DriftDetector.detect_drift()
            top_n: Number of features to include
            
        Returns:
            DataFrame with comparison data
        """
        top_features = self.rank_unstable_features(drift_results, top_n)
        
        table_data = []
        for feat in top_features:
            table_data.append({
                'Feature': feat['feature'],
                'Severity': feat['severity'],
                'PSI': round(feat['psi'], 3),
                'Baseline Mean': round(feat['baseline_mean'], 3),
                'Live Mean': round(feat['live_mean'], 3),
                'Change %': round(feat['pct_change'], 1),
                'Direction': feat['shift_direction']
            })
        
        return pd.DataFrame(table_data)
