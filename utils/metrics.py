"""
Statistical Metrics for Data Drift Detection
Production-grade statistical tests for ML monitoring
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Tuple, Optional


def kolmogorov_smirnov_test(baseline: np.ndarray, live: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for distribution comparison
    
    Args:
        baseline: Reference distribution
        live: Current distribution
        
    Returns:
        Tuple of (statistic, p-value)
    """
    # Remove NaN values
    baseline_clean = baseline[~np.isnan(baseline)]
    live_clean = live[~np.isnan(live)]
    
    if len(baseline_clean) == 0 or len(live_clean) == 0:
        return 0.0, 1.0
    
    statistic, pvalue = stats.ks_2samp(baseline_clean, live_clean)
    return statistic, pvalue


def population_stability_index(baseline: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    PSI is widely used in credit scoring and risk modeling
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change
    PSI >= 0.25: Significant change
    
    Args:
        baseline: Reference distribution
        live: Current distribution
        bins: Number of bins for histogram
        
    Returns:
        PSI score
    """
    # Remove NaN values
    baseline_clean = baseline[~np.isnan(baseline)]
    live_clean = live[~np.isnan(live)]
    
    if len(baseline_clean) == 0 or len(live_clean) == 0:
        return 0.0
    
    # Create bins based on baseline distribution
    min_val = min(baseline_clean.min(), live_clean.min())
    max_val = max(baseline_clean.max(), live_clean.max())
    
    # Handle edge case where min == max
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    baseline_hist, _ = np.histogram(baseline_clean, bins=bin_edges)
    live_hist, _ = np.histogram(live_clean, bins=bin_edges)
    
    # Normalize to get proportions
    baseline_prop = baseline_hist / len(baseline_clean)
    live_prop = live_hist / len(live_clean)
    
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    baseline_prop = np.where(baseline_prop == 0, epsilon, baseline_prop)
    live_prop = np.where(live_prop == 0, epsilon, live_prop)
    
    # Calculate PSI
    psi = np.sum((live_prop - baseline_prop) * np.log(live_prop / baseline_prop))
    
    return psi


def jensen_shannon_divergence(baseline: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions
    
    JS divergence is a symmetric and smoothed version of KL divergence
    Range: [0, 1], where 0 means identical distributions
    
    Args:
        baseline: Reference distribution
        live: Current distribution
        bins: Number of bins for histogram
        
    Returns:
        JS divergence score
    """
    # Remove NaN values
    baseline_clean = baseline[~np.isnan(baseline)]
    live_clean = live[~np.isnan(live)]
    
    if len(baseline_clean) == 0 or len(live_clean) == 0:
        return 0.0
    
    # Create bins
    min_val = min(baseline_clean.min(), live_clean.min())
    max_val = max(baseline_clean.max(), live_clean.max())
    
    # Handle edge case where min == max
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    baseline_hist, _ = np.histogram(baseline_clean, bins=bin_edges)
    live_hist, _ = np.histogram(live_clean, bins=bin_edges)
    
    # Normalize to get probability distributions (ensure they sum to 1)
    baseline_prob = baseline_hist / baseline_hist.sum() if baseline_hist.sum() > 0 else baseline_hist
    live_prob = live_hist / live_hist.sum() if live_hist.sum() > 0 else live_hist
    
    # Calculate JS divergence
    js_div = jensenshannon(baseline_prob, live_prob)
    
    return js_div


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of probability distribution
    
    Used for prediction confidence monitoring
    Higher entropy = more uncertainty
    
    Args:
        probabilities: Array of probabilities
        
    Returns:
        Entropy value
    """
    # Remove zero probabilities to avoid log(0)
    probs_clean = probabilities[probabilities > 0]
    
    if len(probs_clean) == 0:
        return 0.0
    
    entropy = -np.sum(probs_clean * np.log2(probs_clean))
    return entropy


def calculate_prediction_confidence(probabilities: np.ndarray) -> float:
    """
    Calculate average prediction confidence
    
    For binary classification: max probability
    For multi-class: max probability
    
    Args:
        probabilities: Array of predicted probabilities (can be 1D for binary or 2D for multi-class)
        
    Returns:
        Average confidence score
    """
    if len(probabilities) == 0:
        return 0.0
    
    # If 2D array (multi-class), take max probability per sample
    if probabilities.ndim == 2:
        max_probs = np.max(probabilities, axis=1)
    else:
        max_probs = probabilities
    
    return np.mean(max_probs)


def detect_variance_spike(values: np.ndarray, window_size: int = 50, threshold: float = 2.0) -> bool:
    """
    Detect variance spikes in a time series
    
    Used for detecting instability in predictions
    
    Args:
        values: Time series values
        window_size: Rolling window size
        threshold: Multiplier for variance spike detection
        
    Returns:
        True if variance spike detected
    """
    if len(values) < window_size * 2:
        return False
    
    # Calculate rolling variance
    rolling_var = pd.Series(values).rolling(window=window_size).var()
    
    # Get recent variance vs historical baseline
    baseline_var = rolling_var.iloc[:-window_size].mean()
    recent_var = rolling_var.iloc[-window_size:].mean()
    
    if baseline_var == 0 or np.isnan(baseline_var):
        return False
    
    # Spike if recent variance is much higher than baseline
    return recent_var > threshold * baseline_var


def classify_drift_severity(psi: float, js_div: float, ks_stat: float) -> str:
    """
    Classify drift severity based on multiple metrics
    
    Production rule: Use conservative thresholds to avoid false alarms
    
    Args:
        psi: Population Stability Index
        js_div: Jensen-Shannon divergence
        ks_stat: Kolmogorov-Smirnov statistic
        
    Returns:
        Drift severity: "Low", "Medium", or "High"
    """
    # PSI thresholds (industry standard)
    psi_high = 0.25
    psi_medium = 0.1
    
    # JS divergence thresholds
    js_high = 0.5
    js_medium = 0.2
    
    # KS statistic thresholds
    ks_high = 0.3
    ks_medium = 0.15
    
    # Count how many metrics indicate high/medium drift
    high_count = 0
    medium_count = 0
    
    if psi >= psi_high:
        high_count += 1
    elif psi >= psi_medium:
        medium_count += 1
    
    if js_div >= js_high:
        high_count += 1
    elif js_div >= js_medium:
        medium_count += 1
    
    if ks_stat >= ks_high:
        high_count += 1
    elif ks_stat >= ks_medium:
        medium_count += 1
    
    # Classification logic: conservative approach
    if high_count >= 2:
        return "High"
    elif high_count >= 1 or medium_count >= 2:
        return "Medium"
    else:
        return "Low"
