"""
Prediction Confidence Monitoring
Track confidence decay and prediction stability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import calculate_prediction_confidence, detect_variance_spike


class ConfidenceMonitor:
    """
    Monitor prediction confidence over time
    
    Key capabilities:
    - Rolling average confidence tracking
    - Confidence decline detection
    - Instability detection via variance analysis
    - Trend analysis
    
    Use cases:
    - Detect when model encounters out-of-distribution data
    - Identify model uncertainty spikes
    - Monitor prediction quality without ground truth
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize confidence monitor
        
        Args:
            window_size: Rolling window for trend analysis
        """
        self.window_size = window_size
        self.confidence_history: List[float] = []
        
    def add_predictions(self, probabilities: np.ndarray):
        """
        Add new batch of predictions to monitor
        
        Args:
            probabilities: Predicted probabilities (1D or 2D array)
        """
        # Calculate confidence for this batch
        batch_confidence = calculate_prediction_confidence(probabilities)
        self.confidence_history.append(batch_confidence)
    
    def calculate_rolling_average(self, window_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate rolling average confidence
        
        Args:
            window_size: Window size (uses default if None)
            
        Returns:
            Array of rolling averages
        """
        if window_size is None:
            window_size = self.window_size
        
        if len(self.confidence_history) < window_size:
            return np.array([])
        
        series = pd.Series(self.confidence_history)
        rolling_avg = series.rolling(window=window_size).mean()
        
        return rolling_avg.values
    
    def detect_confidence_decline(
        self,
        threshold: float = 0.1,
        comparison_window: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Detect significant confidence decline
        
        Compares recent confidence to baseline confidence
        
        Args:
            threshold: Decline threshold (e.g., 0.1 = 10% decline)
            comparison_window: Window to compare (uses self.window_size if None)
            
        Returns:
            Tuple of (decline_detected, decline_amount)
        """
        if comparison_window is None:
            comparison_window = self.window_size
        
        if len(self.confidence_history) < comparison_window * 2:
            return False, 0.0
        
        # Compare recent window to baseline window
        baseline = np.mean(self.confidence_history[:comparison_window])
        recent = np.mean(self.confidence_history[-comparison_window:])
        
        decline = baseline - recent
        decline_detected = decline > threshold
        
        return decline_detected, decline
    
    def detect_instability(
        self,
        variance_threshold: float = 0.05,
        window_size: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Detect confidence instability via variance analysis
        
        Args:
            variance_threshold: Threshold for variance
            window_size: Analysis window (uses self.window_size if None)
            
        Returns:
            Tuple of (unstable, variance_value)
        """
        if window_size is None:
            window_size = self.window_size
        
        if len(self.confidence_history) < window_size:
            return False, 0.0
        
        recent_window = self.confidence_history[-window_size:]
        variance = np.var(recent_window)
        
        unstable = variance > variance_threshold
        
        return unstable, variance
    
    def get_confidence_metrics(self) -> Dict:
        """
        Get comprehensive confidence metrics
        
        Returns:
            Dictionary with all confidence metrics
        """
        if len(self.confidence_history) == 0:
            return {
                'average_confidence': 0.0,
                'current_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'variance': 0.0,
                'confidence_decline': 0.0,
                'is_declining': False,
                'is_unstable': False,
                'num_observations': 0
            }
        
        # Basic statistics
        avg_confidence = np.mean(self.confidence_history)
        current_confidence = self.confidence_history[-1]
        min_confidence = np.min(self.confidence_history)
        max_confidence = np.max(self.confidence_history)
        variance = np.var(self.confidence_history)
        
        # Decline detection
        is_declining, decline_amount = self.detect_confidence_decline()
        
        # Instability detection
        is_unstable, _ = self.detect_instability()
        
        return {
            'average_confidence': avg_confidence,
            'current_confidence': current_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'variance': variance,
            'confidence_decline': decline_amount,
            'is_declining': is_declining,
            'is_unstable': is_unstable,
            'num_observations': len(self.confidence_history)
        }
    
    def analyze_trend(self) -> Dict:
        """
        Analyze confidence trend over time
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.confidence_history) < 10:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'r_squared': 0.0
            }
        
        # Fit linear trend
        x = np.arange(len(self.confidence_history))
        y = np.array(self.confidence_history)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Classify trend
        if abs(slope) < 0.0001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared
        }
    
    def get_confidence_percentiles(self) -> Dict:
        """
        Calculate confidence percentiles for distribution analysis
        
        Returns:
            Dictionary with percentile values
        """
        if len(self.confidence_history) == 0:
            return {}
        
        percentiles = [5, 25, 50, 75, 95]
        values = np.percentile(self.confidence_history, percentiles)
        
        return {
            f'p{p}': v for p, v in zip(percentiles, values)
        }
    
    def reset(self):
        """Reset confidence history"""
        self.confidence_history = []
