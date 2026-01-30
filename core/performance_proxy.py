"""
Performance Proxy - Estimate model reliability without labels
Production technique for monitoring when ground truth is unavailable
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import calculate_entropy, detect_variance_spike


class PerformanceProxy:
    """
    Estimate model performance without ground truth labels
    
    Techniques used:
    - Prediction entropy analysis (high entropy = high uncertainty)
    - Confidence distribution analysis
    - Temporal variance detection
    - Distribution shift proxies
    
    Important: These are PROXIES, not true accuracy
    Use when labels are delayed or unavailable
    Always validate with ground truth when possible
    """
    
    def __init__(self):
        """Initialize performance proxy"""
        self.prediction_history: list = []
        
    def add_predictions(self, probabilities: np.ndarray):
        """
        Add new batch of predictions
        
        Args:
            probabilities: Predicted probabilities (1D for binary, 2D for multi-class)
        """
        self.prediction_history.append(probabilities)
    
    def calculate_entropy_metrics(self, probabilities: np.ndarray) -> Dict:
        """
        Calculate entropy-based uncertainty metrics
        
        High entropy = high uncertainty = potential accuracy issues
        
        Args:
            probabilities: Predicted probabilities
            
        Returns:
            Dictionary with entropy metrics
        """
        if len(probabilities) == 0:
            return {
                'average_entropy': 0.0,
                'max_entropy': 0.0,
                'high_entropy_ratio': 0.0
            }
        
        # Handle both binary and multi-class
        if probabilities.ndim == 1:
            # Binary classification: convert to 2D
            prob_2d = np.column_stack([probabilities, 1 - probabilities])
        else:
            prob_2d = probabilities
        
        # Calculate entropy for each prediction
        entropies = []
        for prob_dist in prob_2d:
            entropy = calculate_entropy(prob_dist)
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        
        # Maximum possible entropy (log2 of number of classes)
        max_possible_entropy = np.log2(prob_2d.shape[1])
        
        # Threshold for "high entropy" (>70% of max)
        high_entropy_threshold = 0.7 * max_possible_entropy
        high_entropy_ratio = np.mean(entropies > high_entropy_threshold)
        
        return {
            'average_entropy': np.mean(entropies),
            'max_entropy': np.max(entropies),
            'high_entropy_ratio': high_entropy_ratio,
            'max_possible_entropy': max_possible_entropy
        }
    
    def estimate_reliability(self, probabilities: np.ndarray) -> float:
        """
        Estimate model reliability score (0-1)
        
        Based on:
        - Average prediction confidence
        - Entropy distribution
        - Confidence variance
        
        Args:
            probabilities: Predicted probabilities
            
        Returns:
            Reliability score (0-1, higher is better)
        """
        if len(probabilities) == 0:
            return 0.0
        
        # Get max confidence per prediction
        if probabilities.ndim == 2:
            max_probs = np.max(probabilities, axis=1)
        else:
            max_probs = probabilities
        
        # Average confidence (higher is better)
        avg_confidence = np.mean(max_probs)
        
        # Confidence consistency (lower variance is better)
        conf_variance = np.var(max_probs)
        consistency_score = 1.0 / (1.0 + conf_variance)
        
        # Entropy metrics (lower entropy is better)
        entropy_metrics = self.calculate_entropy_metrics(probabilities)
        entropy_score = 1.0 - (entropy_metrics['average_entropy'] / 
                              (entropy_metrics['max_possible_entropy'] + 1e-10))
        
        # Combined reliability score (weighted average)
        reliability = (
            0.5 * avg_confidence +
            0.3 * consistency_score +
            0.2 * entropy_score
        )
        
        return reliability
    
    def detect_accuracy_degradation_risk(
        self,
        probabilities: np.ndarray,
        confidence_threshold: float = 0.7,
        entropy_threshold: float = 0.5
    ) -> Dict:
        """
        Assess risk of accuracy degradation
        
        Args:
            probabilities: Predicted probabilities
            confidence_threshold: Threshold for low confidence warning
            entropy_threshold: Threshold for high entropy warning
            
        Returns:
            Dictionary with risk assessment
        """
        if len(probabilities) == 0:
            return {
                'risk_level': 'unknown',
                'risk_score': 0.0,
                'factors': []
            }
        
        risk_factors = []
        risk_score = 0.0
        
        # Check confidence
        if probabilities.ndim == 2:
            max_probs = np.max(probabilities, axis=1)
        else:
            max_probs = probabilities
        
        avg_confidence = np.mean(max_probs)
        low_conf_ratio = np.mean(max_probs < confidence_threshold)
        
        if avg_confidence < confidence_threshold:
            risk_score += 0.3
            risk_factors.append(f"Low average confidence: {avg_confidence:.3f}")
        
        if low_conf_ratio > 0.3:
            risk_score += 0.2
            risk_factors.append(f"High ratio of low-confidence predictions: {low_conf_ratio:.1%}")
        
        # Check entropy
        entropy_metrics = self.calculate_entropy_metrics(probabilities)
        normalized_entropy = (entropy_metrics['average_entropy'] / 
                            (entropy_metrics['max_possible_entropy'] + 1e-10))
        
        if normalized_entropy > entropy_threshold:
            risk_score += 0.3
            risk_factors.append(f"High prediction uncertainty (entropy): {normalized_entropy:.3f}")
        
        if entropy_metrics['high_entropy_ratio'] > 0.2:
            risk_score += 0.2
            risk_factors.append(f"Many uncertain predictions: {entropy_metrics['high_entropy_ratio']:.1%}")
        
        # Classify risk level
        if risk_score >= 0.6:
            risk_level = 'HIGH'
        elif risk_score >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': risk_factors,
            'avg_confidence': avg_confidence,
            'avg_entropy': entropy_metrics['average_entropy']
        }
    
    def get_proxy_metrics(self, probabilities: np.ndarray) -> Dict:
        """
        Get comprehensive proxy metrics
        
        Args:
            probabilities: Predicted probabilities
            
        Returns:
            Dictionary with all proxy metrics
        """
        if len(probabilities) == 0:
            return {
                'estimated_reliability': 0.0,
                'average_confidence': 0.0,
                'average_entropy': 0.0,
                'high_entropy_ratio': 0.0,
                'risk_level': 'unknown'
            }
        
        # Calculate all metrics
        reliability = self.estimate_reliability(probabilities)
        entropy_metrics = self.calculate_entropy_metrics(probabilities)
        risk_assessment = self.detect_accuracy_degradation_risk(probabilities)
        
        # Get confidence stats
        if probabilities.ndim == 2:
            max_probs = np.max(probabilities, axis=1)
        else:
            max_probs = probabilities
        
        return {
            'estimated_reliability': reliability,
            'average_confidence': np.mean(max_probs),
            'confidence_std': np.std(max_probs),
            'average_entropy': entropy_metrics['average_entropy'],
            'high_entropy_ratio': entropy_metrics['high_entropy_ratio'],
            'risk_level': risk_assessment['risk_level'],
            'risk_score': risk_assessment['risk_score'],
            'risk_factors': risk_assessment['factors']
        }
    
    def compare_baseline(
        self,
        baseline_probabilities: np.ndarray,
        current_probabilities: np.ndarray
    ) -> Dict:
        """
        Compare current predictions to baseline
        
        Useful for detecting performance drift over time
        
        Args:
            baseline_probabilities: Reference predictions (e.g., validation set)
            current_probabilities: Current predictions
            
        Returns:
            Dictionary with comparison metrics
        """
        baseline_metrics = self.get_proxy_metrics(baseline_probabilities)
        current_metrics = self.get_proxy_metrics(current_probabilities)
        
        # Calculate shifts
        reliability_shift = (baseline_metrics['estimated_reliability'] - 
                           current_metrics['estimated_reliability'])
        confidence_shift = (baseline_metrics['average_confidence'] - 
                          current_metrics['average_confidence'])
        entropy_shift = (current_metrics['average_entropy'] - 
                        baseline_metrics['average_entropy'])
        
        return {
            'baseline_reliability': baseline_metrics['estimated_reliability'],
            'current_reliability': current_metrics['estimated_reliability'],
            'reliability_shift': reliability_shift,
            'baseline_confidence': baseline_metrics['average_confidence'],
            'current_confidence': current_metrics['average_confidence'],
            'confidence_shift': confidence_shift,
            'baseline_entropy': baseline_metrics['average_entropy'],
            'current_entropy': current_metrics['average_entropy'],
            'entropy_shift': entropy_shift,
            'degradation_detected': reliability_shift > 0.1 or confidence_shift > 0.1
        }
