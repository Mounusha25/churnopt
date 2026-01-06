"""
Drift detection and monitoring.

Implements:
1. Data drift (PSI, KL divergence)
2. Concept drift (performance degradation)
3. Model health monitoring
4. Automated alerting
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from pathlib import Path

from ..utils import setup_logging, load_config, ensure_dir

logger = setup_logging(__name__)


class DriftDetector:
    """
    Monitor data and concept drift in production.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_drift_config = config['data_drift']
        self.concept_drift_config = config['concept_drift']
        self.logger = logger
    
    def compute_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change
        """
        # Create bins based on reference distribution
        breakpoints = np.linspace(
            reference.min(),
            reference.max(),
            bins + 1
        )
        
        # Bin both distributions
        ref_binned = pd.cut(reference, bins=breakpoints, include_lowest=True)
        cur_binned = pd.cut(current, bins=breakpoints, include_lowest=True)
        
        # Get proportions
        ref_props = ref_binned.value_counts(normalize=True, sort=False)
        cur_props = cur_binned.value_counts(normalize=True, sort=False)
        
        # Avoid division by zero
        ref_props = ref_props.replace(0, 0.0001)
        cur_props = cur_props.replace(0, 0.0001)
        
        # Compute PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return float(psi)
    
    def compute_kl_divergence(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Compute Kullback-Leibler divergence.
        
        KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        """
        # Create bins
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1
        )
        
        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        cur_hist, _ = np.histogram(current, bins=breakpoints)
        
        # Normalize
        ref_dist = (ref_hist + 1e-10) / (ref_hist.sum() + bins * 1e-10)
        cur_dist = (cur_hist + 1e-10) / (cur_hist.sum() + bins * 1e-10)
        
        # Compute KL divergence
        kl_div = np.sum(cur_dist * np.log(cur_dist / ref_dist))
        
        return float(kl_div)
    
    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect drift for all features.
        
        Returns:
            Dictionary of {feature: {metric: value}}
        """
        self.logger.info("Detecting feature drift...")
        
        drift_report = {}
        psi_threshold = self.data_drift_config['methods']['psi'].get('threshold', 0.2)
        
        for feature in feature_cols:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            # Skip non-numeric features for now
            if reference_df[feature].dtype not in ['int64', 'float64']:
                continue
            
            try:
                psi = self.compute_psi(reference_df[feature], current_df[feature])
                kl_div = self.compute_kl_divergence(reference_df[feature], current_df[feature])
                
                drift_report[feature] = {
                    'psi': psi,
                    'kl_divergence': kl_div,
                    'drifted': psi > psi_threshold,
                }
                
                if psi > psi_threshold:
                    self.logger.warning(f"  {feature}: PSI = {psi:.4f} (DRIFT DETECTED)")
                    
            except Exception as e:
                self.logger.error(f"  Error computing drift for {feature}: {e}")
        
        return drift_report
    
    def check_model_health(
        self,
        predictions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check model health metrics.
        
        Args:
            predictions_df: DataFrame with churn_probability column
            
        Returns:
            Dictionary of health metrics
        """
        self.logger.info("Checking model health...")
        
        probabilities = predictions_df['churn_probability']
        
        health_metrics = {
            'mean_probability': float(probabilities.mean()),
            'std_probability': float(probabilities.std()),
            'min_probability': float(probabilities.min()),
            'max_probability': float(probabilities.max()),
            'n_predictions': len(probabilities),
        }
        
        # Check for anomalies
        expected_mean = self.config['model_health']['score_distribution'].get('expected_mean', 0.15)
        mean_deviation_threshold = self.config['model_health']['score_distribution'].get('mean_deviation_threshold', 0.05)
        
        mean_deviation = abs(health_metrics['mean_probability'] - expected_mean)
        
        if mean_deviation > mean_deviation_threshold:
            self.logger.warning(
                f"Score distribution shift detected: "
                f"mean={health_metrics['mean_probability']:.4f}, "
                f"expected={expected_mean:.4f}"
            )
            health_metrics['score_distribution_anomaly'] = True
        else:
            health_metrics['score_distribution_anomaly'] = False
        
        return health_metrics
    
    def generate_report(
        self,
        drift_report: Dict[str, Dict[str, float]],
        health_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate monitoring report.
        
        Returns:
            Path to report file
        """
        self.logger.info("Generating monitoring report...")
        
        report_lines = [
            "="*60,
            "DRIFT DETECTION & MONITORING REPORT",
            "="*60,
            "",
            "MODEL HEALTH",
            "-"*60,
            f"Mean Probability: {health_metrics['mean_probability']:.4f}",
            f"Std Probability: {health_metrics['std_probability']:.4f}",
            f"N Predictions: {health_metrics['n_predictions']}",
            f"Anomaly Detected: {health_metrics.get('score_distribution_anomaly', False)}",
            "",
            "FEATURE DRIFT",
            "-"*60,
        ]
        
        # Sort features by PSI
        drifted_features = {
            k: v for k, v in drift_report.items()
            if v.get('drifted', False)
        }
        
        if drifted_features:
            report_lines.append(f"\n⚠️  DRIFT DETECTED IN {len(drifted_features)} FEATURES:\n")
            
            for feature, metrics in sorted(
                drifted_features.items(),
                key=lambda x: x[1]['psi'],
                reverse=True
            ):
                report_lines.append(
                    f"  {feature}: PSI={metrics['psi']:.4f}, KL={metrics['kl_divergence']:.4f}"
                )
        else:
            report_lines.append("\n✅ No significant drift detected")
        
        report_lines.extend([
            "",
            "="*60,
        ])
        
        # Save report
        output_dir = Path("outputs/monitoring_reports")
        ensure_dir(str(output_dir))
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"monitoring_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Report saved to {report_path}")
        
        return str(report_path)
    
    def should_retrain(
        self,
        drift_report: Dict[str, Dict[str, float]],
        health_metrics: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.
        
        Returns:
            (should_retrain, reason)
        """
        # Check if too many features have drifted
        drifted_features = [
            f for f, m in drift_report.items()
            if m.get('drifted', False)
        ]
        
        if len(drifted_features) > 5:
            return True, f"Significant drift in {len(drifted_features)} features"
        
        # Check model health anomaly
        if health_metrics.get('score_distribution_anomaly', False):
            return True, "Score distribution anomaly detected"
        
        return False, ""
    
    def run(
        self,
        reference_date: str,
        current_date: str
    ) -> Dict[str, Any]:
        """
        Execute drift detection pipeline.
        
        Args:
            reference_date: Reference date for baseline
            current_date: Current date for comparison
            
        Returns:
            Monitoring results
        """
        self.logger.info("="*60)
        self.logger.info("DRIFT DETECTION & MONITORING")
        self.logger.info("="*60)
        
        # Load data (simplified - in production, load from feature store)
        from ..feature_engineering import FeatureStore
        
        feature_store = FeatureStore(load_config('configs/features.yaml'))
        
        # Get reference and current data
        reference_df = feature_store.get_offline_training_data(
            end_date=reference_date
        ).tail(10000)  # Sample for speed
        
        current_df = feature_store.get_offline_training_data(
            start_date=current_date
        ).head(10000)  # Sample for speed
        
        # Get numeric feature columns
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['customer_id', 'churn_label_next_period']
        feature_cols = [c for c in numeric_cols if c not in exclude]
        
        # Detect drift
        drift_report = self.detect_feature_drift(reference_df, current_df, feature_cols)
        
        # Check model health (using current predictions)
        # In production, load actual predictions
        current_df['churn_probability'] = np.random.beta(2, 5, len(current_df))  # Simulate
        health_metrics = self.check_model_health(current_df)
        
        # Generate report
        report_path = self.generate_report(drift_report, health_metrics)
        
        # Check if retraining needed
        should_retrain, reason = self.should_retrain(drift_report, health_metrics)
        
        if should_retrain:
            self.logger.warning(f"⚠️  RETRAINING RECOMMENDED: {reason}")
        else:
            self.logger.info("✅ No retraining needed at this time")
        
        self.logger.info("="*60)
        
        return {
            'drift_report': drift_report,
            'health_metrics': health_metrics,
            'should_retrain': should_retrain,
            'retrain_reason': reason,
            'report_path': report_path,
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-date', required=True)
    parser.add_argument('--current-date', required=True)
    parser.add_argument('--config', default='configs/monitoring.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    detector = DriftDetector(config)
    detector.run(args.reference_date, args.current_date)


if __name__ == '__main__':
    main()
