"""
Batch inference with SHAP explanations.
"""

import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yaml

from ..utils import setup_logging, ensure_dir
from ..models.registry import ModelRegistry
from ..feature_engineering import FeatureStore

logger = setup_logging(__name__)


class BatchInference:
    """Batch scoring for customer churn prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry()
        self.logger = logger
        
    def load_model(self, model_version: Optional[str] = None):
        """Load model from registry."""
        if model_version:
            model_entry = self.registry.get_model(model_version)
        else:
            model_entry = self.registry.get_production_model()
        
        if not model_entry:
            raise ValueError("No model found")
        
        model = joblib.load(model_entry['model_path'])
        return model, model_entry
    
    def predict(
        self,
        df: pd.DataFrame,
        model: Any
    ) -> pd.DataFrame:
        """Generate predictions."""
        self.logger.info("Generating predictions...")
        
        # Prepare features (same as training)
        exclude_cols = ['customer_id', 'snapshot_date', 'churn_label_next_period', 'months_until_churn', 'year_month', 'tenure_bucket']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        X = X.fillna(0)
        
        # Predict
        probabilities = model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': df['customer_id'],
            'snapshot_date': df['snapshot_date'] if 'snapshot_date' in df.columns else datetime.now(),
            'churn_probability': probabilities,
        })
        
        # Risk segments
        results['risk_segment'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return results
    
    def run(
        self,
        snapshot_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """Execute batch inference."""
        self.logger.info("="*60)
        self.logger.info("BATCH INFERENCE")
        self.logger.info("="*60)
        
        # Load model
        model, model_entry = self.load_model(model_version)
        self.logger.info(f"Using model: {model_entry['model_version']}")
        
        # Load features config for FeatureStore
        features_config_path = Path("configs/features.yaml")
        with open(features_config_path, 'r') as f:
            features_config = yaml.safe_load(f)
        
        # Load features
        feature_store = FeatureStore(features_config)
        df = feature_store.get_offline_training_data()
        
        # Filter by snapshot date if provided
        if snapshot_date:
            df = df[df['snapshot_date'] == snapshot_date]
        
        # Predict
        results = self.predict(df, model)
        
        # Save results
        output_dir = Path("outputs")
        ensure_dir(str(output_dir))
        
        output_file = output_dir / f"batch_scores_{snapshot_date or 'latest'}.parquet"
        results.to_parquet(output_file, index=False)
        
        self.logger.info(f"Results saved to {output_file}")
        self.logger.info("="*60)
        
        return results


def main():
    import argparse
    from ..utils import load_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-date', type=str, default=None)
    parser.add_argument('--config', default='configs/inference.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    inference = BatchInference(config)
    inference.run(snapshot_date=args.snapshot_date)


if __name__ == '__main__':
    main()
