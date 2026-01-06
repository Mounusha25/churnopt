"""
Training pipeline with time-aware cross-validation and MLflow tracking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime

from ..utils import setup_logging, ensure_dir
from ..models.registry import ModelRegistry
from ..feature_engineering import get_feature_names

logger = setup_logging(__name__)


class ChurnModelTrainer:
    """
    Train churn prediction models with production best practices.
    
    Features:
    - Time-aware cross-validation
    - Multiple model types
    - Hyperparameter tuning
    - Comprehensive evaluation
    - Experiment tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config['model']['type']
        self.model_config = config['model'].get(self.model_type, {})
        self.logger = logger
        self.registry = ModelRegistry(config.get('registry', {}).get('path', 'models'))
        
    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and labels for training.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        self.logger.info("Preparing training data...")
        
        # Exclude metadata columns
        exclude_cols = [
            'customer_id', 'snapshot_date', 'churn_label_next_period',
            'months_until_churn', 'year_month', 'tenure_bucket'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical columns
        X = df[feature_cols].copy()
        
        # Convert remaining categorical to numeric
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Fill NaN with 0
        X = X.fillna(0)
        
        y = df['churn_label_next_period']
        
        self.logger.info(f"  Features: {len(feature_cols)}")
        self.logger.info(f"  Samples: {len(X)}")
        self.logger.info(f"  Churn rate: {y.mean():.2%}")
        
        return X, y, feature_cols
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train a churn prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        self.logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            # Auto-calculate scale_pos_weight for class imbalance
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            self.logger.info(f"Class distribution: {neg_count} negative, {pos_count} positive")
            self.logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
            
            # Override scale_pos_weight in config
            model_config = self.model_config.copy()
            model_config['scale_pos_weight'] = scale_pos_weight
            # Remove base_score to let XGBoost calculate it automatically
            model_config.pop('base_score', None)
            
            model = xgb.XGBClassifier(**model_config)
            
        elif self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**self.model_config)
            
        elif self.model_type == 'logistic':
            model = LogisticRegression(**self.model_config)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.fit(X_train, y_train)
        
        self.logger.info("Model training complete")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Evaluating model...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
        }
        
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(
        self,
        model: Any,
        feature_names: List[str],
        metrics: Dict[str, float]
    ) -> str:
        """
        Save model and register in model registry.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            metrics: Evaluation metrics
            
        Returns:
            Model version string
        """
        # Generate model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"model_v_{timestamp}"
        
        model_dir = Path("models") / model_version
        ensure_dir(str(model_dir))
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Prepare metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': feature_names,
            'metrics': metrics,
            'config': self.model_config,
            'trained_at': datetime.now().isoformat(),
        }
        
        # Register model
        self.registry.register_model(
            str(model_path),
            model_version,
            metadata,
            status='staging'
        )
        
        self.logger.info(f"Model saved as {model_version}")
        
        return model_version
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Dictionary with model and metrics
        """
        self.logger.info("="*60)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("="*60)
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        
        # Time-aware train/test split
        cutoff_date = pd.to_datetime(self.config['data'].get('val_start_date', '2024-07-01'))
        
        if 'snapshot_date' in df.columns:
            train_mask = pd.to_datetime(df['snapshot_date']) < cutoff_date
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
        else:
            # Fallback to standard split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model
        model = self.train_model(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Save model
        model_version = self.save_model(model, feature_names, metrics)
        
        self.logger.info("="*60)
        self.logger.info("TRAINING PIPELINE COMPLETE")
        self.logger.info("="*60)
        
        return {
            'model': model,
            'model_version': model_version,
            'metrics': metrics,
            'feature_names': feature_names,
        }
