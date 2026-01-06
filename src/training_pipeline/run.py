"""
Run training pipeline.

Usage:
    python -m src.training_pipeline.run --config configs/training.yaml
"""

import argparse
from ..utils import load_config
from ..feature_engineering import FeatureStore
from .trainer import ChurnModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument('--config', default='configs/training.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load features
    feature_store = FeatureStore(load_config('configs/features.yaml'))
    df = feature_store.get_offline_training_data()
    
    # Train model
    trainer = ChurnModelTrainer(config)
    result = trainer.run(df)
    
    print(f"\n✅ Training complete!")
    print(f"Model version: {result['model_version']}")
    print(f"ROC-AUC: {result['metrics']['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
