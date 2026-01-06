"""
End-to-end pipeline runner.

This script demonstrates the complete workflow from data ingestion to deployment.

Usage:
    python run_pipeline.py --mode full
    python run_pipeline.py --mode train_only
    python run_pipeline.py --mode inference_only
"""

import argparse
import logging
from pathlib import Path

from src.utils import load_config, setup_logging
from src.data_ingestion import TelcoDataLoader, TemporalDatasetBuilder
from src.feature_engineering import FeatureEngineer, FeatureStore
from src.training_pipeline import ChurnModelTrainer
from src.inference import BatchInference
from src.decision_engine import DecisionEngine
from src.models import ModelRegistry


def run_full_pipeline():
    """Execute the complete end-to-end pipeline."""
    logger = setup_logging(__name__, log_file="logs/pipeline.log")
    logger.info("="*80)
    logger.info("STARTING FULL PIPELINE EXECUTION")
    logger.info("="*80)
    
    try:
        # 1. Data Ingestion & Temporalization
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA INGESTION & TEMPORALIZATION")
        logger.info("="*80)
        
        data_config = load_config('configs/data.yaml')
        
        # Load and clean data
        loader = TelcoDataLoader(data_config)
        df_clean = loader.run()
        
        # Build temporal dataset
        temporal_builder = TemporalDatasetBuilder(data_config)
        datasets = temporal_builder.run(df_clean)
        df_temporal = datasets['temporal']
        
        logger.info(f"✓ Created {len(df_temporal)} temporal snapshots")
        
        # 2. Feature Engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        features_config = load_config('configs/features.yaml')
        
        feature_engineer = FeatureEngineer(features_config)
        df_features = feature_engineer.run(df_temporal)
        
        logger.info(f"✓ Engineered {len(df_features.columns)} features")
        
        # 3. Feature Store Materialization
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FEATURE STORE MATERIALIZATION")
        logger.info("="*80)
        
        feature_store = FeatureStore(features_config)
        
        # Save to offline store
        feature_store.materialize_offline_features(df_features)
        
        # Save to online store (latest snapshot per customer)
        feature_store.write_online_features(df_features)
        
        logger.info("✓ Features materialized to offline and online stores")
        
        # 4. Model Training
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*80)
        
        training_config = load_config('configs/training.yaml')
        
        trainer = ChurnModelTrainer(training_config)
        result = trainer.run(df_features)
        
        logger.info(f"✓ Model trained: {result['model_version']}")
        logger.info(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {result['metrics']['pr_auc']:.4f}")
        
        # Promote to production if first model
        registry = ModelRegistry()
        production_model = registry.get_production_model()
        
        if production_model is None:
            registry.promote_to_production(result['model_version'])
            logger.info(f"✓ Promoted {result['model_version']} to production")
        
        # 5. Batch Inference
        logger.info("\n" + "="*80)
        logger.info("STEP 5: BATCH INFERENCE")
        logger.info("="*80)
        
        inference_config = load_config('configs/inference.yaml')
        
        batch_inference = BatchInference(inference_config)
        scores = batch_inference.run(snapshot_date=None)
        
        logger.info(f"✓ Generated predictions for {len(scores)} customers")
        
        # 6. Decision Engine
        logger.info("\n" + "="*80)
        logger.info("STEP 6: DECISION ENGINE")
        logger.info("="*80)
        
        decision_config = load_config('configs/decision.yaml')
        
        # Save scores for decision engine
        scores_path = "outputs/latest_scores.parquet"
        scores.to_parquet(scores_path, index=False)
        
        decision_engine = DecisionEngine(decision_config)
        targeted = decision_engine.run(scores_path)
        
        logger.info(f"✓ Identified {len(targeted)} customers for retention")
        
        # 7. Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETE ✓")
        logger.info("="*80)
        logger.info("\nResults:")
        logger.info(f"  📊 Temporal snapshots: {len(df_temporal):,}")
        logger.info(f"  🔧 Features engineered: {len(df_features.columns)}")
        logger.info(f"  🤖 Model: {result['model_version']}")
        logger.info(f"  📈 ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        logger.info(f"  🎯 Customers to target: {len(targeted):,}")
        logger.info(f"  💰 Expected total value: ${targeted['expected_value'].sum():,.2f}")
        logger.info("\n" + "="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return False


def run_training_only():
    """Run only the training pipeline."""
    logger = setup_logging(__name__, log_file="logs/training.log")
    logger.info("Running training pipeline only...")
    
    try:
        training_config = load_config('configs/training.yaml')
        features_config = load_config('configs/features.yaml')
        
        # Load features
        feature_store = FeatureStore(features_config)
        df = feature_store.get_offline_training_data()
        
        # Train
        trainer = ChurnModelTrainer(training_config)
        result = trainer.run(df)
        
        logger.info(f"✓ Training complete: {result['model_version']}")
        logger.info(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


def run_inference_only():
    """Run only inference and decision pipeline."""
    logger = setup_logging(__name__, log_file="logs/inference.log")
    logger.info("Running inference pipeline only...")
    
    try:
        inference_config = load_config('configs/inference.yaml')
        decision_config = load_config('configs/decision.yaml')
        
        # Batch inference
        batch_inference = BatchInference(inference_config)
        scores = batch_inference.run()
        
        # Decision engine
        scores_path = "outputs/latest_scores.parquet"
        scores.to_parquet(scores_path, index=False)
        
        decision_engine = DecisionEngine(decision_config)
        targeted = decision_engine.run(scores_path)
        
        logger.info(f"✓ Inference complete")
        logger.info(f"  Customers scored: {len(scores):,}")
        logger.info(f"  Customers targeted: {len(targeted):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --mode full
  
  # Train new model only
  python run_pipeline.py --mode train_only
  
  # Run inference only
  python run_pipeline.py --mode inference_only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'train_only', 'inference_only'],
        default='full',
        help='Pipeline execution mode'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 CUSTOMER CHURN PREDICTION PLATFORM")
    print("="*80 + "\n")
    
    success = False
    
    if args.mode == 'full':
        success = run_full_pipeline()
    elif args.mode == 'train_only':
        success = run_training_only()
    elif args.mode == 'inference_only':
        success = run_inference_only()
    
    if success:
        print("\n✅ Pipeline completed successfully!\n")
        return 0
    else:
        print("\n❌ Pipeline failed. Check logs for details.\n")
        return 1


if __name__ == '__main__':
    exit(main())
