"""
Main runner for data ingestion pipeline.

Usage:
    python -m src.data_ingestion.run --config configs/data.yaml
"""

import argparse
from pathlib import Path

from .loader import TelcoDataLoader
from .temporal_builder import TemporalDatasetBuilder
from ..utils import load_config, setup_logging


def main():
    """Run complete data ingestion and temporalization pipeline."""
    parser = argparse.ArgumentParser(
        description="Data ingestion and temporalization pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration file'
    )
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(__name__, log_file="logs/data_ingestion.log")
    logger.info("Starting data ingestion pipeline")
    
    # Load configuration
    config = load_config(args.config)
    
    # Step 1: Load and clean data
    loader = TelcoDataLoader(config)
    df_clean = loader.run()
    
    # Step 2: Build temporal dataset
    temporal_builder = TemporalDatasetBuilder(config)
    datasets = temporal_builder.run(df_clean)
    
    logger.info("Pipeline complete!")
    logger.info(f"  Temporal snapshots: {len(datasets['temporal'])} rows")
    logger.info(f"  Unique customers: {len(datasets['customers'])} rows")


if __name__ == '__main__':
    main()
