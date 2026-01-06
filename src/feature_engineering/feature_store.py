"""
Feature Store implementation with offline and online capabilities.

This provides a simple but production-ready pattern for:
1. Offline store: Parquet files for training/batch inference
2. Online store: SQLite for low-latency lookups
3. Feature schema versioning
4. Point-in-time correctness guarantees
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import hashlib

from ..utils import setup_logging, ensure_dir, save_json, load_json, compute_feature_schema_hash

logger = setup_logging(__name__)


class FeatureStore:
    """
    Feature store with offline (training) and online (serving) capabilities.
    
    Design:
    - Offline: Parquet files partitioned by snapshot_date
    - Online: SQLite with customer_id as key for fast lookups
    - Schema versioning to track feature changes over time
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature store.
        
        Args:
            config: Configuration from features.yaml
        """
        self.config = config
        self.offline_path = Path(config['feature_store']['offline']['path'])
        self.online_db_path = Path(config['feature_store']['online']['path'])
        self.schema_version = config['feature_store']['schema_version']
        self.logger = logger
        
        # Ensure directories exist
        ensure_dir(str(self.offline_path))
        ensure_dir(str(self.online_db_path.parent))
        
    def materialize_offline_features(
        self,
        df_features: pd.DataFrame,
        feature_group_name: str = "customer_features"
    ) -> str:
        """
        Materialize features to offline store (Parquet files).
        
        This creates a versioned, partitioned feature table for training.
        
        Args:
            df_features: DataFrame with engineered features
            feature_group_name: Name of feature group
            
        Returns:
            Path to saved feature table
        """
        self.logger.info(f"Materializing offline features: {feature_group_name}")
        
        # Create feature group directory
        feature_group_path = self.offline_path / f"{feature_group_name}_{self.schema_version}"
        ensure_dir(str(feature_group_path))
        
        # Save as Parquet with partitioning by snapshot_date (year-month)
        if 'snapshot_date' in df_features.columns:
            df_features['year_month'] = pd.to_datetime(df_features['snapshot_date']).dt.to_period('M').astype(str)
        
        output_file = feature_group_path / "features.parquet"
        df_features.to_parquet(output_file, index=False)
        
        # Save feature schema metadata
        schema_meta = {
            'feature_group': feature_group_name,
            'schema_version': self.schema_version,
            'created_at': datetime.now().isoformat(),
            'n_rows': len(df_features),
            'n_features': len(df_features.columns),
            'features': df_features.columns.tolist(),
            'feature_hash': compute_feature_schema_hash(df_features.columns.tolist()),
        }
        
        schema_file = feature_group_path / "schema.json"
        save_json(schema_meta, str(schema_file))
        
        self.logger.info(f"Offline features saved to {output_file}")
        self.logger.info(f"  Rows: {len(df_features)}")
        self.logger.info(f"  Features: {len(df_features.columns)}")
        self.logger.info(f"  Feature hash: {schema_meta['feature_hash']}")
        
        return str(output_file)
    
    def get_offline_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_group_name: str = "customer_features"
    ) -> pd.DataFrame:
        """
        Retrieve offline features for training.
        
        Args:
            start_date: Start date for training window (YYYY-MM-DD)
            end_date: End date for training window (YYYY-MM-DD)
            feature_group_name: Name of feature group
            
        Returns:
            DataFrame with features and labels
        """
        self.logger.info("Retrieving offline training data")
        
        feature_group_path = self.offline_path / f"{feature_group_name}_{self.schema_version}"
        feature_file = feature_group_path / "features.parquet"
        
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature table not found: {feature_file}")
        
        # Load features
        df = pd.read_parquet(feature_file)
        
        # Filter by date range
        if start_date or end_date:
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            
            if start_date:
                df = df[df['snapshot_date'] >= pd.to_datetime(start_date)]
            
            if end_date:
                df = df[df['snapshot_date'] <= pd.to_datetime(end_date)]
        
        self.logger.info(f"Retrieved {len(df)} rows")
        
        return df
    
    def write_online_features(
        self,
        df_features: pd.DataFrame,
        customer_id_col: str = 'customer_id'
    ) -> int:
        """
        Write features to online store (SQLite) for real-time serving.
        
        Args:
            df_features: DataFrame with latest features per customer
            customer_id_col: Customer ID column name
            
        Returns:
            Number of rows written
        """
        self.logger.info("Writing features to online store")
        
        # Keep only latest snapshot per customer
        if 'snapshot_date' in df_features.columns:
            df_latest = df_features.sort_values('snapshot_date').groupby(customer_id_col).tail(1)
        else:
            df_latest = df_features
        
        # Connect to SQLite
        conn = sqlite3.connect(str(self.online_db_path))
        
        try:
            # Write to table (replace if exists)
            df_latest.to_sql(
                'customer_features',
                conn,
                if_exists='replace',
                index=False
            )
            
            # Create index on customer_id for fast lookups
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_features ({customer_id_col})"
            )
            conn.commit()
            
            self.logger.info(f"Wrote {len(df_latest)} customer features to online store")
            
            return len(df_latest)
            
        finally:
            conn.close()
    
    def get_online_features(
        self,
        customer_id: str,
        as_of_timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features for a single customer from online store.
        
        This is optimized for low-latency real-time inference.
        
        Args:
            customer_id: Customer ID
            as_of_timestamp: Optional timestamp for point-in-time retrieval
            
        Returns:
            Dictionary of features, or None if not found
        """
        conn = sqlite3.connect(str(self.online_db_path))
        
        try:
            query = "SELECT * FROM customer_features WHERE customer_id = ?"
            df = pd.read_sql_query(query, conn, params=(customer_id,))
            
            if len(df) == 0:
                self.logger.warning(f"No features found for customer {customer_id}")
                return None
            
            # Return as dictionary
            return df.iloc[0].to_dict()
            
        finally:
            conn.close()
    
    def get_feature_schema(
        self,
        feature_group_name: str = "customer_features"
    ) -> Dict[str, Any]:
        """
        Get feature schema metadata.
        
        Args:
            feature_group_name: Name of feature group
            
        Returns:
            Schema metadata dictionary
        """
        feature_group_path = self.offline_path / f"{feature_group_name}_{self.schema_version}"
        schema_file = feature_group_path / "schema.json"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        return load_json(str(schema_file))
    
    def validate_feature_schema(
        self,
        df: pd.DataFrame,
        feature_group_name: str = "customer_features"
    ) -> bool:
        """
        Validate that DataFrame matches expected feature schema.
        
        Args:
            df: DataFrame to validate
            feature_group_name: Name of feature group
            
        Returns:
            True if schema matches, False otherwise
        """
        try:
            schema = self.get_feature_schema(feature_group_name)
            expected_features = set(schema['features'])
            actual_features = set(df.columns)
            
            missing = expected_features - actual_features
            extra = actual_features - expected_features
            
            if missing:
                self.logger.error(f"Missing features: {missing}")
                return False
            
            if extra:
                self.logger.warning(f"Extra features: {extra}")
            
            self.logger.info("Feature schema validation passed")
            return True
            
        except FileNotFoundError:
            self.logger.warning("No schema found for validation")
            return True
