"""
Temporal dataset builder - expands customers into monthly snapshots.

This is the CORE of the temporalization strategy:
1. Each customer gets multiple rows (one per month they were active)
2. Features are computed with strict point-in-time correctness
3. Labels are assigned only for the prediction horizon
4. No future data leakage

Key principle: At snapshot_date T, we can only use information available up to T.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from ..utils import (
    setup_logging,
    generate_monthly_dates,
    get_months_between,
    add_months,
    ensure_dir,
)

logger = setup_logging(__name__)


class TemporalDatasetBuilder:
    """
    Builds temporalized customer snapshots from static churn data.
    
    Strategy:
    - For each customer, create monthly snapshots from their start to churn/end
    - Each snapshot represents what we knew about the customer at that month
    - Label indicates whether they churned in the NEXT period
    - This enables proper time-series modeling with no label leakage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize temporal dataset builder.
        
        Args:
            config: Configuration dictionary from data.yaml
        """
        self.config = config
        self.start_date = pd.to_datetime(config['temporal']['start_date'])
        self.end_date = pd.to_datetime(config['temporal']['end_date'])
        self.prediction_horizon = config['temporal']['prediction_horizon_months']
        self.logger = logger
        
    def expand_customer_to_snapshots(
        self,
        customer_row: pd.Series,
        observation_start: datetime,
        observation_end: datetime
    ) -> pd.DataFrame:
        """
        Expand a single customer into monthly snapshots.
        
        Logic:
        - Customer joined at some point (inferred from tenure)
        - We observe them monthly until they churn or observation ends
        - Each snapshot has features valid at that point in time
        - Churn label is 1 only for the last N months before actual churn
        
        Args:
            customer_row: Single row from cleaned dataset
            observation_start: Start of observation period
            observation_end: End of observation period
            
        Returns:
            DataFrame with multiple rows (one per month) for this customer
        """
        customer_id = customer_row['customer_id']
        tenure_months = int(customer_row['tenure'])
        churned = int(customer_row['churn'])
        
        # Infer customer start date
        # Since we don't have actual dates, simulate that customer started
        # `tenure` months before observation_end
        customer_start = add_months(observation_end, -tenure_months)
        
        # Customer end date
        if churned:
            # For churned customers, distribute churn dates throughout the observation period
            # Assumption: customer churned sometime during their tenure
            # Randomly place churn between their start date + min months and observation_end
            # This creates a more realistic temporal distribution of churn
            import random
            random.seed(hash(customer_id) % (2**32))  # Deterministic but varies per customer
            
            # Churn could happen any time from month 1 to their final tenure month
            # Place churn uniformly in the observation window
            min_churn_month = max(1, int(tenure_months * 0.1))  # At least 10% into tenure
            max_churn_offset = min(tenure_months, get_months_between(observation_start, observation_end))
            churn_offset_months = random.randint(min_churn_month, max_churn_offset)
            
            customer_churn_date = add_months(customer_start, churn_offset_months)
            # Last snapshot should be BEFORE churn (to predict it)
            customer_end = add_months(customer_churn_date, -1)
        else:
            # Customer still active at observation_end
            customer_end = observation_end
        
        # Generate monthly snapshots
        snapshot_dates = pd.date_range(
            start=max(customer_start, observation_start),
            end=customer_end,
            freq='MS'  # Month start
        )
        
        snapshots = []
        
        for i, snapshot_date in enumerate(snapshot_dates):
            # Tenure at this snapshot
            tenure_at_snapshot = get_months_between(customer_start, snapshot_date)
            
            # Determine churn label
            # Label is 1 ONLY if customer will churn EXACTLY in prediction_horizon months
            # This is the last observable snapshot before churn
            months_until_end = get_months_between(snapshot_date, add_months(customer_end, 1))
            
            # For churned customers: label=1 only when exactly prediction_horizon months remain
            # For active customers: always label=0 (no churn in next period)
            if churned and months_until_end == self.prediction_horizon:
                churn_label_next_period = 1
                months_until_churn = self.prediction_horizon
            else:
                churn_label_next_period = 0
                months_until_churn = None if not churned else months_until_end
            
            # Create snapshot row
            snapshot = {
                'customer_id': customer_id,
                'snapshot_date': snapshot_date,
                'tenure_months': tenure_at_snapshot,
                'churn_label_next_period': churn_label_next_period,
                
                # Copy static features (demographics, service types)
                # These don't change over time in this dataset
                'gender_male': customer_row.get('gender_male', 0),
                'senior_citizen': customer_row.get('senior_citizen', 0),
                'partner': customer_row.get('partner', 0),
                'dependents': customer_row.get('dependents', 0),
                
                # Service features (assume constant for simplicity)
                'phone_service': customer_row.get('phone_service', 0),
                'multiple_lines': customer_row.get('multiple_lines', 0),
                'internet_service': customer_row.get('internet_service', ''),
                'online_security': customer_row.get('online_security', 0),
                'online_backup': customer_row.get('online_backup', 0),
                'device_protection': customer_row.get('device_protection', 0),
                'tech_support': customer_row.get('tech_support', 0),
                'streaming_tv': customer_row.get('streaming_tv', 0),
                'streaming_movies': customer_row.get('streaming_movies', 0),
                
                # Contract features
                'contract': customer_row.get('contract', ''),
                'paperless_billing': customer_row.get('paperless_billing', 0),
                'payment_method': customer_row.get('payment_method', ''),
                
                # Financial features (will be augmented with rolling features later)
                'monthly_charges': customer_row.get('monthly_charges', 0),
                'total_charges': customer_row.get('total_charges', 0),
                
                # Metadata
                'months_until_churn': months_until_churn,
            }
            
            snapshots.append(snapshot)
        
        return pd.DataFrame(snapshots)
    
    def build_temporal_dataset(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete temporal dataset from cleaned static data.
        
        This is where we transform the static dataset into a time-series dataset.
        
        Args:
            df_clean: Cleaned customer data (one row per customer)
            
        Returns:
            Temporal dataset (multiple rows per customer)
        """
        self.logger.info("Building temporal dataset...")
        self.logger.info(f"  Observation period: {self.start_date} to {self.end_date}")
        self.logger.info(f"  Prediction horizon: {self.prediction_horizon} months")
        
        all_snapshots = []
        
        for idx, customer_row in df_clean.iterrows():
            customer_snapshots = self.expand_customer_to_snapshots(
                customer_row,
                self.start_date,
                self.end_date
            )
            all_snapshots.append(customer_snapshots)
            
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"  Processed {idx + 1}/{len(df_clean)} customers")
        
        # Combine all snapshots
        df_temporal = pd.concat(all_snapshots, ignore_index=True)
        
        # Sort by customer and date for temporal consistency
        df_temporal.sort_values(['customer_id', 'snapshot_date'], inplace=True)
        df_temporal.reset_index(drop=True, inplace=True)
        
        self.logger.info(f"Temporal dataset created:")
        self.logger.info(f"  Total snapshots: {len(df_temporal)}")
        self.logger.info(f"  Unique customers: {df_temporal['customer_id'].nunique()}")
        self.logger.info(f"  Churn rate: {df_temporal['churn_label_next_period'].mean():.2%}")
        
        # Validate temporal order
        from ..utils import validate_temporal_consistency
        if not validate_temporal_consistency(df_temporal):
            raise ValueError("Temporal consistency validation failed!")
        
        return df_temporal
    
    def create_customers_master(self, df_temporal: pd.DataFrame) -> pd.DataFrame:
        """
        Create a master customer table with summary information.
        
        This is useful for analytics and joins.
        
        Args:
            df_temporal: Temporal dataset
            
        Returns:
            Customer master table (one row per customer)
        """
        customer_master = df_temporal.groupby('customer_id').agg({
            'snapshot_date': ['min', 'max'],
            'tenure_months': 'max',
            'churn_label_next_period': 'max',  # 1 if ever churned
            'monthly_charges': 'mean',
            'total_charges': 'max',
            'gender_male': 'first',
            'senior_citizen': 'first',
            'partner': 'first',
            'dependents': 'first',
        }).reset_index()
        
        # Flatten column names
        customer_master.columns = [
            'customer_id',
            'first_snapshot_date',
            'last_snapshot_date',
            'max_tenure',
            'churned',
            'avg_monthly_charges',
            'total_charges',
            'gender_male',
            'senior_citizen',
            'partner',
            'dependents',
        ]
        
        return customer_master
    
    def save_temporal_dataset(
        self,
        df_temporal: pd.DataFrame,
        df_customers: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Save temporal dataset and customer master to disk.
        
        Args:
            df_temporal: Temporal dataset
            df_customers: Customer master table
            
        Returns:
            Dictionary with paths to saved files
        """
        datasets_path = self.config['paths']['datasets_output']
        ensure_dir(datasets_path)
        
        # Save temporal snapshots
        temporal_path = f"{datasets_path}/monthly_customer_snapshots.parquet"
        df_temporal.to_parquet(temporal_path, index=False)
        self.logger.info(f"Saved temporal snapshots to {temporal_path}")
        
        # Save customer master
        customers_path = f"{datasets_path}/customers_master.csv"
        df_customers.to_csv(customers_path, index=False)
        self.logger.info(f"Saved customer master to {customers_path}")
        
        return {
            'temporal': temporal_path,
            'customers': customers_path,
        }
    
    def run(self, df_clean: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Execute full temporal dataset building pipeline.
        
        Args:
            df_clean: Cleaned customer data
            
        Returns:
            Dictionary with temporal and customer dataframes
        """
        self.logger.info("="*60)
        self.logger.info("STARTING TEMPORAL DATASET CONSTRUCTION")
        self.logger.info("="*60)
        
        # Build temporal dataset
        df_temporal = self.build_temporal_dataset(df_clean)
        
        # Create customer master
        df_customers = self.create_customers_master(df_temporal)
        
        # Save datasets
        self.save_temporal_dataset(df_temporal, df_customers)
        
        self.logger.info("="*60)
        self.logger.info("TEMPORAL DATASET CONSTRUCTION COMPLETE")
        self.logger.info("="*60)
        
        return {
            'temporal': df_temporal,
            'customers': df_customers,
        }
