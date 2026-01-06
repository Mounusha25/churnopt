"""
Feature engineering module with rolling aggregations and temporal features.

Critical design principle: POINT-IN-TIME CORRECTNESS
- All features at snapshot_date T use ONLY data available up to T
- No future information leakage
- Rolling windows look backward only
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from ..utils import setup_logging, get_time_windows, add_months

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Computes features with strict point-in-time correctness.
    
    Features include:
    1. Rolling aggregations (3m, 6m, 12m windows)
    2. Trend features (slopes, changes)
    3. Service adoption features
    4. Temporal features
    5. Simulated behavioral features
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration from features.yaml
        """
        self.config = config
        self.windows = config.get('rolling_windows', {
            'short': 3,
            'medium': 6,
            'long': 12
        })
        self.logger = logger
        
    def compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling aggregation features.
        
        For each snapshot, compute features over previous N months.
        
        Args:
            df: Temporal dataset with monthly snapshots
            
        Returns:
            DataFrame with added rolling features
        """
        self.logger.info("Computing rolling features...")
        df = df.copy()
        
        # Sort to ensure temporal order
        df.sort_values(['customer_id', 'snapshot_date'], inplace=True)
        
        # Group by customer
        for customer_id, group in df.groupby('customer_id'):
            indices = group.index
            
            for idx in indices:
                snapshot_date = group.loc[idx, 'snapshot_date']
                
                # Get data up to this snapshot (point-in-time correctness)
                historical_data = group[group['snapshot_date'] <= snapshot_date]
                
                # Rolling 3-month features
                data_3m = historical_data[
                    historical_data['snapshot_date'] >= add_months(snapshot_date, -self.windows['short'])
                ]
                
                df.loc[idx, 'avg_monthly_charges_3m'] = data_3m['monthly_charges'].mean()
                df.loc[idx, 'std_monthly_charges_3m'] = data_3m['monthly_charges'].std()
                
                # Rolling 6-month features
                data_6m = historical_data[
                    historical_data['snapshot_date'] >= add_months(snapshot_date, -self.windows['medium'])
                ]
                
                df.loc[idx, 'avg_monthly_charges_6m'] = data_6m['monthly_charges'].mean()
                
                # Compute trend (linear regression slope)
                if len(data_6m) >= 2:
                    x = np.arange(len(data_6m))
                    y = data_6m['monthly_charges'].values
                    
                    # Simple linear regression
                    if len(x) > 0 and y.std() > 0:
                        slope = np.polyfit(x, y, 1)[0]
                        df.loc[idx, 'charges_trend_6m'] = slope
                    else:
                        df.loc[idx, 'charges_trend_6m'] = 0
                else:
                    df.loc[idx, 'charges_trend_6m'] = 0
        
        # Fill NaN with 0 or forward-fill
        rolling_cols = [
            'avg_monthly_charges_3m', 'std_monthly_charges_3m',
            'avg_monthly_charges_6m', 'charges_trend_6m'
        ]
        
        for col in rolling_cols:
            if col in df.columns:
                df[col].fillna(0, inplace=True)
        
        self.logger.info(f"Added {len(rolling_cols)} rolling features")
        
        return df
    
    def compute_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute service adoption features.
        
        Args:
            df: DataFrame with service columns
            
        Returns:
            DataFrame with service features
        """
        self.logger.info("Computing service features...")
        df = df.copy()
        
        # Total number of services
        service_cols = [
            'phone_service', 'multiple_lines', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        available_service_cols = [col for col in service_cols if col in df.columns]
        
        df['num_services_total'] = df[available_service_cols].sum(axis=1)
        
        # Internet service type (categorical)
        if 'internet_service' in df.columns:
            df['has_fiber_optic'] = (df['internet_service'] == 'Fiber Optic').astype(int)
            df['has_dsl'] = (df['internet_service'] == 'DSL').astype(int)
            df['has_cable'] = (df['internet_service'] == 'Cable').astype(int)
            df['no_internet'] = (df['internet_service'].isna() | (df['internet_service'] == 'No')).astype(int)
        
        self.logger.info("Service features computed")
        
        return df
    
    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal and tenure-based features.
        
        Args:
            df: DataFrame with tenure information
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Computing temporal features...")
        df = df.copy()
        
        # Tenure buckets (categorical)
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 12, 24, 48, 1000],
            labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+']
        )
        
        # Tenure bucket as numeric
        tenure_bucket_map = {'0-6m': 0, '6-12m': 1, '12-24m': 2, '24-48m': 3, '48m+': 4}
        df['tenure_bucket_num'] = df['tenure_bucket'].map(tenure_bucket_map)
        
        # Contract-related features
        if 'contract' in df.columns:
            df['is_month_to_month'] = (df['contract'] == 'Month-to-Month').astype(int)
            df['is_one_year'] = (df['contract'] == 'One Year').astype(int)
            df['is_two_year'] = (df['contract'] == 'Two Year').astype(int)
            
            # Estimate contract remaining months (simplified)
            df['contract_remaining_months'] = 0
            df.loc[df['is_one_year'] == 1, 'contract_remaining_months'] = \
                np.maximum(12 - df['tenure_months'], 0)
            df.loc[df['is_two_year'] == 1, 'contract_remaining_months'] = \
                np.maximum(24 - df['tenure_months'], 0)
        
        # Payment method features
        if 'payment_method' in df.columns:
            df['payment_electronic_check'] = (
                df['payment_method'] == 'Electronic check'
            ).astype(int)
            df['payment_auto'] = (
                df['payment_method'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])
            ).astype(int)
        
        self.logger.info("Temporal features computed")
        
        return df
    
    def compute_simulated_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate behavioral features (in production, these would come from event tables).
        
        Features like:
        - Support tickets
        - Late payments  
        - Plan changes
        
        For this demo, we'll create realistic simulated values based on
        patterns that correlate with churn risk.
        
        Args:
            df: DataFrame with customer snapshots
            
        Returns:
            DataFrame with simulated behavioral features
        """
        self.logger.info("Computing simulated behavioral features...")
        df = df.copy()
        
        np.random.seed(42)
        
        # Support tickets (churners likely have more)
        # Base rate + increased for high-risk segments
        base_tickets = np.random.poisson(0.5, size=len(df))
        
        # Increase for month-to-month contracts
        if 'is_month_to_month' in df.columns:
            mtm_boost = np.random.poisson(1, size=len(df)) * df['is_month_to_month']
            df['num_support_tickets_3m'] = base_tickets + mtm_boost
        else:
            df['num_support_tickets_3m'] = base_tickets
        
        # Late payments (higher for electronic check)
        base_late = np.random.binomial(1, 0.1, size=len(df))
        
        if 'payment_electronic_check' in df.columns:
            late_boost = np.random.binomial(1, 0.2, size=len(df)) * df['payment_electronic_check']
            df['num_late_payments_6m'] = base_late + late_boost
        else:
            df['num_late_payments_6m'] = base_late
        
        # Plan changes
        df['num_plan_changes_6m'] = np.random.poisson(0.2, size=len(df))
        
        # Days since last contact (simulate)
        df['days_since_last_contact'] = np.random.randint(1, 180, size=len(df))
        
        self.logger.info("Simulated behavioral features created")
        
        return df
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from existing columns.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with derived features
        """
        self.logger.info("Computing derived features...")
        df = df.copy()
        
        # Charges per tenure month
        df['charges_per_tenure_month'] = df['monthly_charges'] / (df['tenure_months'] + 1)
        
        # Total charges estimate
        df['estimated_total_charges'] = df['monthly_charges'] * df['tenure_months']
        
        # Service adoption rate (services per tenure month)
        if 'num_services_total' in df.columns:
            df['services_per_month'] = df['num_services_total'] / (df['tenure_months'] + 1)
        
        # High-value customer flag
        df['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        
        self.logger.info("Derived features computed")
        
        return df
    
    def run(self, df_temporal: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        
        Args:
            df_temporal: Temporal dataset from temporalization step
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("="*60)
        self.logger.info("STARTING FEATURE ENGINEERING")
        self.logger.info("="*60)
        
        df = df_temporal.copy()
        
        # Compute feature groups
        df = self.compute_service_features(df)
        df = self.compute_temporal_features(df)
        df = self.compute_simulated_behavioral_features(df)
        df = self.compute_derived_features(df)
        
        # Rolling features (computationally intensive, so last)
        # For large datasets, this should be optimized or computed incrementally
        self.logger.info("Computing rolling features (this may take a while)...")
        # For demo purposes, compute for a sample
        if len(df) > 50000:
            self.logger.warning(
                f"Large dataset ({len(df)} rows). Rolling features will be approximated."
            )
            # For demo: just fill with current values
            df['avg_monthly_charges_3m'] = df['monthly_charges']
            df['std_monthly_charges_3m'] = 0
            df['avg_monthly_charges_6m'] = df['monthly_charges']
            df['charges_trend_6m'] = 0
        else:
            df = self.compute_rolling_features(df)
        
        self.logger.info("="*60)
        self.logger.info("FEATURE ENGINEERING COMPLETE")
        self.logger.info(f"Total features: {len(df.columns)}")
        self.logger.info("="*60)
        
        return df


def get_feature_names(df: pd.DataFrame, exclude_meta: bool = True) -> List[str]:
    """
    Get list of feature column names.
    
    Args:
        df: DataFrame with features
        exclude_meta: Whether to exclude metadata columns
        
    Returns:
        List of feature column names
    """
    meta_cols = [
        'customer_id', 'snapshot_date', 'churn_label_next_period',
        'months_until_churn', 'tenure_bucket'  # categorical, not numeric
    ]
    
    if exclude_meta:
        return [col for col in df.columns if col not in meta_cols]
    else:
        return df.columns.tolist()
