"""
Data ingestion and cleaning for IBM Telco Customer Churn dataset.

This module handles:
1. Loading raw data
2. Data cleaning and validation
3. Initial preprocessing
4. Saving to interim storage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..utils import (
    setup_logging,
    check_missing_values,
    check_duplicates,
    get_data_quality_report,
    ensure_dir,
)

logger = setup_logging(__name__)


class TelcoDataLoader:
    """
    Loader for IBM Telco Customer Churn dataset.
    
    Handles data loading, cleaning, and initial validation with
    careful attention to data quality issues.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary from data.yaml
        """
        self.config = config
        self.raw_data_path = config['paths']['raw_data']
        self.interim_path = config['paths']['interim_data']
        self.logger = logger
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw telco dataset.
        
        Returns:
            Raw dataframe
        """
        self.logger.info(f"Loading raw data from {self.raw_data_path}")
        
        try:
            df = pd.read_csv(self.raw_data_path)
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.raw_data_path}")
            raise
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the telco dataset.
        
        Key operations:
        1. Handle missing values
        2. Fix data type issues
        3. Handle categorical encodings
        4. Remove duplicates
        5. Create derived fields
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        self.logger.info("Starting data cleaning...")
        df = df.copy()
        
        # 0. Map IBM Telco column names to standard names (if present)
        column_mapping = {
            'Customer ID': 'customerID',
            'Tenure in Months': 'tenure',
            'Monthly Charge': 'MonthlyCharges',
            'Total Charges': 'TotalCharges',
            'Churn Label': 'Churn',
            'Married': 'Partner',
            'Senior Citizen': 'SeniorCitizen',
            'Phone Service': 'PhoneService',
            'Multiple Lines': 'MultipleLines',
            'Internet Type': 'InternetService',  # Use Internet Type (has Fiber/DSL/None)
            'Online Security': 'OnlineSecurity',
            'Online Backup': 'OnlineBackup',
            'Device Protection Plan': 'DeviceProtection',
            'Premium Tech Support': 'TechSupport',
            'Streaming TV': 'StreamingTV',
            'Streaming Movies': 'StreamingMovies',
            'Paperless Billing': 'PaperlessBilling',
            'Payment Method': 'PaymentMethod',
            'Contract': 'Contract'
        }
        # Drop Internet Service (Yes/No) and keep Internet Type (Fiber Optic/DSL/None)
        if 'Internet Type' in df.columns and 'Internet Service' in df.columns:
            df.drop(columns=['Internet Service'], inplace=True)
        
        df.rename(columns=column_mapping, inplace=True)
        
        # 1. Convert customerID to string and rename
        if 'customerID' in df.columns:
            df.rename(columns={'customerID': 'customer_id'}, inplace=True)
        
        df['customer_id'] = df['customer_id'].astype(str)
        
        # 2. Handle TotalCharges (often has empty strings instead of numbers)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'],
                errors='coerce'  # Convert invalid values to NaN
            )
            
            # Impute missing TotalCharges
            # For new customers (tenure=0), TotalCharges should be ~MonthlyCharges
            mask = df['TotalCharges'].isnull() & (df['tenure'] == 0)
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
            
            # For others, use tenure * MonthlyCharges as estimate
            mask = df['TotalCharges'].isnull() & (df['tenure'] > 0)
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
        
        # 3. Standardize column names (snake_case)
        df.columns = [self._to_snake_case(col) for col in df.columns]
        
        # 4. Convert binary categorical to numeric
        binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
        
        binary_cols = [
            'partner', 'dependents', 'phone_service', 'paperless_billing',
            'churn'
        ]
        
        for col in binary_cols:
            if col in df.columns:
                # Handle different possible values
                if df[col].dtype == 'object':
                    df[col] = df[col].map(binary_map)
        
        # 5. Handle 'No phone service' and 'No internet service' values
        # These should be treated as 'No' for the specific services
        service_cols = [
            'multiple_lines', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        for col in service_cols:
            if col in df.columns:
                df[col] = df[col].replace({
                    'No phone service': 'No',
                    'No internet service': 'No',
                    'Yes': 1,
                    'No': 0
                })
        
        # 6. Convert gender
        if 'gender' in df.columns:
            df['gender_male'] = (df['gender'] == 'Male').astype(int)
        
        # 7. Handle senior_citizen (already 0/1 in original data)
        # No action needed
        
        # 8. Remove duplicates
        n_before = len(df)
        df.drop_duplicates(subset=['customer_id'], inplace=True)
        n_after = len(df)
        
        if n_before != n_after:
            self.logger.warning(f"Removed {n_before - n_after} duplicate customer records")
        
        # 9. Validate critical fields
        critical_fields = ['customer_id', 'tenure', 'monthly_charges', 'churn']
        missing = [col for col in critical_fields if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing critical columns: {missing}")
        
        # 10. Drop rows with missing critical values
        df.dropna(subset=critical_fields, inplace=True)
        
        self.logger.info(f"Data cleaning complete. Shape: {df.shape}")
        
        return df
    
    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate cleaned data.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            True if validation passes
        """
        self.logger.info("Validating data...")
        
        # Check for missing values
        missing_threshold = self.config['cleaning'].get('drop_missing_threshold', 0.5)
        problematic = check_missing_values(df, threshold=missing_threshold)
        
        if problematic:
            self.logger.warning(f"Columns with excessive missing values: {problematic}")
        
        # Check duplicates
        n_duplicates = check_duplicates(df, subset=['customer_id'])
        
        if n_duplicates > 0:
            self.logger.error(f"Found {n_duplicates} duplicate customer IDs")
            return False
        
        # Check value ranges
        # Tenure should be non-negative
        if (df['tenure'] < 0).any():
            self.logger.error("Found negative tenure values")
            return False
        
        # Monthly charges should be positive
        if (df['monthly_charges'] <= 0).any():
            self.logger.error("Found non-positive monthly charges")
            return False
        
        # Churn should be 0 or 1
        if not df['churn'].isin([0, 1]).all():
            self.logger.error("Churn column contains invalid values")
            return False
        
        self.logger.info("Data validation passed")
        return True
    
    def save_interim_data(self, df: pd.DataFrame) -> str:
        """
        Save cleaned data to interim storage.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Path to saved file
        """
        ensure_dir(self.interim_path)
        output_path = Path(self.interim_path) / "telco_cleaned.parquet"
        
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved interim data to {output_path}")
        
        return str(output_path)
    
    def run(self) -> pd.DataFrame:
        """
        Execute full data ingestion pipeline.
        
        Returns:
            Cleaned dataframe
        """
        self.logger.info("="*60)
        self.logger.info("STARTING DATA INGESTION PIPELINE")
        self.logger.info("="*60)
        
        # Load raw data
        df_raw = self.load_raw_data()
        
        # Generate quality report
        self.logger.info("Raw data quality report:")
        quality_report = get_data_quality_report(df_raw)
        self.logger.info(f"  Rows: {quality_report['n_rows']}")
        self.logger.info(f"  Columns: {quality_report['n_columns']}")
        self.logger.info(f"  Duplicates: {quality_report['n_duplicates']}")
        
        # Clean data
        df_clean = self.clean_data(df_raw)
        
        # Validate
        if not self.validate_data(df_clean):
            raise ValueError("Data validation failed")
        
        # Save interim data
        self.save_interim_data(df_clean)
        
        self.logger.info("="*60)
        self.logger.info("DATA INGESTION COMPLETE")
        self.logger.info("="*60)
        
        return df_clean


def run_data_ingestion(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    """
    Convenience function to run data ingestion.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Cleaned dataframe
    """
    from ..utils import load_config
    
    config = load_config(config_path)
    loader = TelcoDataLoader(config)
    return loader.run()
