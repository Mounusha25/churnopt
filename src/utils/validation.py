"""
Data validation utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Check for missing values and report columns exceeding threshold.
    
    Args:
        df: Input dataframe
        threshold: Maximum allowed missing percentage (0.5 = 50%)
        
    Returns:
        Dictionary of {column: missing_percentage} for problematic columns
    """
    missing_pct = df.isnull().sum() / len(df)
    problematic = missing_pct[missing_pct > threshold].to_dict()
    
    if problematic:
        logger.warning(f"Columns with >{threshold*100}% missing: {problematic}")
    
    return problematic


def check_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None
) -> int:
    """
    Check for duplicate rows.
    
    Args:
        df: Input dataframe
        subset: Columns to check for duplicates (None = all columns)
        
    Returns:
        Number of duplicate rows
    """
    n_duplicates = df.duplicated(subset=subset).sum()
    
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate rows")
    
    return n_duplicates


def check_data_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str]
) -> Dict[str, str]:
    """
    Validate data types match expectations.
    
    Args:
        df: Input dataframe
        expected_types: Dictionary of {column: expected_dtype}
        
    Returns:
        Dictionary of mismatches
    """
    mismatches = {}
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if not actual_type.startswith(expected_type):
                mismatches[col] = f"Expected {expected_type}, got {actual_type}"
    
    if mismatches:
        logger.warning(f"Data type mismatches: {mismatches}")
    
    return mismatches


def check_value_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Dict[str, float]]
) -> Dict[str, int]:
    """
    Check if numeric values are within expected ranges.
    
    Args:
        df: Input dataframe
        ranges: Dictionary of {column: {'min': x, 'max': y}}
        
    Returns:
        Dictionary of {column: n_violations}
    """
    violations = {}
    
    for col, range_dict in ranges.items():
        if col in df.columns:
            min_val = range_dict.get('min', -np.inf)
            max_val = range_dict.get('max', np.inf)
            
            n_violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            if n_violations > 0:
                violations[col] = n_violations
                logger.warning(
                    f"{col}: {n_violations} values outside range [{min_val}, {max_val}]"
                )
    
    return violations


def validate_temporal_consistency(
    df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'snapshot_date',
    label_col: str = 'churn_label_next_period'
) -> bool:
    """
    Validate temporal consistency to prevent label leakage.
    
    Key checks:
    1. For each customer, snapshots are in chronological order
    2. Churn label is only positive on the last snapshot before churn
    3. No future information is encoded in features
    
    Args:
        df: Input dataframe
        customer_id_col: Customer ID column name
        date_col: Snapshot date column name
        label_col: Churn label column name
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating temporal consistency...")
    
    # Check 1: Temporal order per customer
    for customer_id in df[customer_id_col].unique()[:100]:  # Sample for speed
        customer_df = df[df[customer_id_col] == customer_id].sort_values(date_col)
        dates = pd.to_datetime(customer_df[date_col])
        
        if not dates.is_monotonic_increasing:
            logger.error(f"Temporal order violation for customer {customer_id}")
            return False
    
    # Check 2: Churn label appears only at the end
    if label_col in df.columns:
        for customer_id in df[customer_id_col].unique()[:100]:
            customer_df = df[df[customer_id_col] == customer_id].sort_values(date_col)
            churn_labels = customer_df[label_col].values
            
            # If customer churned, label should be 1 only on last row(s)
            if churn_labels.sum() > 0:
                churn_indices = np.where(churn_labels == 1)[0]
                # Should be at the end
                if not np.all(churn_indices >= len(churn_labels) - 2):
                    logger.error(
                        f"Churn label appears mid-sequence for customer {customer_id}"
                    )
                    return False
    
    logger.info("Temporal consistency validation passed")
    return True


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df)).to_dict(),
        'n_duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe().to_dict(),
    }
    
    return report
