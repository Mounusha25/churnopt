"""
Date and time utilities for temporal data processing.
"""

from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd


def generate_monthly_dates(
    start_date: str,
    end_date: str
) -> List[datetime]:
    """
    Generate list of monthly dates between start and end.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        List of datetime objects representing month starts
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    dates = pd.date_range(start=start, end=end, freq='MS')  # MS = month start
    return dates.to_list()


def get_months_between(date1: datetime, date2: datetime) -> int:
    """
    Calculate number of months between two dates.
    
    Args:
        date1: First date
        date2: Second date
        
    Returns:
        Number of months (can be negative if date2 < date1)
    """
    return (date2.year - date1.year) * 12 + (date2.month - date1.month)


def add_months(date: datetime, months: int) -> datetime:
    """
    Add specified number of months to a date.
    
    Args:
        date: Base date
        months: Number of months to add (can be negative)
        
    Returns:
        New date
    """
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return datetime(year, month, day)


def get_time_windows(
    reference_date: datetime,
    windows_months: List[int]
) -> List[Tuple[datetime, datetime]]:
    """
    Get time windows for rolling aggregations.
    
    Args:
        reference_date: Reference date (point-in-time)
        windows_months: List of window sizes in months (e.g., [3, 6, 12])
        
    Returns:
        List of (start_date, end_date) tuples for each window
    """
    windows = []
    for months in windows_months:
        start_date = add_months(reference_date, -months)
        windows.append((start_date, reference_date))
    return windows


def validate_temporal_order(df: pd.DataFrame, date_col: str = 'snapshot_date') -> bool:
    """
    Validate that data is in temporal order.
    
    This is critical for ensuring point-in-time correctness.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column {date_col} not found in dataframe")
    
    dates = pd.to_datetime(df[date_col])
    return dates.is_monotonic_increasing or dates.is_monotonic_decreasing
