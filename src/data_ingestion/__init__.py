"""
Data ingestion module.

Handles loading, cleaning, and temporalizing the Telco dataset.
"""

from .loader import TelcoDataLoader, run_data_ingestion
from .temporal_builder import TemporalDatasetBuilder

__all__ = [
    'TelcoDataLoader',
    'TemporalDatasetBuilder',
    'run_data_ingestion',
]
