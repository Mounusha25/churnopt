"""
Sample unit tests for the temporal dataset builder.

These demonstrate testing patterns for temporal ML systems.
Run with: pytest tests/test_data_ingestion/test_temporal_builder.py -v
"""

import pytest
import pandas as pd
from datetime import datetime

from src.data_ingestion.temporal_builder import TemporalDatasetBuilder


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'temporal': {
            'snapshot_start_date': '2020-01-01',
            'snapshot_end_date': '2020-06-01',
            'snapshot_frequency': 'MS',
            'label_window_days': 30,
            'min_history_months': 3
        }
    }


@pytest.fixture
def sample_customers():
    """Sample customer data for testing."""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'tenure': [12, 6, 24],
        'MonthlyCharges': [50.0, 70.0, 90.0],
        'TotalCharges': [600.0, 420.0, 2160.0],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer'],
        'Churn': ['Yes', 'No', 'No']
    })


class TestTemporalDatasetBuilder:
    """Test suite for TemporalDatasetBuilder."""
    
    def test_initialization(self, sample_config):
        """Test builder initialization."""
        builder = TemporalDatasetBuilder(sample_config)
        assert builder.snapshot_start_date == pd.Timestamp('2020-01-01')
        assert builder.snapshot_end_date == pd.Timestamp('2020-06-01')
    
    def test_expand_customer_creates_snapshots(self, sample_config, sample_customers):
        """Test that customers are expanded into monthly snapshots."""
        builder = TemporalDatasetBuilder(sample_config)
        customer = sample_customers.iloc[0]
        
        snapshots = builder.expand_customer_to_snapshots(customer)
        
        # Should create multiple snapshots
        assert len(snapshots) > 0
        
        # Each snapshot should have snapshot_date
        assert all('snapshot_date' in s for s in snapshots)
        
        # Tenure should increase over time
        tenures = [s['tenure'] for s in snapshots]
        assert tenures == sorted(tenures)
    
    def test_churn_label_timing(self, sample_config, sample_customers):
        """Test that churn labels appear only in the correct time window."""
        builder = TemporalDatasetBuilder(sample_config)
        customer = sample_customers[sample_customers['Churn'] == 'Yes'].iloc[0]
        
        snapshots = builder.expand_customer_to_snapshots(customer)
        
        # Count how many snapshots have churn=1
        churn_counts = sum(s['churn'] == 1 for s in snapshots)
        
        # Should have at least one churn label
        assert churn_counts >= 1
        
        # All earlier snapshots should have churn=0
        churn_idx = next(i for i, s in enumerate(snapshots) if s['churn'] == 1)
        for i in range(churn_idx):
            assert snapshots[i]['churn'] == 0
    
    def test_no_future_leakage(self, sample_config, sample_customers):
        """Test that features don't leak future information."""
        builder = TemporalDatasetBuilder(sample_config)
        
        # Customer with 12 months tenure
        customer = sample_customers.iloc[0]
        snapshots = builder.expand_customer_to_snapshots(customer)
        
        # Check tenure progression is logical
        for snap in snapshots:
            # Tenure at snapshot should be <= original tenure
            assert snap['tenure'] <= customer['tenure']
            
            # TotalCharges should be proportional to tenure
            expected_total = snap['tenure'] * customer['MonthlyCharges']
            # Allow 10% tolerance
            assert abs(snap['TotalCharges'] - expected_total) < expected_total * 0.1
    
    def test_min_history_filter(self, sample_config, sample_customers):
        """Test that customers with insufficient history are filtered."""
        # Set high minimum history
        config_high_min = sample_config.copy()
        config_high_min['temporal']['min_history_months'] = 100
        
        builder = TemporalDatasetBuilder(config_high_min)
        result = builder.run(sample_customers)
        
        # Should filter out customers with < 100 months tenure
        assert len(result['temporal']) == 0 or result['temporal']['tenure'].min() >= 100
    
    def test_temporal_consistency(self, sample_config, sample_customers):
        """Test temporal consistency - snapshot_date >= customer start."""
        builder = TemporalDatasetBuilder(sample_config)
        result = builder.run(sample_customers)
        
        df = result['temporal']
        
        # All snapshot dates should be within configured range
        assert df['snapshot_date'].min() >= builder.snapshot_start_date
        assert df['snapshot_date'].max() <= builder.snapshot_end_date
        
        # Each customer's snapshots should be in chronological order
        for customer_id in df['customerID'].unique():
            customer_data = df[df['customerID'] == customer_id].sort_values('snapshot_date')
            
            # Dates should be increasing
            dates = customer_data['snapshot_date'].values
            assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
            
            # Tenure should be increasing
            tenure = customer_data['tenure'].values
            assert all(tenure[i] <= tenure[i+1] for i in range(len(tenure)-1))


class TestTemporalValidation:
    """Tests for temporal validation utilities."""
    
    def test_no_duplicate_snapshots(self, sample_config, sample_customers):
        """Test that we don't create duplicate (customer, date) pairs."""
        builder = TemporalDatasetBuilder(sample_config)
        result = builder.run(sample_customers)
        
        df = result['temporal']
        
        # Check for duplicates
        duplicates = df.groupby(['customerID', 'snapshot_date']).size()
        assert (duplicates == 1).all(), "Found duplicate (customer, snapshot_date) pairs"
    
    def test_churn_label_validity(self, sample_config, sample_customers):
        """Test that churn labels are valid (0 or 1)."""
        builder = TemporalDatasetBuilder(sample_config)
        result = builder.run(sample_customers)
        
        df = result['temporal']
        
        # Churn should only be 0 or 1
        assert df['churn'].isin([0, 1]).all()
        
        # Churned customers should have at least one churn=1
        churned_ids = sample_customers[sample_customers['Churn'] == 'Yes']['customerID'].values
        for cid in churned_ids:
            customer_data = df[df['customerID'] == cid]
            if len(customer_data) > 0:
                assert customer_data['churn'].max() == 1


@pytest.mark.parametrize("tenure,expected_min_snapshots", [
    (3, 1),   # 3 months tenure -> at least 1 snapshot
    (6, 4),   # 6 months tenure -> at least 4 snapshots
    (12, 10), # 12 months tenure -> at least 10 snapshots
])
def test_snapshot_count_by_tenure(sample_config, tenure, expected_min_snapshots):
    """Test that longer tenure results in more snapshots."""
    builder = TemporalDatasetBuilder(sample_config)
    
    customer = pd.Series({
        'customerID': 'TEST',
        'tenure': tenure,
        'MonthlyCharges': 50.0,
        'TotalCharges': tenure * 50.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'Churn': 'No'
    })
    
    snapshots = builder.expand_customer_to_snapshots(customer)
    
    assert len(snapshots) >= expected_min_snapshots


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
