"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Return config directory."""
    return project_root / "configs"


@pytest.fixture
def sample_telco_data():
    """Generate sample telco data for testing."""
    return pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(100)],
        'gender': ['Male', 'Female'] * 50,
        'SeniorCitizen': [0, 1] * 50,
        'Partner': ['Yes', 'No'] * 50,
        'Dependents': ['No', 'Yes'] * 50,
        'tenure': range(1, 101),
        'PhoneService': ['Yes'] * 100,
        'MultipleLines': ['No', 'Yes', 'No phone service'] * 33 + ['No'],
        'InternetService': ['DSL', 'Fiber optic', 'No'] * 33 + ['DSL'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'TechSupport': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'StreamingTV': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'] * 33 + ['Month-to-month'],
        'PaperlessBilling': ['Yes', 'No'] * 50,
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] * 25,
        'MonthlyCharges': [50.0 + i * 0.5 for i in range(100)],
        'TotalCharges': [50.0 * (i + 1) for i in range(100)],
        'Churn': ['Yes' if i % 4 == 0 else 'No' for i in range(100)]
    })


@pytest.fixture
def sample_temporal_data():
    """Generate sample temporal snapshot data."""
    dates = pd.date_range('2020-01-01', '2020-06-01', freq='MS')
    
    data = []
    for customer_id in [f'C{i:04d}' for i in range(10)]:
        for i, date in enumerate(dates):
            data.append({
                'customerID': customer_id,
                'snapshot_date': date,
                'tenure': i + 1,
                'MonthlyCharges': 50.0 + i * 2,
                'TotalCharges': 50.0 * (i + 1),
                'churn': 1 if i == len(dates) - 1 and customer_id in ['C0000', 'C0004'] else 0
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    return pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(50)],
        'snapshot_date': pd.Timestamp('2020-06-01'),
        'tenure': range(1, 51),
        'MonthlyCharges': [50.0 + i for i in range(50)],
        'avg_charges_3m': [50.0 + i * 1.1 for i in range(50)],
        'avg_charges_6m': [50.0 + i * 1.2 for i in range(50)],
        'charges_trend_3m': [0.05 * (i % 10) for i in range(50)],
        'tenure_months_categorical': ['0-12'] * 25 + ['13-24'] * 25,
        'total_services': range(1, 51),
        'churn': [1 if i % 5 == 0 else 0 for i in range(50)]
    })
