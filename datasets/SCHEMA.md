# Data Schema Documentation

## Overview

This document describes the data schema for the Customer Churn Prediction Platform, including the temporal dataset structure and feature definitions.

## 📊 Temporal Snapshots Schema

**File**: `datasets/monthly_customer_snapshots.parquet`

### Core Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `customer_id` | string | Unique customer identifier | "7590-VHVEG" |
| `snapshot_date` | datetime | Month-start date of snapshot | 2024-01-01 |
| `churn_label_next_period` | int | Binary label: 1 if churned next month, 0 otherwise | 0 |
| `tenure_months` | int | Months customer has been active at snapshot date | 12 |

### Demographic Features

| Column | Type | Description |
|--------|------|-------------|
| `gender_male` | int | 1 if male, 0 if female |
| `senior_citizen` | int | 1 if age >= 65, 0 otherwise |
| `partner` | int | 1 if has partner, 0 otherwise |
| `dependents` | int | 1 if has dependents, 0 otherwise |

### Service Features

| Column | Type | Description |
|--------|------|-------------|
| `phone_service` | int | Has phone service |
| `multiple_lines` | int | Has multiple phone lines |
| `internet_service` | string | Type: "DSL", "Fiber optic", "No" |
| `online_security` | int | Has online security add-on |
| `online_backup` | int | Has online backup add-on |
| `device_protection` | int | Has device protection add-on |
| `tech_support` | int | Has tech support add-on |
| `streaming_tv` | int | Has streaming TV service |
| `streaming_movies` | int | Has streaming movie service |
| `num_services_total` | int | Total number of services subscribed |

### Contract Features

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `contract` | string | Contract type | "Month-to-month", "One year", "Two year" |
| `is_month_to_month` | int | Binary indicator for MTM contract | 0 or 1 |
| `is_one_year` | int | Binary indicator for 1-year contract | 0 or 1 |
| `is_two_year` | int | Binary indicator for 2-year contract | 0 or 1 |
| `contract_remaining_months` | int | Estimated months left in contract | 0-24 |
| `paperless_billing` | int | Uses paperless billing | 0 or 1 |
| `payment_method` | string | Payment method type | "Electronic check", "Credit card", etc. |
| `payment_electronic_check` | int | Uses electronic check | 0 or 1 |
| `payment_auto` | int | Uses automatic payment | 0 or 1 |

### Financial Features

| Column | Type | Description |
|--------|------|-------------|
| `monthly_charges` | float | Monthly bill amount (USD) |
| `total_charges` | float | Total charges to date (USD) |
| `avg_monthly_charges_3m` | float | Average monthly charges over past 3 months |
| `avg_monthly_charges_6m` | float | Average monthly charges over past 6 months |
| `charges_trend_6m` | float | Linear trend in charges (slope) over 6 months |
| `std_monthly_charges_3m` | float | Standard deviation of charges over 3 months |

### Temporal Features

| Column | Type | Description |
|--------|------|-------------|
| `tenure_bucket` | string | Categorized tenure: "0-6m", "6-12m", "12-24m", "24-48m", "48m+" |
| `tenure_bucket_num` | int | Numeric encoding of tenure bucket (0-4) |

### Behavioral Features (Simulated)

| Column | Type | Description |
|--------|------|-------------|
| `num_support_tickets_3m` | int | Number of support tickets in past 3 months |
| `num_late_payments_6m` | int | Number of late payments in past 6 months |
| `num_plan_changes_6m` | int | Number of plan changes in past 6 months |
| `days_since_last_contact` | int | Days since last customer contact |

### Derived Features

| Column | Type | Description |
|--------|------|-------------|
| `has_fiber_optic` | int | Has fiber optic internet |
| `has_dsl` | int | Has DSL internet |
| `no_internet` | int | No internet service |
| `charges_per_tenure_month` | float | Average monthly spend per tenure month |
| `services_per_month` | float | Services adopted per tenure month |
| `is_high_value` | int | In top 25% of monthly charges |

### Metadata

| Column | Type | Description |
|--------|------|-------------|
| `months_until_churn` | int | Months until actual churn (for churned customers) |
| `year_month` | string | YYYY-MM partition key |

---

## 📋 Customer Master Schema

**File**: `datasets/customers_master.csv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique customer identifier |
| `first_snapshot_date` | datetime | First observed month |
| `last_snapshot_date` | datetime | Last observed month |
| `max_tenure` | int | Maximum tenure observed |
| `churned` | int | 1 if customer churned, 0 if still active |
| `avg_monthly_charges` | float | Average monthly charges across all snapshots |
| `total_charges` | float | Total charges |
| `gender_male` | int | Gender indicator |
| `senior_citizen` | int | Senior citizen indicator |
| `partner` | int | Has partner |
| `dependents` | int | Has dependents |

---

## 🎯 Point-in-Time Correctness

### Critical Principle

**At snapshot_date T, only information available up to T can be used.**

### Enforcement Mechanisms

1. **Temporal Sorting**: All data sorted by `customer_id`, `snapshot_date`
2. **Rolling Windows**: Look backward only (e.g., avg_charges_3m uses previous 3 months)
3. **Label Timing**: `churn_label_next_period` predicts NEXT month, not current
4. **Validation**: `validate_temporal_consistency()` checks for violations

### Example

For `customer_id = "0001"` at `snapshot_date = 2024-06-01`:
- ✅ Can use: `monthly_charges` from Jan-May 2024
- ✅ Can use: `tenure_months` as of June 1, 2024
- ❌ Cannot use: Any data from July 2024 onwards
- ❌ Cannot use: Final churn status (this is the label!)

---

## 📊 Data Statistics

### Dataset Size
- **Temporal snapshots**: ~170,000 rows (7,043 customers × ~24 months avg)
- **Unique customers**: 7,043
- **Time range**: 2020-01-01 to 2024-12-31
- **Churn rate**: ~27% (varies by month)

### Class Distribution
- Negative samples (no churn): ~73%
- Positive samples (churn): ~27%

### Missing Values
- All critical fields have 0% missing after cleaning
- Optional fields may have NaN (filled with 0 or imputed)

### Feature Distributions

**Tenure**:
- Min: 0 months
- Max: 72 months
- Mean: 32 months
- Median: 29 months

**Monthly Charges**:
- Min: $18.25
- Max: $118.75
- Mean: $64.76
- Median: $70.35

---

## 🔄 Feature Updates

### Version History

**v1** (Current):
- Initial temporal dataset
- 50+ engineered features
- Simulated behavioral features

**Future Versions**:
- v2: Add real usage events
- v3: Add customer interactions
- v4: Add external data (economic indicators)

### Feature Schema Hashing

Each feature set version has a unique hash:
```python
from src.utils import compute_feature_schema_hash
hash = compute_feature_schema_hash(feature_list)
# Example: "a3f5c2d1"
```

This ensures model-feature compatibility during inference.

---

## 🔍 Data Quality Checks

### Automated Validations

1. **No duplicates**: Unique (customer_id, snapshot_date) pairs
2. **Temporal order**: Snapshots in chronological order per customer
3. **Label consistency**: Churn label only appears at end of customer timeline
4. **Value ranges**: 
   - `tenure_months >= 0`
   - `monthly_charges > 0`
   - `churn_label_next_period in {0, 1}`
5. **Missing values**: < 5% missing in any column

### Manual Checks

Review distribution shifts:
```python
from src.utils import get_data_quality_report
report = get_data_quality_report(df)
```

---

## 📝 Usage Examples

### Load Temporal Dataset

```python
import pandas as pd

df = pd.read_parquet('datasets/monthly_customer_snapshots.parquet')
print(f"Loaded {len(df)} snapshots for {df['customer_id'].nunique()} customers")
```

### Filter by Date Range

```python
train_df = df[
    (df['snapshot_date'] >= '2020-01-01') &
    (df['snapshot_date'] <= '2023-12-31')
]
```

### Get Features for Training

```python
from src.feature_engineering import get_feature_names

feature_cols = get_feature_names(df, exclude_meta=True)
X = df[feature_cols]
y = df['churn_label_next_period']
```

---

## 🛡️ Data Privacy & Compliance

- All `customer_id` values are anonymized
- No PII (personally identifiable information) included
- Synthetic data used for behavioral features
- Production deployment should implement:
  - Data encryption at rest and in transit
  - Access controls and audit logs
  - GDPR/CCPA compliance for customer data

---

**Last Updated**: January 2026  
**Schema Version**: v1  
**Maintained By**: ML Engineering Team
