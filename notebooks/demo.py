"""
Example Jupyter notebook demonstrating the churn prediction platform.

This notebook walks through:
1. Data exploration
2. Feature engineering
3. Model training
4. Evaluation
5. Decision making

Save this as: notebooks/demo.ipynb (convert from .py using jupytext or similar)
"""

# %% [markdown]
# # Customer Churn Prediction Platform - Demo
# 
# This notebook demonstrates the end-to-end churn prediction system.

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project to path
sys.path.append('..')

from src.utils import load_config
from src.data_ingestion import TelcoDataLoader, TemporalDatasetBuilder
from src.feature_engineering import FeatureEngineer, FeatureStore
from src.training_pipeline import ChurnModelTrainer
from src.models import ModelRegistry

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# %% [markdown]
# ## 2. Load and Explore Data

# %%
# Load raw data
data_config = load_config('../configs/data.yaml')
loader = TelcoDataLoader(data_config)
df = loader.run()

print(f"Shape: {df.shape}")
print(f"\nChurn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
df.head()

# %%
# Visualize churn distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Churn by contract type
df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()).plot(
    kind='bar', ax=axes[0], color='steelblue'
)
axes[0].set_title('Churn Rate by Contract Type')
axes[0].set_ylabel('Churn Rate')

# Churn by tenure
df.groupby(pd.cut(df['tenure'], bins=[0, 12, 24, 48, 100]))['Churn'].apply(
    lambda x: (x=='Yes').mean()
).plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Churn Rate by Tenure')
axes[1].set_ylabel('Churn Rate')

# Monthly charges distribution
df[df['Churn']=='Yes']['MonthlyCharges'].hist(ax=axes[2], alpha=0.5, label='Churned', bins=30)
df[df['Churn']=='No']['MonthlyCharges'].hist(ax=axes[2], alpha=0.5, label='Retained', bins=30)
axes[2].set_title('Monthly Charges Distribution')
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Build Temporal Dataset

# %%
# Create temporal snapshots
temporal_builder = TemporalDatasetBuilder(data_config)
datasets = temporal_builder.run(df)
df_temporal = datasets['temporal']

print(f"Temporal snapshots: {len(df_temporal):,}")
print(f"Unique customers: {df_temporal['customerID'].nunique():,}")
print(f"Date range: {df_temporal['snapshot_date'].min()} to {df_temporal['snapshot_date'].max()}")

# %% [markdown]
# ## 4. Engineer Features

# %%
# Generate features
features_config = load_config('../configs/features.yaml')
feature_engineer = FeatureEngineer(features_config)
df_features = feature_engineer.run(df_temporal)

print(f"Total features: {len(df_features.columns)}")
print(f"\nSample features:")
print(df_features.head())

# %%
# Feature importance preview (using correlation with target)
numeric_features = df_features.select_dtypes(include=[np.number]).columns
correlations = df_features[numeric_features].corrwith(df_features['churn']).abs()
top_features = correlations.nlargest(15)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='steelblue')
plt.title('Top 15 Features by Correlation with Churn')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Train Model

# %%
# Train churn model
training_config = load_config('../configs/training.yaml')
trainer = ChurnModelTrainer(training_config)
result = trainer.run(df_features)

print(f"\n{'='*60}")
print("Model Training Results")
print(f"{'='*60}")
print(f"Model Version: {result['model_version']}")
print(f"Algorithm: {result['algorithm']}")
print(f"\nPerformance Metrics:")
for metric, value in result['metrics'].items():
    print(f"  {metric}: {value:.4f}")
print(f"{'='*60}")

# %% [markdown]
# ## 6. Evaluate Model

# %%
# Load production model
registry = ModelRegistry()
prod_model_info = registry.get_production_model()

if prod_model_info:
    print(f"Production Model: {prod_model_info['version']}")
    print(f"ROC-AUC: {prod_model_info['metrics']['roc_auc']:.4f}")
    
    # Feature importance
    import joblib
    model = joblib.load(prod_model_info['path'])
    
    feature_importance = pd.DataFrame({
        'feature': result['feature_names'][:len(model.feature_importances_)],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Top 20 Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Make Predictions & Decisions

# %%
# Run batch inference
from src.inference import BatchInference
from src.decision_engine import DecisionEngine

inference_config = load_config('../configs/inference.yaml')
batch_inference = BatchInference(inference_config)
scores = batch_inference.run()

print(f"Scored {len(scores):,} customers")
print(f"\nRisk distribution:")
print(scores['risk_segment'].value_counts())

# %%
# Apply decision engine
decision_config = load_config('../configs/decision.yaml')
decision_engine = DecisionEngine(decision_config)

scores_path = '../outputs/latest_scores.parquet'
scores.to_parquet(scores_path, index=False)

targeted = decision_engine.run(scores_path)

print(f"\n{'='*60}")
print("Decision Engine Results")
print(f"{'='*60}")
print(f"Total customers scored: {len(scores):,}")
print(f"Customers targeted: {len(targeted):,}")
print(f"Expected total value: ${targeted['expected_value'].sum():,.2f}")
print(f"\nRecommended actions:")
print(targeted['recommended_action'].value_counts())
print(f"{'='*60}")

# %%
# Visualize targeting decisions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Expected value by segment
targeted.groupby('segment')['expected_value'].sum().plot(
    kind='bar', ax=axes[0], color='green'
)
axes[0].set_title('Expected Value by Customer Segment')
axes[0].set_ylabel('Total Expected Value ($)')

# Churn probability distribution for targeted customers
axes[1].hist(targeted['churn_probability'], bins=30, color='coral', edgecolor='black')
axes[1].set_title('Churn Probability Distribution (Targeted Customers)')
axes[1].set_xlabel('Churn Probability')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Summary
# 
# We've demonstrated:
# - ✅ Data loading and exploration
# - ✅ Temporal dataset construction with point-in-time correctness
# - ✅ Feature engineering with rolling aggregations
# - ✅ Model training with time-aware validation
# - ✅ Decision making with expected value optimization
# 
# **Next Steps:**
# - Deploy the API for real-time predictions
# - Set up monitoring for drift detection
# - Configure retraining automation
# - Implement A/B testing framework

# %% [markdown]
# ---
# **Production Deployment:**
# ```bash
# # Start API server
# uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
# 
# # Run batch inference
# python -m src.inference.batch
# 
# # Monitor drift
# python -m src.monitoring.drift_detector
# ```
