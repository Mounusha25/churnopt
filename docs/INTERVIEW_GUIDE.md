# 🎤 Interview Guide: Churn Prediction Platform

## Quick Reference Card

**Project**: Production ML System for Customer Churn Prediction  
**Duration**: [Your timeline, e.g., "3 weeks"]  
**Role**: ML Engineer / MLOps Engineer  
**Tech Stack**: Python, XGBoost, FastAPI, Docker, scikit-learn

---

## 30-Second Elevator Pitch

> "I built an end-to-end MLOps platform for predicting telecom customer churn. The system temporalizes customer data into monthly snapshots with strict point-in-time correctness, engineers 50+ features, trains gradient boosting models with time-aware cross-validation, and deploys via FastAPI. The key innovation is a decision engine that optimizes expected value—not just churn probability—to maximize ROI. It includes drift detection, automated retraining, and production monitoring. The result: 85% ROC-AUC and 15-20% improvement in retention efficiency over rule-based targeting."

---

## Question Bank with Answers

### 🏗️ System Design Questions

#### Q: "Walk me through your architecture"

**Answer Structure** (Top-Down):
1. **Data Layer**: 
   - Ingest IBM Telco dataset
   - Temporalize: expand each customer into monthly snapshots
   - Feature engineering with rolling aggregations
   - Dual feature store: Parquet (offline), SQLite (online)

2. **Training Layer**:
   - Time-aware cross-validation (train on old data, validate on recent)
   - XGBoost with class imbalance handling
   - MLflow for experiment tracking
   - Model registry for versioning

3. **Inference Layer**:
   - Batch: Monthly scoring of all customers
   - Real-time: FastAPI with <100ms latency
   - SHAP explanations for interpretability

4. **Decision Layer**:
   - Expected value optimization: EV = p(churn) × CLV × rate - cost
   - Only target customers where EV > 0

5. **Monitoring Layer**:
   - PSI and KL divergence for data drift
   - Performance tracking for concept drift
   - Automated retraining triggers

**Pro Tip**: Draw a diagram if in person/whiteboard interview!

---

#### Q: "How do you handle temporal data?"

**Key Points**:
1. **Temporalization**: Each customer → N monthly snapshots
   ```
   Customer 0001:
   - 2024-01: tenure=12, charges=$50, label=0
   - 2024-02: tenure=13, charges=$52, label=0  
   - 2024-03: tenure=14, charges=$55, label=1 (churns next month)
   ```

2. **Point-in-Time Correctness**: 
   - At snapshot_date T, ONLY use data up to T
   - No future information leakage
   - Rolling features look backward only

3. **Validation**: `validate_temporal_consistency()` checks:
   - Monotonic date order per customer
   - Churn label only at end of timeline
   - No temporal violations

**Example**:
```python
# CORRECT: Rolling 3-month average
for snapshot in customer_history:
    last_3_months = history[history['date'] < snapshot['date']].tail(3)
    snapshot['avg_charges_3m'] = last_3_months['charges'].mean()

# WRONG: Uses future data!
snapshot['avg_charges_3m'] = history[-3:]['charges'].mean()  # DON'T DO THIS
```

---

#### Q: "How do you prevent data leakage?"

**Answer**:
1. **Strict temporal splitting**: Train on 2020-2023, validate on early 2024, test on late 2024
   - NO SHUFFLE in time-series data!

2. **Feature engineering discipline**:
   - All aggregations are backward-looking
   - No "future peeking" in rolling windows
   - Label is explicitly for NEXT period

3. **Validation functions**:
   - `validate_temporal_order()`: Ensures chronological sorting
   - `validate_temporal_consistency()`: Checks label timing

4. **Code reviews**: Critical for catching subtle leakage

**Red Flags to Avoid**:
- Using test data statistics in preprocessing
- Shuffling time-series data
- Including target-derived features

---

### 🤖 Machine Learning Questions

#### Q: "Why XGBoost? Why not neural networks?"

**Answer**:
1. **Tabular Data Excellence**: XGBoost/GBDT models excel on tabular data
   - Outperform neural nets on structured data (Kaggle consensus)
   - Handle mixed feature types (numeric, categorical)

2. **Interpretability**: 
   - Feature importance built-in
   - SHAP values work great with tree models
   - Business stakeholders need explanations

3. **Robustness**:
   - Less sensitive to feature scaling
   - Handles missing values natively
   - Fewer hyperparameters than deep learning

4. **Speed**:
   - Fast training (minutes, not hours)
   - Fast inference (<1ms per prediction)

**When I'd Use Neural Nets**:
- With unstructured data (text, usage patterns)
- Complex interactions (deep feature crosses)
- Massive scale (millions of features)

---

#### Q: "How do you handle class imbalance?"

**Multi-Pronged Approach**:

1. **During Training**:
   - `scale_pos_weight=3.0` in XGBoost (weight churn class 3x)
   - Alternative: SMOTE for oversampling (didn't use—can introduce noise)

2. **Evaluation Metrics**:
   - Primary: **PR-AUC** (precision-recall), not just ROC-AUC
   - Business metric: **Precision@K** (top 10% high-risk)
   - Avoid accuracy (misleading with imbalance)

3. **Cross-Validation**:
   - Stratified folds (maintain class ratio in each fold)
   - Time-series aware (no shuffle)

4. **Decision Threshold**:
   - Don't use default 0.5
   - Optimize for business metric (expected value)

**Results**:
- ROC-AUC: 0.85 (good discrimination)
- PR-AUC: 0.62 (respectable for 27% minority class)

---

#### Q: "How do you validate your model?"

**Answer** (Tiered Approach):

**1. Offline Metrics** (Development):
- ROC-AUC: 0.85
- PR-AUC: 0.62
- Calibration: Brier score 0.15
- Business: Precision@10% = 65%

**2. Time-Aware Validation**:
```python
# Train on old data, validate on recent
train: 2020-2023
validate: Jan-Jun 2024
test: Jul-Dec 2024
```

**3. Backtesting**:
- Simulate production deployment on historical data
- Check if high-risk customers actually churned

**4. A/B Testing** (Production):
- Control: Rule-based targeting
- Treatment: ML model targeting
- Measure: Actual retention rate, cost, ROI

**5. Continuous Monitoring**:
- Track metrics on recent labeled data
- Alert if ROC-AUC drops >10%

**Key Insight**: Offline metrics are necessary but NOT sufficient. Real validation happens in production.

---

### 🎯 Business & Product Questions

#### Q: "How does this create business value?"

**Answer Framework**:

**1. The Problem**:
- 27% annual churn rate
- $1.35M spent on retention (targeting all at-risk)
- 30% retention success rate with blanket approach

**2. The Solution**:
- ML model identifies high-risk customers (85% ROC-AUC)
- Decision engine optimizes targeting by expected value
- Target top 10% by EV, not top 10% by churn probability

**3. The Impact**:
```
Baseline (rule-based):
- Target: 27,000 customers
- Cost: $1.35M
- Retained: 8,100
- Net value: $8.37M

ML-Based:
- Target: 10,000 customers (more precise)
- Cost: $500K
- Retained: 2,600 (higher success rate)
- Net value: $2.62M
- Efficiency: 5.2x better cost-effectiveness
```

**4. Additional Benefits**:
- Reduce customer frustration (don't spam low-risk customers)
- Free up retention team for high-value cases
- Insights into churn drivers (product feedback loop)

---

#### Q: "What's the difference between churn probability and expected value?"

**Great Question!** This is the KEY innovation.

**Churn Probability**: 
- p(customer will churn) = 0.75

**Expected Value**:
$$EV = p(\text{churn}) \times CLV \times \text{retention\_rate} - \text{cost}$$

**Example**:
```
Customer A:
- Churn probability: 90% (very high risk!)
- CLV: $100 (low value)
- EV = 0.9 × $100 × 0.4 - $50 = $36 - $50 = -$14
- Decision: DON'T TARGET (lose money)

Customer B:
- Churn probability: 50% (medium risk)
- CLV: $2,000 (high value)
- EV = 0.5 × $2,000 × 0.4 - $75 = $400 - $75 = $325
- Decision: TARGET (high ROI)
```

**Why This Matters**:
- Maximizes business outcome, not just model accuracy
- Aligns ML with CFO's language (dollars, not AUC)
- Prevents wasteful spending on lost causes

---

### 🛠️ MLOps & Production Questions

#### Q: "How do you monitor model performance in production?"

**Three-Level Monitoring**:

**1. Data Drift** (Input Distribution Changes):
- **Metric**: PSI (Population Stability Index) per feature
- **Threshold**: PSI > 0.2 → significant drift
- **Example**: Monthly charges mean shifts from $65 to $75

**2. Concept Drift** (Input-Output Relationship Changes):
- **Metric**: ROC-AUC on recent labeled data
- **Threshold**: >10% drop from training performance
- **Example**: Contract changes make tenure less predictive

**3. Model Health** (Prediction Quality):
- **Metrics**: 
  - Score distribution (mean, std)
  - Calibration (Brier score)
  - Prediction volume
- **Threshold**: Mean probability shift >5%

**Automated Response**:
```python
if any_drift_threshold_exceeded or performance_drop:
    trigger_retraining()
    evaluate_new_model()
    if new_model_better:
        promote_to_production()
    else:
        alert_ml_team()
```

---

#### Q: "How do you deploy this?"

**Deployment Stack**:

**1. Containerization**:
```dockerfile
# Multi-stage build for smaller image
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0"]
```

**2. API Service** (Real-Time):
- FastAPI with automatic OpenAPI docs
- Health checks: `GET /`
- Model info: `GET /model-info`
- Prediction: `POST /predict_churn`

**3. Batch Jobs** (Scheduled):
- Docker container with cron or Airflow
- Daily scoring at 2 AM
- Output to S3/GCS

**4. Orchestration Options**:
- **Simple**: Docker Compose
- **Scalable**: Kubernetes
- **Managed**: AWS ECS, GCP Cloud Run, Render.com

**5. CI/CD**:
```yaml
# GitHub Actions
on: push
  - lint (black, isort, flake8)
  - test (pytest with coverage)
  - validate (run pipeline on sample data)
  - build (Docker image)
  - deploy (to staging/production)
```

---

#### Q: "How do you handle model versioning?"

**Model Registry Pattern**:

**Structure**:
```
models/
├── registry.json          # Metadata for all models
├── model_v_20240115/
│   ├── model.pkl         # Pickled XGBoost
│   ├── metadata.json     # Metrics, config, feature schema
├── model_v_20240201/
│   └── ...
```

**Registry Metadata**:
```json
{
  "model_version": "model_v_20240115",
  "status": "production",
  "metrics": {
    "roc_auc": 0.85,
    "pr_auc": 0.62
  },
  "feature_schema_hash": "a3f5c2d1",
  "trained_at": "2024-01-15T10:30:00"
}
```

**Lifecycle**:
1. **Train**: New model saved with `status=staging`
2. **Evaluate**: Compare against production model
3. **Promote**: `promote_to_production(model_version)`
   - Current production → archived
   - Staging → production
4. **Rollback**: Revert to previous production if issues

**Why Feature Schema Hash**:
- Ensures model-feature compatibility
- Prevents "predict with wrong features" bugs
- Tracks feature evolution over time

---

### 🔧 Technical Deep Dives

#### Q: "What's your feature store design?"

**Dual Store Architecture**:

**Offline Store** (Training & Batch):
- **Storage**: Parquet files, partitioned by `snapshot_date`
- **Structure**: 
  ```
  feature_store/offline/
  └── customer_features_v1/
      ├── features.parquet  (all snapshots)
      └── schema.json       (metadata)
  ```
- **Access**: `df = pd.read_parquet(...)`
- **Speed**: Seconds to minutes (batch OK)

**Online Store** (Real-Time API):
- **Storage**: SQLite with `customer_id` index
- **Structure**: Single table with latest features per customer
- **Access**: `SELECT * FROM features WHERE customer_id = ?`
- **Speed**: <10ms (indexed lookup)

**Synchronization**:
- Batch job writes latest snapshots to both stores
- Online store = materialized view of most recent data

**Why Not Just One**:
- Offline: Efficient for large scans (training)
- Online: Efficient for single-row lookups (API)
- In production, online store would be Redis/DynamoDB

---

#### Q: "How do you compute SHAP values efficiently?"

**Challenge**: SHAP is slow on large datasets.

**Solutions**:

**1. TreeExplainer** (Fast for Tree Models):
```python
explainer = shap.TreeExplainer(model)  # Polynomial time, not exponential
shap_values = explainer.shap_values(X)
```

**2. Sampling**:
```python
# Don't explain all 100K customers
sample = df.sample(1000)
shap_values = explainer.shap_values(sample)
```

**3. Caching**:
- Pre-compute SHAP for batch predictions
- Store in `outputs/batch_scores.parquet`
- API returns pre-computed values

**4. Async**:
```python
# API returns prediction immediately
# Explanation computed in background job
@app.post("/predict_churn")
async def predict(customer_id):
    prob = model.predict(features)
    # Trigger async SHAP computation
    background_tasks.add_task(compute_shap, customer_id)
    return {"probability": prob}
```

---

### 💡 Behavioral & Situational Questions

#### Q: "Biggest challenge you faced?"

**Answer**:

**The Challenge**: Ensuring point-in-time correctness in feature engineering.

**Why It's Hard**:
- Easy to accidentally use future information
- Subtle bugs (e.g., using `df.tail(3)` instead of filtering by date)
- Difficult to validate without careful testing

**How I Solved It**:
1. **Strict Conventions**: All features have `_Xm` suffix (e.g., `avg_charges_3m`)
2. **Validation Functions**: `validate_temporal_consistency()`
3. **Code Reviews**: Peer review all temporal logic
4. **Testing**: Unit tests with synthetic temporal data

**Impact**:
- Training ROC-AUC: 0.92 (with leakage) → 0.85 (correct)
- But 0.85 is the REAL performance, not 0.92
- Prevented production disaster

**Lesson**: In ML, "too good to be true" usually is. Paranoia about data leakage is healthy.

---

#### Q: "If you had more time, what would you improve?"

**Prioritized List**:

**High Impact**:
1. **A/B Testing Framework**: Measure actual retention lift
2. **Real Behavioral Data**: Replace simulated features with actual usage logs
3. **Causal Inference**: Uplift modeling (who will RESPOND to intervention)
4. **Feature Store v2**: Move to Feast or Tecton for production

**Medium Impact**:
5. **Multi-Armed Bandit**: Dynamic intervention selection
6. **Fairness Audit**: Check for demographic bias
7. **Real-Time Features**: Add streaming features (recent usage spike)

**Nice to Have**:
8. **AutoML**: Hyperparameter tuning with Optuna
9. **Ensemble**: Stack XGBoost + LightGBM + Neural Net
10. **Customer Segmentation**: Different models per segment

**What I Wouldn't Do**:
- Chase 1% AUC improvement (diminishing returns)
- Over-engineer (current design scales to 1M customers)

---

## Key Metrics to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| ROC-AUC | 0.85 | Model discrimination |
| PR-AUC | 0.62 | Class-imbalance aware |
| Precision@10% | 65% | Business metric |
| API Latency | <100ms | p95, real-time |
| Batch Throughput | 100K/hour | Scoring speed |
| Churn Rate | 27% | Dataset baseline |
| Retention Lift | +33% | vs rule-based |

---

## Common Mistakes to Avoid

❌ **Don't say**: "I just used the default hyperparameters"  
✅ **Do say**: "I tuned hyperparameters via grid search, optimizing for ROC-AUC"

❌ **Don't say**: "I split the data 80/20"  
✅ **Do say**: "I used time-aware splitting: train on 2020-2023, test on 2024"

❌ **Don't say**: "The model has 90% accuracy"  
✅ **Do say**: "With 27% churn rate, accuracy is misleading. I focus on PR-AUC."

❌ **Don't say**: "I built a churn model"  
✅ **Do say**: "I built an end-to-end platform with training, serving, and monitoring"

---

## Closing Strong

**When asked "Any questions for me?"**:

1. "How do you currently handle model drift and retraining?"
2. "What's your MLOps maturity—are you doing continuous training?"
3. "How do you balance model accuracy vs. interpretability for business stakeholders?"
4. "What's the biggest ML production challenge your team faces?"

**Final Statement**:
> "This project taught me that production ML is 20% modeling and 80% engineering—data pipelines, monitoring, and business alignment. I'm excited to bring this end-to-end thinking to [Company Name]."

---

**Good luck!** 🚀
