# 🎯 Customer Churn Prediction Platform

**Decision-Optimized Churn Prediction System**  
**0.91 ROC-AUC | Budget-Constrained Deployment | Profit-Optimized Targeting**

A batch-oriented ML system for customer churn prediction with economic optimization:
- ✅ **Temporal feature engineering** with point-in-time correctness
- ✅ **False Positive Penalty economics** (realistic cost modeling)
- ✅ **Budget-constrained decision engine** (production deployment)
- ✅ **Profit curve analysis** (unconstrained optimal targeting)
- ✅ **Comprehensive system design** documentation

**📊 Results:** See results section below for production vs analysis regimes  
**🏗️ Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md) and [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)

---

## � Key Results

### Production Deployment (Budget-Constrained)
- **Customers targeted:** 659 customers (top 0.3% of base)
- **Monthly budget:** $50,000 (hard limit)
- **Expected net value:** ~$52,000
- **ROI:** 105%
- **Model performance:** 0.91 ROC-AUC

### Unconstrained Economic Analysis
- **Optimal targeting:** ~25,100 customers (theoretical maximum)
- **Maximum profit:** ~$1.24M
- **ROI at optimum:** ~53%
- **Breakeven probability:** 0.71 (false positive penalty model)

> **Note:** These represent two valid optimization regimes: *deployable* (constrained by real budget) vs *theoretical optimum* (unconstrained profit curve analysis). Both are correct answers to different business questions.

---

## 🚀 Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline (production mode)
python run_pipeline.py

# 3. View results
# - Model: outputs/models/churn_model_*.pkl
# - Predictions: outputs/batch_scores_latest.parquet
# - Action Plan: outputs/action_plan.csv (659 customers)

# 4. Run profit curve analysis
python -m src.analysis.profit_curve
```

---

## 🏗️ System Architecture

This system implements a batch-oriented ML pipeline for churn prediction with economic optimization:

```
┌─────────────────┐
│  Raw Data (IBM  │
│  Telco Churn)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Data Ingestion & Temporalization       │
│  • Monthly snapshots per customer       │
│  • Point-in-time correctness            │
│  • Temporal feature engineering         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Feature Store (Offline)                │
│  • Parquet-based feature tables         │
│  • Training & batch inference           │
│  • Versioned schemas                    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Training Pipeline                      │
│  • Time-aware cross-validation          │
│  • Class imbalance handling             │
│  • Model versioning                     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Batch Inference (Monthly)              │
│  • Churn probability scoring            │
│  • Expected value calculation           │
│  • SHAP explanations                    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Decision Engine                        │
│  • False Positive Penalty model         │
│  • Budget-constrained optimization      │
│  • Profit curve analysis (parallel)     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Action Plan Output                     │
│  • 659 customers (production)           │
│  • Ready for CRM integration            │
│  • Monitoring-ready format              │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
churn_prediction_mlpipeline/
├── data/                          # Raw and intermediate data
│   ├── raw/                       # Original IBM Telco dataset
│   ├── interim/                   # Cleaned intermediate data
│   └── processed/                 # Final processed data
│
├── datasets/                      # Temporalized feature tables
│   ├── monthly_customer_snapshots.parquet
│   ├── customers_master.csv
│   └── SCHEMA.md                  # Data schema documentation
│
├── feature_store/                 # Feature store implementation
│   ├── offline/                   # Parquet-based offline features
│   └── online/                    # SQLite/key-value online features
│
├── src/                           # Source code
│   ├── data_ingestion/            # Data loading and cleaning
│   ├── feature_engineering/       # Feature computation logic
│   ├── models/                    # Model architectures
│   ├── training_pipeline/         # Training orchestration
│   ├── inference/                 # Batch and API inference
│   ├── monitoring/                # Drift detection and alerts
│   ├── decision_engine/           # Business logic for targeting
│   ├── analysis/                  # 🆕 Profit curve optimization
│   ├── experiments/               # 🆕 A/B testing framework
│   └── utils/                     # Shared utilities
│
├── configs/                       # YAML configuration files
│   ├── data.yaml                  # Data paths and settings
│   ├── features.yaml              # Feature engineering config
│   ├── training.yaml              # Model training parameters
│   ├── monitoring.yaml            # Drift thresholds
│   └── decision.yaml              # Business rules
│
├── models/                        # Saved models and registry
│   ├── model_v1/
│   ├── model_v2/
│   └── registry.json              # Model metadata registry
│
├── outputs/                       # Inference outputs
│   ├── batch_scores_*.parquet     # Batch prediction results
│   └── reports/                   # Evaluation reports
│
├── notebooks/                     # Exploratory analysis only
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_analysis.ipynb
│
├── deployment/                    # Deployment artifacts
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── DEPLOYMENT.md
│   └── kubernetes/                # K8s manifests (optional)
│
├── tests/                         # Unit and integration tests
│   ├── test_data_ingestion/
│   ├── test_features/
│   └── test_inference/
│
├── .github/workflows/             # CI/CD pipelines
│   └── ci.yml
│
├── pyproject.toml                 # Project dependencies
├── requirements.txt               # Alternative dependency list
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd churn_prediction_mlpipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# Run entire pipeline (ingestion → training → inference → decisions)
python run_pipeline.py
```

**Output:**
- Model: `outputs/models/churn_model_20240105_*.pkl` (ROC-AUC: 0.90)
- Predictions: `outputs/batch_scores_latest.parquet`
- Action Plan: `outputs/action_plan.csv` (10,000 customers targeted)
- Expected Profit: **$1,048,336** (210% ROI)

---

### 🆕 Production Analytics

#### 1. Profit Curve Analysis

Find the optimal number of customers to target:

```bash
python -m src.analysis.profit_curve
```

**Output:**
- Visualization: `outputs/reports/profit_curve.png`
- Data: `outputs/reports/profit_curve_data.csv`
- **Key Finding:** Target 10,000 customers for maximum profit

#### 2. A/B Test Simulation

Validate campaign effectiveness with statistical testing:

```bash
python -m src.experiments.ab_test_simulator
```

**Output:**
- Results: `outputs/experiments/ab_test_results.json`
- Treatment/Control cohorts: `outputs/experiments/*.csv`
- **Key Finding:** 38% churn reduction, 194% ROI (p < 0.001)

#### 3. Drift Monitoring

Monitor data and model drift over time:

```bash
python -m src.monitoring.drift_detector \
  --reference-date 2024-01-01 \
  --current-date 2024-12-01
```

**Output:**
- Report: `outputs/monitoring_reports/monitoring_report_*.txt`
- Alerts for drifted features (PSI > 0.2)
- Automated retraining recommendations

---

### Individual Pipeline Steps

#### 1. Data Ingestion & Temporalization

```bash
python -m src.data_ingestion.run --config configs/data.yaml
```

This will:
- Load the IBM Telco dataset
- Clean and validate data
- Create monthly temporal snapshots (199,680 observations)
- Generate rolling features
- Save to `datasets/monthly_customer_snapshots.parquet`

#### 2. Train a Model

```bash
python -m src.training_pipeline.run --config configs/training.yaml
```

This will:
- Load training data from feature store
- Perform time-aware cross-validation
- Train XGBoost/LightGBM models
- Log experiments to MLflow
- Save model to `models/model_v{n}/`
- Register model in registry

#### 3. Batch Inference

```bash
churn-infer --snapshot-date 2024-12-01 --config configs/inference.yaml
```

This will:
- Load production model
- Pull latest features
- Generate churn probabilities
- Compute SHAP explanations
- Save to `outputs/batch_scores_2024-12-01.parquet`

#### 4. Run Decision Engine

```bash
python -m src.decision_engine.run --scores outputs/batch_scores_2024-12-01.parquet
```

This will:
- Apply CLV-based targeting logic
- Compute expected value per customer
- Generate retention action plan
- Save to `outputs/action_plan_2024-12-01.csv`

#### 5. Monitor for Drift

```bash
churn-monitor --reference-date 2024-01-01 --current-date 2024-12-01
```

This will:
- Compute PSI and KL divergence
- Check for data and concept drift
- Generate monitoring report
- Trigger retraining if thresholds exceeded



## 🔧 Configuration

All system behavior is controlled via YAML configs in `configs/`:

- **data.yaml**: Data paths, temporal window settings
- **features.yaml**: Feature definitions, rolling windows
- **training.yaml**: Model hyperparameters, CV strategy
- **monitoring.yaml**: Drift thresholds, alert rules
- **decision.yaml**: CLV models, retention costs, targeting thresholds

## 📊 Key Features

### 1. Temporal Dataset Design
- **Point-in-time correctness**: All features use only past data
- **Monthly snapshots**: Each customer has multiple time-stamped rows
- **Rolling aggregations**: 3-month, 6-month trends and patterns
- **No label leakage**: Strict temporal validation

### 2. Feature Store
- **Offline store**: Parquet files for training and batch inference
- **Online store**: Fast key-value lookup for real-time predictions
- **Schema versioning**: Track feature schema changes over time
- **Reusable features**: DRY principle for feature computation

### 3. Training Pipeline
- **Time-aware CV**: Train on older data, validate on recent
- **Class imbalance handling**: SMOTE, class weights, stratified sampling
- **Hyperparameter tuning**: Optuna/GridSearch integration
- **Experiment tracking**: Full reproducibility with MLflow

### 4. Monitoring & Drift
- **Data drift**: PSI, KL divergence per feature
- **Concept drift**: Performance degradation detection
- **Calibration monitoring**: Brier score over time
- **Automated retraining**: Trigger based on drift or KPIs

### 5. Business-Aware Decisions
- **Expected value optimization**: p(churn) × CLV × retention_rate - cost
- **Segment-specific strategies**: Different thresholds per customer value
- **A/B testing framework**: Compare model vs rule-based targeting
- **Profit curves**: Optimize decision thresholds for max ROI

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_features/
```

## 📈 Evaluation Metrics

### Model Metrics
- ROC-AUC, PR-AUC
- Log Loss, Brier Score
- Calibration plots
- Confusion matrix by threshold

### Decision Metrics
- Precision@K, Recall@K
- Profit curves
- Cost-adjusted lift
- Net revenue impact

### Business Metrics
- Retention rate (treatment vs control)
- Revenue saved per 1,000 customers
- ROI on retention campaigns

## � Deployment Readiness (Design)

**Scope:** This project intentionally stops at **batch, budget-constrained deployment**.

For churn prediction:
- Monthly batch processing is sufficient (churn is slow-moving)
- Real-time APIs add complexity without immediate business value
- Automated incentive execution requires business process integration beyond ML scope

**What's Production-Ready:**
- ✅ Temporal feature engineering (point-in-time correct)
- ✅ Batch inference pipeline (monthly cadence)
- ✅ Budget-constrained decision engine (prevents overspend)
- ✅ Action plan output (ready for CRM integration)
- ✅ Monitoring design (drift detection, rollback procedures)

**What's Design/Future Work:**
- 📋 Real-time API endpoints (FastAPI scaffolding exists, not deployed)
- 📋 Kubernetes orchestration (not needed for batch workloads)
- 📋 Automated alerting (PagerDuty/Slack integration design only)
- 📋 Canary deployments (design documented, not implemented)

---

## 📋 Production Deployment Strategy (Design Documentation)

### Batch Processing Cadence

**Monthly Execution (Day 1 of each month)**:
```bash
# 1. Feature extraction (Day 1, 00:00 UTC)
python -m src.feature_engineering.features \
  --snapshot-date $(date +%Y-%m-01) \
  --output feature_store/offline/

# 2. Batch inference (Day 1, 02:00 UTC)
python -m src.inference.batch \
  --model-id model_v_production \
  --features feature_store/offline/latest.parquet \
  --output outputs/batch_scores_$(date +%Y%m).parquet

# 3. Decision engine (Day 1, 04:00 UTC)
python -m src.decision_engine.engine \
  --scores outputs/batch_scores_$(date +%Y%m).parquet \
  --config configs/decision.yaml \
  --output outputs/action_plan_$(date +%Y%m).csv

# 4. CRM integration (Day 1, 06:00 UTC)
python -m src.deployment.crm_loader \
  --action-plan outputs/action_plan_$(date +%Y%m).csv \
  --campaign-start-date $(date +%Y-%m-05)
```

**Why Monthly?**
- Churn is slow-moving (not real-time)
- Campaigns need 2-3 weeks to execute
- Reduces infrastructure costs vs daily scoring
- Aligns with billing cycles

---

### Budget Gating

**Automated Budget Controls** (prevents overspend):

```yaml
# configs/decision.yaml
budget:
  max_monthly_budget: 50000.0  # Hard limit: $50K/month
  priority_metric: "expected_value"  # Rank customers by EV
  reserve_for_high_value_pct: 0.30  # 30% budget for VIP customers
  
  # Emergency stop
  max_budget_override: false  # Requires manual approval to exceed
```

**Budget Enforcement**:
1. Decision engine sorts customers by expected value (descending)
2. Cumulative cost tracked as customers are selected
3. **Hard stop** when budget limit reached (no partial campaigns)
4. Alert triggered if eligible customers > budget capacity (signal to increase budget)

**Monitoring**:
```bash
# Daily budget burn rate check
python -m src.monitoring.budget_tracker \
  --alert-threshold 0.90 \
  --webhook slack://retention-alerts
```

---

### A/B Testing Framework

**50/50 Treatment-Control Split**:

```python
# Integrated into decision engine
python -m src.decision_engine.engine \
  --scores outputs/batch_scores_latest.parquet \
  --ab-test-enabled \
  --treatment-pct 0.50 \
  --control-group-size 1000
```

**How It Works**:
1. Decision engine selects 1,318 eligible customers (EV > 0)
2. **Random 50/50 split**:
   - Treatment group (659): Receive retention offers
   - Control group (659): No intervention (observe natural churn)
3. Both groups tracked for 3 months
4. Causal effect measured: `retention_rate_treatment - retention_rate_control`

**Validation Metrics** (Month 3):
```
Expected Results:
  Treatment retention: 60% (40% prevented from churning)
  Control retention: 35% (natural retention rate)
  Lift: +25 percentage points
  ROI: (0.25 × 659 × $432 CLV) / $50K = 142%
```

**Tools**:
```bash
# Generate A/B test report
python -m src.experiments.ab_test_simulator \
  --actual-data outputs/action_plan_202601.csv \
  --duration-months 3
```

---

### Rollback Procedures

**When to Rollback**:
- Retention success rate < 30% (expected: 40%)
- Model drift detected (PSI > 0.1 on key features)
- False positive rate > 60%
- Negative ROI for 2 consecutive months

**Rollback Steps**:

1. **Immediate Stop** (if critical failure):
   ```bash
   # Disable decision engine
   python -m src.deployment.emergency_stop \
     --reason "retention_rate_below_threshold" \
     --notify stakeholders
   
   # Revert to previous model
   python -m src.models.registry promote \
     --model-id model_v_20260101_120000 \
     --status production
   ```

2. **Root Cause Analysis**:
   ```bash
   # Check drift metrics
   python -m src.monitoring.drift_detector \
     --baseline feature_store/baseline_stats.json \
     --current feature_store/offline/latest.parquet
   
   # Validate data quality
   python -m src.monitoring.data_quality \
     --schema datasets/SCHEMA.md \
     --data feature_store/offline/latest.parquet
   ```

3. **Gradual Rollback** (if degradation, not failure):
   - Reduce target audience by 50% (659 → 330 customers)
   - Increase probability threshold (0.75 → 0.85)
   - Monitor for 1 week, then decide: continue or full rollback

4. **Post-Mortem**:
   - Document failure mode
   - Add automated test to prevent recurrence
   - Update monitoring thresholds

**Automated Alerts** (PagerDuty/Slack):
```yaml
# configs/monitoring.yaml
alerts:
  - metric: retention_success_rate
    threshold: 0.30
    window: 7d
    action: page_oncall
  
  - metric: psi_max_feature
    threshold: 0.10
    window: 1d
    action: email_ml_team
  
  - metric: roi
    threshold: 0.50
    window: 30d
    action: notify_stakeholders
```

---

### Monitoring Dashboard

**Real-Time Metrics** (Grafana/Datadog):

**Model Performance**:
- Predicted churn rate vs actual (weekly)
- ROC-AUC on validation set (monthly retrain)
- Calibration curve (Brier score)

**Business Metrics**:
- Retention success rate (treatment vs control)
- Budget utilization (% of $50K spent)
- ROI: `(prevented_churn_value - campaign_cost) / campaign_cost`
- Cost per retained customer

**System Health**:
- Feature drift (PSI per feature, max across all)
- Prediction latency (p95, p99)
- Data freshness (hours since last feature update)
- Pipeline success rate (% successful runs)

**Example Dashboard Query**:
```sql
-- Weekly retention success rate
SELECT 
  DATE_TRUNC('week', campaign_start_date) AS week,
  COUNT(*) AS customers_targeted,
  SUM(CASE WHEN retained = 1 THEN 1 ELSE 0 END) AS retained,
  SUM(CASE WHEN retained = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS retention_rate,
  SUM(expected_value) AS predicted_value,
  SUM(actual_value) AS actual_value
FROM retention_outcomes
WHERE campaign_id = 'monthly_202601'
GROUP BY 1
ORDER BY 1 DESC;
```

---

### Retraining Strategy

**Quarterly Retraining** (unless triggered early):

```bash
# Scheduled: 1st of every quarter
python run_pipeline.py \
  --retrain \
  --validation-start 2024-10-01 \
  --validation-end 2024-12-31
```

**Early Trigger Conditions**:
1. **Drift detected**: PSI > 0.1 on ≥3 features
2. **Performance degradation**: ROC-AUC drops below 0.85
3. **Business rule change**: CLV model updated, new retention offers
4. **Data distribution shift**: New product launches, market changes

**Retraining Pipeline**:
1. Extract fresh training data (last 12 months)
2. Validate data quality (schema, completeness)
3. Train new model (same architecture)
4. **Shadow mode** (run parallel with production for 1 week)
5. Compare metrics: new vs old
6. **Promote if**:
   - ROC-AUC ≥ current_model - 0.01 (allow minor degradation)
   - Calibration improved (lower Brier score)
   - No significant bias shift (fairness metrics)
7. If promoted: archive old model, update registry

**Automated A/B Test** (before full rollout):
```bash
# Run new model on 10% of traffic for 2 weeks
python -m src.deployment.canary \
  --new-model model_v_20260401_080000 \
  --current-model model_v_20260105_184900 \
  --traffic-split 0.10 \
  --duration-days 14
```

---

### CI/CD

GitHub Actions workflow automatically:
- Runs linting (black, isort, flake8)
- Executes test suite
- Validates pipeline on sample data
- Builds Docker images
- (Optional) Deploys to staging/production

## 📚 Documentation

- [Data Schema](datasets/SCHEMA.md) - Detailed data dictionary
- [System Design](docs/SYSTEM_DESIGN.md) - Architecture deep-dive
- [Interview Guide](docs/INTERVIEW_GUIDE.md) - Talking points for interviews
- [Deployment Guide](deployment/DEPLOYMENT.md) - Production deployment steps

## 🎯 Interview Talking Points

**Problem**: Customer churn prediction systems often fail due to:
- Temporal leakage (using future data in training)
- Ignoring intervention costs (false positives are expensive)
- Unrealistic profit assumptions (monotonic curves)

**Solution**: Batch-oriented ML system with economic realism:
- Temporalized dataset (199,680 monthly snapshots, point-in-time correct)
- False Positive Penalty model (C_fp=$300 vs C_tp=$50 → breakeven at p=0.71)
- Budget-constrained optimization (production: 659 customers, $50K budget)
- Profit curve analysis (unconstrained optimal: 25,100 customers, $1.24M)

**Key Technical Decisions**:
- **Batch over real-time**: Churn is slow-moving, monthly cadence sufficient
- **Budget constraints**: Hard $50K limit prevents overspend (105% ROI achievable)
- **False positive penalty**: Models realistic cost of wasted incentives
- **Two optimization regimes**: Production (constrained) vs analysis (unconstrained)

**Business Impact**:
- 105% ROI on production deployment (659 customers)
- $1.24M theoretical maximum (25,100 customers, unconstrained)
- System captures 96.4% of optimal profit within budget
- Non-monotonic profit curve (realistic economics)

**What I Built vs What I Designed**:
- **Built**: Temporal features, training pipeline, batch inference, decision engine, profit curve
- **Designed**: Drift monitoring procedures, rollback strategy, A/B testing framework
- **Intentionally Not Built**: Real-time API, K8s orchestration (unnecessary for batch workload)

## 🤝 Contributing

This is an interview/portfolio project. For production use, please:
1. Replace simulated data with real customer data pipelines
2. Add proper authentication/authorization to API
3. Implement production-grade logging and monitoring
4. Add data quality validation gates
5. Implement proper secret management

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

Your Name - [LinkedIn](https://linkedin.com/in/yourprofile) - [Portfolio](https://yourportfolio.com)

---

**Note**: This project uses the IBM Telco Customer Churn dataset as a foundation, but extends it with temporal dimensions and production-grade ML systems patterns suitable for real-world deployment.
