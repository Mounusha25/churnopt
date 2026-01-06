# Customer Churn Prediction – Decision-Optimized ML System

**0.91 ROC-AUC | Budget-Constrained Deployment | Profit-Optimized Targeting**

A production-ready, batch-oriented ML system for customer churn prediction that separates prediction from decision-making and optimizes expected business value under real constraints.

---

## Problem

Most churn models optimize accuracy but fail to answer:
> *Who should we actually target, given limited budget and expensive false positives?*

---

## Solution

A decision-optimized churn system that:
- Uses temporal, point-in-time-correct features
- Applies a False Positive Penalty to model wasted incentives
- Optimizes expected value, not accuracy
- Enforces hard budget and capacity constraints
- Supports unconstrained profit analysis for strategy planning

---

## Key Results

### Production Deployment (Constrained)
- Customers targeted: **659 (top 0.3%)**
- Monthly budget: **$50,000**
- Expected net value: **~$52,000**
- ROI: **~105%**
- Model performance: **0.9068 ROC-AUC**

### Unconstrained Economic Analysis
- Theoretical optimum: **~25,100 customers**
- Maximum profit: **~$1.24M**
- Breakeven churn probability: **p = 0.71**

> These represent two valid optimization regimes: deployable under real constraints vs economic optimum without constraints.

---

## Architecture

```
Raw Data → Temporal Features → Model Training
         → Batch Inference → Decision Engine
         → Budget-Constrained Action Plan
```

- Monthly batch execution
- Versioned models and features
- CSV action plan for CRM execution
- Human-in-the-loop deployment

---

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# View outputs
ls outputs/action_plan.csv              # 659 targeted customers
ls outputs/reports/profit_curve.png     # Economic analysis
```

---

## Repository Structure

```
src/
├── data_ingestion/        # Temporal dataset creation
├── feature_engineering/   # Point-in-time features
├── training_pipeline/     # Model training
├── inference/             # Batch scoring
├── decision_engine/       # EV optimization & budget gating
└── analysis/              # Profit curve analysis

configs/
└── decision.yaml          # Business rules (C_tp, C_fp, budget)

outputs/
├── action_plan.csv        # Production targeting list
└── reports/
    └── profit_curve.png   # Unconstrained analysis
```

---

## Technical Details

### Temporal Feature Engineering
- No label leakage (strict point-in-time correctness)
- Monthly snapshots capture customer evolution
- Rolling aggregations for trend detection

### False Positive Penalty Economics
- C_fp = $300 (wasted incentive cost)
- C_tp = $50 (retention campaign cost)
- Breakeven: p = 0.71

### Budget-Constrained Optimization
- Ranks customers by expected value
- Hard stops at $50K monthly budget
- Captures 96.4% of optimal profit within constraints

### Non-Monotonic Profit Curve
- Profit peaks at ~25,100 customers, then declines
- Validates realistic false positive costs

---

## Deployment

This system uses batch, budget-gated deployment:
- Monthly cadence aligns with billing cycles
- Budget controls prevent overspend
- Human-in-the-loop for campaign execution

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) – System design & budget gate logic
- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) – Technical implementation
- [PROFIT_CURVE_ECONOMICS.md](PROFIT_CURVE_ECONOMICS.md) – False Positive Penalty math
- [OPTIMIZATION_MODES.md](OPTIMIZATION_MODES.md) – Constrained vs unconstrained

---

## License

MIT License

---

Built on IBM Telco Customer Churn data, extended with temporal modeling and production-grade decision optimization.
