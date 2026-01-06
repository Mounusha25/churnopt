# What I Built: Customer Churn Prediction System

**One-Page Summary for Recruiters**

---

## 🎯 Project Goal

Build a batch-oriented ML system to predict customer churn and optimize retention spending with realistic economics.

---

## 🛠️ What I Actually Built

### 1. Temporal Dataset Engineering
- **199,680 monthly customer snapshots** from 7,043 customers
- Point-in-time correct features (no temporal leakage)
- Rolling aggregations (3-month, 6-month trends)

### 2. ML Training Pipeline
- **Logistic Regression model**: 0.91 ROC-AUC
- Time-aware cross-validation
- Class imbalance handling (SMOTE, class weights)

### 3. Economic Decision Engine
- **False Positive Penalty model**: C_fp=$300 vs C_tp=$50
- Budget-constrained optimization ($50K monthly limit)
- Expected value calculation per customer
- **Production output**: 659 customers targeted, 105% ROI

### 4. Profit Curve Analysis
- Non-monotonic profit curve (realistic economics)
- Unconstrained optimal: 25,100 customers, $1.24M profit
- Breakeven probability: 0.71 (false positive penalty threshold)
- System validation: captures 96.4% of optimal profit

### 5. Comprehensive Documentation
- Architecture diagrams (batch flow, budget gates)
- Interview guide (Q&A, talking points)
- System design (temporal features, optimization modes)
- Production deployment strategy (rollback, monitoring)

---

## 📊 Key Results

| Metric | Production (Budget-Constrained) | Unconstrained (Analysis) |
|--------|--------------------------------|--------------------------|
| **Customers** | 659 (0.3% of base) | 25,100 (optimal) |
| **Budget** | $50,000 (hard limit) | No constraint |
| **Expected Profit** | ~$52,000 | ~$1.24M |
| **ROI** | 105% | 53% |
| **Model Performance** | 0.91 ROC-AUC | 0.91 ROC-AUC |

> **Note**: Both are correct — they solve different optimization problems (deployable vs theoretical optimum).

---

## 🎓 What I Learned

### Technical Skills
- Temporal feature engineering (avoiding label leakage)
- False positive penalty economics (realistic cost modeling)
- Budget-constrained optimization (production tradeoffs)
- Profit curve analysis (non-monotonic economics)

### Engineering Judgment
- **When NOT to deploy**: Real-time APIs unnecessary for slow-moving churn
- **Batch over streaming**: Monthly cadence sufficient, simpler to operate
- **Design vs implementation**: Document production strategy without over-building
- **Economic realism**: False positives are expensive (C_fp=6×C_tp)

### Business Thinking
- Two optimization regimes: constrained (deployable) vs unconstrained (analysis)
- Budget constraints matter: 659 vs 25,100 customers
- ROI varies by scale: 105% (small) vs 53% (large) due to marginal economics
- System captures 96.4% of optimal profit within budget

---

## 🚫 What I Intentionally Did NOT Build

**Why?** Demonstrating engineering judgment (knowing when to stop)

- ❌ **Real-time API**: Churn is slow-moving, batch is sufficient
- ❌ **Kubernetes**: Not needed for monthly batch workloads
- ❌ **Automated alerts**: Design documented, implementation overkill for portfolio
- ❌ **Canary deployments**: Good design pattern, but premature for v1

---

## 💼 Interviewer FAQs

**Q: Why only 659 customers in production?**  
A: $50K budget constraint. Decision engine ranks by expected value and stops at budget limit. Unconstrained optimal is 25,100 customers ($1.24M profit), but production systems have real constraints.

**Q: Why not deploy real-time API?**  
A: Churn is slow-moving (monthly billing cycles). Batch processing is simpler, cheaper, and sufficient. Real-time adds complexity without business value here.

**Q: How did you validate the system?**  
A: Two ways: (1) Profit curve shows non-monotonic economics (peaks then declines), (2) Unconstrained mode captures 96.4% of theoretical optimal, proving decision engine logic is correct.

**Q: What's the false positive penalty?**  
A: Cost of offering incentive to customer who wouldn't have churned anyway. Set at $300 (wasted offer) vs $50 (true positive cost). This creates breakeven probability of 0.71 and realistic profit curve.

**Q: Production-ready?**  
A: For batch deployment, yes. Has temporal correctness, budget constraints, monitoring design, rollback procedures. Would need CRM integration and business process approval for actual deployment.

---

## 📁 Repository Contents

- **Temporal dataset**: 199,680 snapshots (datasets/monthly_customer_snapshots.parquet)
- **Trained model**: Logistic Regression, 0.91 ROC-AUC (outputs/models/)
- **Production action plan**: 659 customers (outputs/action_plan.csv)
- **Profit curve visualization**: outputs/reports/profit_curve.png
- **Architecture docs**: ARCHITECTURE.md, SYSTEM_DESIGN.md
- **Interview prep**: INTERVIEW_GUIDE.md, INTERVIEW_ARTIFACTS_CHECKLIST.txt

---

## 🔗 Key Artifacts for Review

1. **Architecture diagram**: [ARCHITECTURE.md](ARCHITECTURE.md) — Budget gate comparison
2. **Profit curve**: [outputs/reports/profit_curve.png](outputs/reports/profit_curve.png) — Non-monotonic economics
3. **Action plan**: [outputs/action_plan.csv](outputs/action_plan.csv) — 659 customers, production-ready
4. **Decision config**: [configs/decision.yaml](configs/decision.yaml) — False positive penalty settings
5. **README**: [README.md](README.md) — Deployment strategy section

---

## ⏱️ Time Investment

- **Dataset engineering**: 2 days (temporal snapshots, rolling features)
- **Model training**: 1 day (pipeline, cross-validation)
- **Decision engine**: 2 days (expected value, budget constraints)
- **Profit curve analysis**: 1 day (optimization, visualization)
- **Documentation**: 2 days (architecture, interview prep)

**Total**: ~8 days (solo project)

---

## 🎤 Elevator Pitch (30 seconds)

*"I built a churn prediction system with realistic economics. Most projects ignore false positive costs — I modeled them explicitly at $300 per wasted incentive. This creates a non-monotonic profit curve that peaks at 25,100 customers, then declines. In production mode with a $50K budget, the system targets 659 high-value customers at 105% ROI. The key insight: both 659 and 25,100 are correct — they solve different optimization problems. I validated this by showing the decision engine captures 96.4% of the theoretical optimal profit."*

---

**GitHub**: [Your Repo URL]  
**Contact**: [Your Email]  
**LinkedIn**: [Your LinkedIn]
