# Profit Curve Economics: Why It Peaks

## 🎯 The Problem with Naive Models

Most churn prediction systems use a simplistic expected value calculation:

```
EV = p × CLV × success_rate - cost
```

**This formula is WRONG** for production systems because:
- It assumes the same cost whether the customer would churn or not
- It ignores the **false positive penalty** of targeting customers who wouldn't have left
- It creates monotonically increasing profit curves (never peaks)
- It leads to over-targeting and wasted marketing spend

## ✅ The Correct Formula: False Positive Penalty Model

```
EV = p × (success_rate × CLV - C_tp) - (1-p) × C_fp
```

Where:
- **p** = churn probability (model prediction)
- **C_tp** = cost when customer WOULD churn (true positive)
- **C_fp** = cost when customer WOULD NOT churn (false positive)
- **success_rate** = probability intervention prevents churn
- **CLV** = customer lifetime value

## 🔑 Key Insight: False Positives Are Expensive

When you target a customer who wouldn't have churned anyway:
- **Wasted discount/incentive**: Pure loss, no benefit realized
- **Gaming behavior**: Customers learn to threaten churn for discounts
- **Opportunity cost**: Resources spent on wrong customers
- **Brand damage**: Over-contacting happy customers

Therefore: **C_fp > C_tp** (often 2-3× higher)

## 📊 Our Implementation

**Configuration** (from [`configs/decision.yaml`](configs/decision.yaml)):
```yaml
cost_true_positive: $50   # Intervention when customer would churn
cost_false_positive: $300  # Wasted incentive (6× higher!)
success_rate: 0.40         # 40% of interventions prevent churn
```

**Breakeven Analysis** (from [`outputs/reports/breakeven_analysis.csv`](outputs/reports/breakeven_analysis.csv)):

| Churn Prob | True Positive Value | False Positive Loss | Expected Value | Above Breakeven? |
|------------|---------------------|---------------------|----------------|------------------|
| 0.95       | $116.66            | $15.00              | **+$101.66**   | ✅ YES          |
| 0.85       | $104.38            | $45.00              | **+$59.38**    | ✅ YES          |
| **0.71**   | **$87.19**         | **$87.00**          | **+$0.19**     | ✅ BREAKEVEN    |
| 0.70       | $85.96             | $90.00              | **-$4.04**     | ❌ NO           |
| 0.50       | $61.40             | $150.00             | **-$88.60**    | ❌ NO           |
| 0.30       | $36.84             | $210.00             | **-$173.16**   | ❌ NO           |

**Breakeven Formula:**
```
p_breakeven = C_fp / (success_rate × CLV - C_tp + C_fp)
p_breakeven = 300 / (0.4 × 432 - 50 + 300) = 0.710
```

## 📈 Profit Curve Results

See visualization: [`outputs/reports/profit_curve.png`](outputs/reports/profit_curve.png)

**Key Metrics:**
- **Optimal Targeting**: 25,100 customers
- **Optimal Threshold**: p > 0.710 (exactly at breakeven!)
- **Maximum Profit**: $1,242,232
- **ROI at Optimum**: 53%

**Curve Behavior:**
1. **Sharp Rise** (0 → 5,000 customers): ROI = 126%, high-probability customers
2. **Continued Growth** (5,000 → 25,100): ROI decreases to 53% as the system targets lower probabilities
3. **PEAK at 25,100** customers where **marginal EV crosses zero** (p ≈ 0.71)
4. **Decline** (25,100+): False positive penalty dominates, negative marginal profit

## 🧮 Why It Peaks: The Math

For a customer at churn probability **p**:

**Marginal EV** = p × (0.4 × $432 - $50) - (1-p) × $300

Simplifies to:

**Marginal EV** = 122.8p - 300(1-p) = 422.8p - 300

**Marginal EV = 0** when: p = 300/422.8 = **0.710**

As the system moves down the ranked list:
- **p decreases** → True positive value drops
- **(1-p) increases** → False positive loss grows  
- Eventually **(1-p) × C_fp dominates** → Marginal EV becomes negative
- **Profit peaks and declines**

This is **guaranteed mathematically** with proper cost modeling.

## 🎓 Production Implications

### Interview Gold
When asked "Why does your profit curve peak?":

> "Because I model false positive costs explicitly. As the system targets lower-probability customers, the (1-p) × C_fp term grows and eventually dominates, making marginal profit negative. The global optimum occurs where marginal EV crosses zero, not at maximum targeting."

### Business Value
- **Prevents over-targeting**: Stops at 25,100 customers, not 100,000
- **Maximizes profit**: $1.24M vs potentially negative if targeting continued
- **Realistic economics**: Accounts for wasted incentives and gaming behavior
- **Explainable**: Can justify to stakeholders why the system doesn't target everyone

### Technical Excellence
- **Non-monotonic optimization**: Demonstrates understanding of diminishing returns
- **Cost modeling**: Separates true/false positive costs (rare in practice)
- **Marginal analysis**: Uses calculus-based optimization (dProfit/dN = 0)
- **Production-ready**: Solves real business problem, not just maximizing accuracy

## 🚀 Extensions

Future improvements:
1. **Dynamic C_fp**: Increase over time as customers learn gaming behavior
2. **Segment-specific costs**: Different FP penalties by customer tier
3. **Lifetime FP penalty**: Model long-term brand damage, not just immediate cost
4. **Multi-period optimization**: Account for repeated targeting campaigns

## 📚 References

**Code Implementation:**
- Decision Engine: [`src/decision_engine/engine.py`](src/decision_engine/engine.py#L95-L130)
- Profit Curve Analyzer: [`src/analysis/profit_curve.py`](src/analysis/profit_curve.py#L89-L110)
- Configuration: [`configs/decision.yaml`](configs/decision.yaml#L47-L68)

**Key Outputs:**
- Profit Curve Visualization: [`outputs/reports/profit_curve.png`](outputs/reports/profit_curve.png)
- Profit Curve Data: [`outputs/reports/profit_curve_data.csv`](outputs/reports/profit_curve_data.csv)
- Breakeven Analysis: [`outputs/reports/breakeven_analysis.csv`](outputs/reports/breakeven_analysis.csv)

---

**Bottom Line:** This is how production ML systems should model economics. False positives are expensive, and ignoring them leads to suboptimal business decisions.
