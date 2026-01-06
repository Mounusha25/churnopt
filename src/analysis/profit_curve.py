"""
Profit curve analysis for optimal targeting threshold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from ..utils import setup_logging, ensure_dir

logger = setup_logging(__name__)


class ProfitCurveAnalyzer:
    """Analyze profit vs targeting threshold to find optimal operating point."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger
        
        # Get variable cost configuration
        self.variable_cost_config = config.get('retention', {}).get('costs', {}).get('variable_cost_model', {})
        
    def compute_profit_curve(
        self,
        scores: pd.DataFrame,
        retention_config: Dict,
        step_size: int = 100
    ) -> pd.DataFrame:
        """
        Compute profit at different targeting thresholds.
        
        Args:
            scores: DataFrame with churn_probability, clv, etc.
            retention_config: Retention strategy configuration
            step_size: Number of customers to add per step
            
        Returns:
            DataFrame with threshold, customers_targeted, profit, roi
        """
        self.logger.info("Computing profit curve...")
        
        # Sort by churn probability descending
        scores_sorted = scores.sort_values('churn_probability', ascending=False).copy()
        
        # Get config values
        retention_cost = retention_config['costs'].get('default_cost', 50)
        success_rate = retention_config.get('success_rate', 0.40)
        
        # Check if using variable cost model
        variable_cost_config = retention_config['costs'].get('variable_cost_model', {})
        use_variable_cost = variable_cost_config.get('enabled', False)
        
        # Check for false positive cost model
        cost_tp = variable_cost_config.get('cost_true_positive')
        cost_fp = variable_cost_config.get('cost_false_positive')
        use_fp_penalty = cost_tp is not None and cost_fp is not None
        
        if use_variable_cost and use_fp_penalty:
            self.logger.info(f"Using FALSE POSITIVE PENALTY model:")
            self.logger.info(f"  True Positive cost: ${cost_tp}")
            self.logger.info(f"  False Positive cost: ${cost_fp}")
            self.logger.info(f"  Creates natural profit peak as (1-p)×C_fp dominates")
        elif use_variable_cost:
            base_cost = variable_cost_config.get('base_cost', 30.0)
            risk_multiplier = variable_cost_config.get('risk_multiplier', 200.0)
            use_quadratic = variable_cost_config.get('use_quadratic', False)
            
            if use_quadratic:
                self.logger.info(f"Using QUADRATIC cost: cost = {base_cost} + {risk_multiplier} × (churn_prob)²")
            else:
                self.logger.info(f"Using LINEAR cost: cost = {base_cost} + {risk_multiplier} × churn_prob")
        else:
            self.logger.info(f"Using FIXED cost: ${retention_cost} per customer")
        
        # Simulate CLV if not present
        if 'clv' not in scores_sorted.columns:
            default_clv = self.config.get('clv', {}).get('simple', {}).get('monthly_revenue', 60) * 24 * 0.3
            scores_sorted['clv'] = default_clv
        
        results = []
        
        # Test different numbers of customers to target
        max_customers = min(len(scores_sorted), 30000)  # Extended to see profit decline
        
        for n_customers in range(step_size, max_customers + 1, step_size):
            top_n = scores_sorted.iloc[:n_customers]
            p = top_n['churn_probability']
            
            if use_fp_penalty:
                # FALSE POSITIVE PENALTY MODEL
                # EV = p × (success_rate × CLV - C_tp) - (1-p) × C_fp
                true_positive_value = (p * (success_rate * top_n['clv'] - cost_tp)).sum()
                false_positive_loss = ((1 - p) * cost_fp).sum()
                
                profit = true_positive_value - false_positive_loss
                total_cost = (p * cost_tp + (1 - p) * cost_fp).sum()
                expected_savings = true_positive_value + false_positive_loss  # For reporting
                
            elif use_variable_cost:
                # Calculate expected savings
                expected_savings = (p * top_n['clv'] * success_rate).sum()
                
                # Variable cost: higher-risk customers cost more
                if use_quadratic:
                    total_cost = (base_cost * n_customers) + (risk_multiplier * (p ** 2).sum())
                else:
                    total_cost = (base_cost * n_customers) + (risk_multiplier * p.sum())
                
                profit = expected_savings - total_cost
            else:
                # Fixed cost model
                expected_savings = (p * top_n['clv'] * success_rate).sum()
                total_cost = n_customers * retention_cost
                profit = expected_savings - total_cost
            
            roi = (profit / total_cost * 100) if total_cost > 0 else 0
            
            # Threshold is the minimum churn probability in this group
            threshold = top_n['churn_probability'].min()
            
            results.append({
                'n_customers': n_customers,
                'threshold': threshold,
                'expected_savings': expected_savings,
                'total_cost': total_cost,
                'profit': profit,
                'roi': roi,
                'profit_per_customer': profit / n_customers
            })
        
        df_curve = pd.DataFrame(results)
        
        # Calculate marginal profit (profit change per additional customer batch)
        df_curve['marginal_profit'] = df_curve['profit'].diff().fillna(0)
        
        # Find optimal point (where marginal profit crosses zero)
        optimal_idx = df_curve['profit'].idxmax()
        optimal = df_curve.iloc[optimal_idx]
        
        self.logger.info(f"Optimal targeting:")
        self.logger.info(f"  Customers: {optimal['n_customers']:.0f}")
        self.logger.info(f"  Threshold: {optimal['threshold']:.4f}")
        self.logger.info(f"  Profit: ${optimal['profit']:,.2f}")
        self.logger.info(f"  ROI: {optimal['roi']:.1f}%")
        
        return df_curve
    
    def plot_profit_curve(
        self,
        curve_df: pd.DataFrame,
        output_path: str = "outputs/reports/profit_curve.png"
    ):
        """Create visualization of profit curve."""
        ensure_dir(Path(output_path).parent)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Profit Curve Analysis - Optimal Targeting Threshold', fontsize=16, fontweight='bold')
        
        # 1. Profit vs Number of Customers
        ax1 = axes[0, 0]
        ax1.plot(curve_df['n_customers'], curve_df['profit'], linewidth=2, color='#2E86AB')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        optimal_idx = curve_df['profit'].idxmax()
        optimal_n = curve_df.iloc[optimal_idx]['n_customers']
        optimal_profit = curve_df.iloc[optimal_idx]['profit']
        ax1.axvline(x=optimal_n, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                    label=f'Optimal: {optimal_n:.0f} customers')
        ax1.scatter([optimal_n], [optimal_profit], color='green', s=100, zorder=5, marker='*')
        ax1.annotate('Global Optimum\n(Marginal EV = 0)', 
                    xy=(optimal_n, optimal_profit),
                    xytext=(optimal_n * 0.7, optimal_profit * 1.1),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8))
        ax1.set_xlabel('Number of Customers Targeted', fontsize=11)
        ax1.set_ylabel('Profit ($)', fontsize=11)
        ax1.set_title('Profit vs Customers Targeted', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        
        # 2. ROI vs Number of Customers
        ax2 = axes[0, 1]
        ax2.plot(curve_df['n_customers'], curve_df['roi'], linewidth=2, color='#A23B72')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Customers Targeted', fontsize=11)
        ax2.set_ylabel('ROI (%)', fontsize=11)
        ax2.set_title('ROI vs Customers Targeted', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Marginal Profit (where it crosses zero = optimal point)
        ax3 = axes[1, 0]
        ax3.plot(curve_df['n_customers'], curve_df['marginal_profit'], linewidth=2, color='#F18F01')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Marginal Profit = 0')
        ax3.axvline(x=optimal_n, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Number of Customers Targeted', fontsize=11)
        ax3.set_ylabel('Marginal Profit ($)', fontsize=11)
        ax3.set_title('Marginal Profit per Batch (crosses zero at optimum)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Churn Probability Threshold
        ax4 = axes[1, 1]
        ax4.plot(curve_df['n_customers'], curve_df['threshold'], linewidth=2, color='#C73E1D')
        ax4.set_xlabel('Number of Customers Targeted', fontsize=11)
        ax4.set_ylabel('Min Churn Probability Threshold', fontsize=11)
        ax4.set_title('Targeting Threshold vs Scale', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Profit curve saved to {output_path}")
        plt.close()
        
        return output_path


def analyze_profit_curve(scores_path: str, config_path: str = 'configs/decision.yaml'):
    """Main entry point for profit curve analysis."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load scores
    scores = pd.read_parquet(scores_path)
    
    # Run analysis
    analyzer = ProfitCurveAnalyzer(config)
    curve_df = analyzer.compute_profit_curve(scores, config['retention'])
    
    # Save results
    ensure_dir('outputs/reports')
    curve_df.to_csv('outputs/reports/profit_curve_data.csv', index=False)
    
    # Plot
    analyzer.plot_profit_curve(curve_df)
    
    return curve_df


if __name__ == '__main__':
    analyze_profit_curve('outputs/batch_scores_latest.parquet')
