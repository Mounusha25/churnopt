"""
A/B test simulation for retention campaigns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats
from pathlib import Path

from ..utils import setup_logging, ensure_dir

logger = setup_logging(__name__)


class ABTestSimulator:
    """Simulate A/B test for retention campaign effectiveness."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger
        
    def simulate_treatment_effect(
        self,
        targeted_customers: pd.DataFrame,
        treatment_fraction: float = 0.5,
        true_retention_rate: float = 0.40,
        noise_std: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate A/B test with treatment and control groups.
        
        Args:
            targeted_customers: Customers selected for intervention
            treatment_fraction: Fraction assigned to treatment (rest is control)
            true_retention_rate: True effectiveness of intervention
            noise_std: Noise in outcome measurement
            
        Returns:
            Tuple of (treatment_df, control_df)
        """
        self.logger.info("Simulating A/B test...")
        self.logger.info(f"  Total customers: {len(targeted_customers)}")
        self.logger.info(f"  Treatment fraction: {treatment_fraction:.1%}")
        
        df = targeted_customers.copy()
        
        # Random assignment to treatment/control
        np.random.seed(42)
        df['group'] = np.random.choice(
            ['treatment', 'control'],
            size=len(df),
            p=[treatment_fraction, 1 - treatment_fraction]
        )
        
        # Simulate churn outcomes
        df['actual_churned'] = np.random.binomial(1, df['churn_probability'])
        
        # Treatment effect: reduce churn for treatment group
        treatment_mask = df['group'] == 'treatment'
        
        # For treatment group, apply retention intervention
        # Some customers who would have churned are retained
        df.loc[treatment_mask, 'intervention_success'] = np.random.binomial(
            1, 
            true_retention_rate,
            size=treatment_mask.sum()
        )
        
        # Retained if intervention succeeded for those who would churn
        df['retained'] = False
        df.loc[treatment_mask, 'retained'] = (
            (df.loc[treatment_mask, 'actual_churned'] == 1) & 
            (df.loc[treatment_mask, 'intervention_success'] == 1)
        )
        
        # Final churn status
        df['final_churned'] = df['actual_churned'].copy()
        df.loc[df['retained'], 'final_churned'] = 0
        
        # Calculate value
        retention_cost = self.config['retention']['costs'].get('default_cost', 50)
        
        df['cost'] = 0
        df.loc[treatment_mask, 'cost'] = retention_cost
        
        df['value_saved'] = 0
        df.loc[df['retained'], 'value_saved'] = df.loc[df['retained'], 'clv']
        
        df['net_value'] = df['value_saved'] - df['cost']
        
        # Split into treatment and control
        treatment_df = df[df['group'] == 'treatment'].copy()
        control_df = df[df['group'] == 'control'].copy()
        
        return treatment_df, control_df
    
    def analyze_results(
        self,
        treatment_df: pd.DataFrame,
        control_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze A/B test results with statistical tests.
        
        Returns:
            Dictionary with test results and metrics
        """
        self.logger.info("="*60)
        self.logger.info("A/B TEST RESULTS")
        self.logger.info("="*60)
        
        # Sample sizes
        n_treatment = len(treatment_df)
        n_control = len(control_df)
        
        # Churn rates
        churn_rate_treatment = treatment_df['final_churned'].mean()
        churn_rate_control = control_df['final_churned'].mean()
        
        # Absolute and relative lift
        absolute_lift = churn_rate_control - churn_rate_treatment
        relative_lift = (absolute_lift / churn_rate_control) * 100 if churn_rate_control > 0 else 0
        
        # Statistical test (two-proportion z-test)
        from statsmodels.stats.proportion import proportions_ztest
        
        counts = np.array([
            treatment_df['final_churned'].sum(),
            control_df['final_churned'].sum()
        ])
        nobs = np.array([n_treatment, n_control])
        
        z_stat, p_value = proportions_ztest(counts, nobs)
        
        # Financial impact
        total_cost_treatment = treatment_df['cost'].sum()
        total_value_saved = treatment_df['value_saved'].sum()
        net_value = total_value_saved - total_cost_treatment
        roi = (net_value / total_cost_treatment * 100) if total_cost_treatment > 0 else 0
        
        # Results
        results = {
            'sample_sizes': {
                'treatment': int(n_treatment),
                'control': int(n_control)
            },
            'churn_rates': {
                'treatment': float(churn_rate_treatment),
                'control': float(churn_rate_control),
                'absolute_lift': float(absolute_lift),
                'relative_lift_pct': float(relative_lift)
            },
            'statistical_test': {
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant_at_5pct': p_value < 0.05,
                'significant_at_1pct': p_value < 0.01
            },
            'financial_impact': {
                'total_cost': float(total_cost_treatment),
                'total_value_saved': float(total_value_saved),
                'net_value': float(net_value),
                'roi_pct': float(roi),
                'value_per_customer': float(net_value / n_treatment) if n_treatment > 0 else 0
            }
        }
        
        # Log results
        self.logger.info(f"\nSample Sizes:")
        self.logger.info(f"  Treatment: {n_treatment:,}")
        self.logger.info(f"  Control: {n_control:,}")
        
        self.logger.info(f"\nChurn Rates:")
        self.logger.info(f"  Treatment: {churn_rate_treatment:.2%}")
        self.logger.info(f"  Control: {churn_rate_control:.2%}")
        self.logger.info(f"  Absolute Lift: {absolute_lift:.2%}")
        self.logger.info(f"  Relative Lift: {relative_lift:.1f}%")
        
        self.logger.info(f"\nStatistical Significance:")
        self.logger.info(f"  Z-statistic: {z_stat:.3f}")
        self.logger.info(f"  P-value: {p_value:.4f}")
        self.logger.info(f"  Significant (α=0.05): {'✓ YES' if p_value < 0.05 else '✗ NO'}")
        
        self.logger.info(f"\nFinancial Impact:")
        self.logger.info(f"  Total Cost: ${total_cost_treatment:,.2f}")
        self.logger.info(f"  Value Saved: ${total_value_saved:,.2f}")
        self.logger.info(f"  Net Value: ${net_value:,.2f}")
        self.logger.info(f"  ROI: {roi:.1f}%")
        
        self.logger.info("="*60)
        
        return results
    
    def save_results(
        self,
        treatment_df: pd.DataFrame,
        control_df: pd.DataFrame,
        results: Dict,
        output_dir: str = 'outputs/experiments'
    ):
        """Save A/B test results."""
        ensure_dir(output_dir)
        
        # Save cohort data
        treatment_df.to_csv(f'{output_dir}/treatment_cohort.csv', index=False)
        control_df.to_csv(f'{output_dir}/control_cohort.csv', index=False)
        
        # Convert numpy types to Python native types for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {
                    k: float(v) if hasattr(v, 'item') else bool(v) if isinstance(v, (bool, np.bool_)) else v
                    for k, v in value.items()
                }
            else:
                results_serializable[key] = float(value) if hasattr(value, 'item') else bool(value) if isinstance(value, (bool, np.bool_)) else value
        
        # Save results summary
        import json
        with open(f'{output_dir}/ab_test_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"A/B test results saved to {output_dir}/")


def run_ab_test_simulation(
    action_plan_path: str = 'outputs/action_plan.csv',
    config_path: str = 'configs/decision.yaml'
):
    """Main entry point for A/B test simulation."""
    
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load targeted customers
    targeted = pd.read_csv(action_plan_path)
    
    if len(targeted) == 0:
        logger.warning("No targeted customers found. Run decision engine first.")
        return None
    
    # Run simulation
    simulator = ABTestSimulator(config)
    treatment_df, control_df = simulator.simulate_treatment_effect(targeted)
    results = simulator.analyze_results(treatment_df, control_df)
    simulator.save_results(treatment_df, control_df, results)
    
    return results


if __name__ == '__main__':
    run_ab_test_simulation()
