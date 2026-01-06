"""
Decision engine for business-aware targeting.

Implements expected value optimization:
EV = p(churn) * CLV * retention_rate - retention_cost
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils import setup_logging, load_config, ensure_dir

logger = setup_logging(__name__)


class DecisionEngine:
    """
    Business-aware decision engine for churn prevention.
    
    Optimizes targeting based on expected value, not just churn probability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clv_config = config['clv']
        self.retention_config = config['retention']
        self.decision_config = config['decision']
        self.logger = logger
    
    def estimate_clv(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate Customer Lifetime Value.
        
        Simple model: CLV = monthly_revenue * expected_lifetime * margin
        """
        self.logger.info("Estimating CLV...")
        
        # Use actual monthly charges if available
        if 'monthly_charges' in df.columns:
            monthly_revenue = df['monthly_charges']
        else:
            monthly_revenue = self.clv_config['simple'].get('monthly_revenue', 60)
        
        # Lifetime depends on contract type and tenure
        default_lifetime = self.clv_config['simple'].get('default_lifetime_months', 24)
        margin = self.clv_config['simple'].get('margin', 0.30)
        
        clv = monthly_revenue * default_lifetime * margin
        
        return clv
    
    def segment_customers(self, df: pd.DataFrame) -> pd.Series:
        """Segment customers by value."""
        clv = self.estimate_clv(df)
        
        # Ensure clv is a Series (1D)
        if isinstance(clv, pd.DataFrame):
            clv = clv.iloc[:, 0]  # Take first column if DataFrame
        elif not isinstance(clv, pd.Series):
            clv = pd.Series(clv)  # Convert to Series if needed
        
        high_threshold = self.decision_config['segmentation']['clv_thresholds']['high_value']
        medium_threshold = self.decision_config['segmentation']['clv_thresholds']['medium_value']
        
        segments = pd.cut(
            clv,
            bins=[0, medium_threshold, high_threshold, np.inf],
            labels=['low_value', 'medium_value', 'high_value']
        )
        
        return segments
    
    def compute_expected_value(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute expected value of targeting each customer.
        
        EV = p_churn * CLV * retention_success_rate - retention_cost
        """
        self.logger.info("Computing expected values...")
        
        df = df.copy()
        
        # Estimate CLV
        df['clv'] = self.estimate_clv(df)
        
        # Segment customers
        df['segment'] = self.segment_customers(df)
        
        # Get retention success rate by segment
        default_success_rate = self.retention_config.get('success_rate', 0.40)
        success_rates = self.retention_config.get('success_rate_by_segment', {})
        
        df['retention_success_rate'] = df['segment'].map(
            lambda x: success_rates.get(x, default_success_rate)
        )
        
        # Get retention cost - use variable cost model if enabled
        variable_cost_config = self.retention_config['costs'].get('variable_cost_model', {})
        
        if variable_cost_config.get('enabled', False):
            # Check if using false positive cost model (realistic!)
            cost_tp = variable_cost_config.get('cost_true_positive')
            cost_fp = variable_cost_config.get('cost_false_positive')
            
            if cost_tp is not None and cost_fp is not None:
                # FALSE POSITIVE PENALTY MODEL
                # EV = p × (success_rate × CLV - C_tp) - (1-p) × C_fp
                # This creates natural profit peak as FP cost dominates at low probabilities
                
                self.logger.info("Using FALSE POSITIVE PENALTY cost model:")
                self.logger.info(f"  True Positive cost (would churn): ${cost_tp}")
                self.logger.info(f"  False Positive cost (wouldn't churn): ${cost_fp}")
                self.logger.info(f"  Formula: EV = p×(success×CLV - {cost_tp}) - (1-p)×{cost_fp}")
                
                # Store both costs for EV calculation
                df['cost_true_positive'] = cost_tp
                df['cost_false_positive'] = cost_fp
                
                # For display purposes, use expected cost
                df['retention_cost'] = (
                    df['churn_probability'] * cost_tp + 
                    (1 - df['churn_probability']) * cost_fp
                )
                
            else:
                # Fallback to quadratic/linear model
                base_cost = variable_cost_config.get('base_cost', 30.0)
                risk_multiplier = variable_cost_config.get('risk_multiplier', 200.0)
                use_quadratic = variable_cost_config.get('use_quadratic', False)
                
                if use_quadratic:
                    df['retention_cost'] = base_cost + (risk_multiplier * df['churn_probability'] ** 2)
                    self.logger.info("Using QUADRATIC variable cost model")
                else:
                    df['retention_cost'] = base_cost + (risk_multiplier * df['churn_probability'])
                    self.logger.info("Using LINEAR variable cost model")
                    
                self.logger.info(f"  Cost range: ${df['retention_cost'].min():.2f} - ${df['retention_cost'].max():.2f}")
        else:
            # Fallback: segment-based fixed costs
            default_cost = self.retention_config['costs'].get('default_cost', 50)
            costs_by_segment = self.retention_config['costs'].get('by_segment', {})
            
            df['retention_cost'] = df['segment'].map(
                lambda x: costs_by_segment.get(x, default_cost)
            )
            
            self.logger.info("Using FIXED cost model by segment")
        
        # Compute expected value
        if 'cost_true_positive' in df.columns and 'cost_false_positive' in df.columns:
            # Use false positive penalty model
            # EV = p × (success_rate × CLV - C_tp) - (1-p) × C_fp
            p = df['churn_probability']
            df['expected_value'] = (
                p * (df['retention_success_rate'] * df['clv'] - df['cost_true_positive']) -
                (1 - p) * df['cost_false_positive']
            )
        else:
            # Legacy model: EV = p × CLV × success_rate - cost
            df['expected_value'] = (
                df['churn_probability'] *
                df['clv'] *
                df['retention_success_rate'] -
                df['retention_cost']
            )
        
        # DEBUG: Log distributions
        self.logger.info("=== EXPECTED VALUE CALCULATION DEBUG ===")
        self.logger.info(f"CLV: mean={df['clv'].mean():.2f}, median={df['clv'].median():.2f}, max={df['clv'].max():.2f}")
        self.logger.info(f"Retention cost: mean={df['retention_cost'].mean():.2f}, median={df['retention_cost'].median():.2f}")
        self.logger.info(f"Retention success rate: mean={df['retention_success_rate'].mean():.4f}")
        self.logger.info(f"Expected value: mean={df['expected_value'].mean():.2f}, median={df['expected_value'].median():.2f}, max={df['expected_value'].max():.2f}")
        self.logger.info(f"Positive EV count: {(df['expected_value'] > 0).sum()} / {len(df)}")
        
        return df
    
    def apply_targeting_rules(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply business rules to determine who to target.
        """
        self.logger.info("Applying targeting rules...")
        
        df = df.copy()
        
        # Rule 1: Minimum expected value
        min_ev = self.decision_config.get('min_expected_value', 0.0)
        df['target'] = df['expected_value'] > min_ev
        
        # Rule 2: Churn probability thresholds
        min_prob = self.decision_config['probability_thresholds'].get('min_churn_probability', 0.30)
        max_prob = self.decision_config['probability_thresholds'].get('max_churn_probability', 0.90)
        
        df['target'] = df['target'] & (df['churn_probability'] >= min_prob) & (df['churn_probability'] <= max_prob)
        
        # DEBUG: Log filtering steps
        self.logger.info(f"After EV filter (EV > {min_ev}): {df['target'].sum()} customers")
        prob_filter = (df['churn_probability'] >= min_prob) & (df['churn_probability'] <= max_prob)
        self.logger.info(f"Probability filter ({min_prob} <= p <= {max_prob}): {prob_filter.sum()} customers")
        self.logger.info(f"After probability filter: {df['target'].sum()} customers")
        
        # Rule 3: Budget constraints
        budget = self.config.get('budget', {}).get('max_monthly_budget', None)
        
        if budget is None:
            # UNCONSTRAINED MODE: No budget limit
            self.logger.info("Budget constraint: NONE (unconstrained mode)")
            df['within_budget'] = df['target']  # All targeted customers are within budget
        else:
            # CONSTRAINED MODE: Apply budget limit
            self.logger.info(f"Budget constraint: ${budget:,.2f}")
            
            # Sort ONLY the customers who passed previous filters by expected value
            df['within_budget'] = False
            eligible = df[df['target']].copy()
            
            if len(eligible) > 0:
                eligible_sorted = eligible.sort_values('expected_value', ascending=False)
                cumulative_cost = eligible_sorted['retention_cost'].cumsum()
                
                within_budget_indices = eligible_sorted[cumulative_cost <= budget].index
                df.loc[within_budget_indices, 'within_budget'] = True
                
                self.logger.info(f"Customers within budget: {len(within_budget_indices)}")
        
        df['target'] = df['target'] & df['within_budget']
        
        return df
    
    def recommend_actions(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recommend specific retention actions based on segment and risk.
        """
        df = df.copy()
        
        # Determine risk level
        df['risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.4, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # Map to recommended action
        strategies = self.config['interventions'].get('strategy', {})
        
        def get_action(row):
            key = f"{row['segment']}_{row['risk_level']}_risk"
            return strategies.get(key, {}).get('action', 'monitor')
        
        df['recommended_action'] = df.apply(get_action, axis=1)
        
        return df
    
    def run(
        self,
        scores_file: str
    ) -> pd.DataFrame:
        """
        Execute decision engine pipeline.
        
        Args:
            scores_file: Path to batch scores parquet file
            
        Returns:
            DataFrame with targeting decisions
        """
        self.logger.info("="*60)
        self.logger.info("DECISION ENGINE")
        self.logger.info("="*60)
        
        # Load scores
        df = pd.read_parquet(scores_file)
        self.logger.info(f"Loaded {len(df)} customer scores")
        
        # Compute expected values
        df = self.compute_expected_value(df)
        
        # Apply targeting rules
        df = self.apply_targeting_rules(df)
        
        # Recommend actions
        df = self.recommend_actions(df)
        
        # Filter to targeted customers
        df_targeted = df[df['target']].copy()
        
        # Save action plan
        output_path = Path("outputs") / "action_plan.csv"
        ensure_dir(str(output_path.parent))
        
        df_targeted[
            ['customer_id', 'churn_probability', 'clv', 'expected_value',
             'segment', 'risk_level', 'recommended_action', 'retention_cost']
        ].to_csv(output_path, index=False)
        
        self.logger.info(f"Targeted {len(df_targeted)} customers")
        self.logger.info(f"Total expected value: ${df_targeted['expected_value'].sum():.2f}")
        self.logger.info(f"Total cost: ${df_targeted['retention_cost'].sum():.2f}")
        self.logger.info(f"Action plan saved to {output_path}")
        self.logger.info("="*60)
        
        return df_targeted


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', required=True, help='Path to batch scores file')
    parser.add_argument('--config', default='configs/decision.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    engine = DecisionEngine(config)
    engine.run(args.scores)


if __name__ == '__main__':
    main()
