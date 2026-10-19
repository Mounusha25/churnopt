"""Champion-challenger evaluation and governed model promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..utils import ensure_dir, load_config, save_json, setup_logging
from .registry import ModelRegistry

logger = setup_logging(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    roc_auc: float
    pr_auc: float
    brier_score: float
    expected_profit: float
    estimated_holdout_profit: float
    targeted_customers: int
    campaign_cost: float


class ModelPromotionGate:
    """Compare a staging model with the production champion before promotion."""

    def __init__(
        self,
        training_config: Dict[str, Any],
        decision_config: Dict[str, Any],
        registry: Optional[ModelRegistry] = None,
        report_dir: str = "outputs/promotion_reports",
    ) -> None:
        self.training_config = training_config
        self.decision_config = decision_config
        self.registry = registry or ModelRegistry(
            training_config.get("registry", {}).get("path", "models")
        )
        self.policy = training_config.get("registry", {}).get("promotion", {})
        self.report_dir = Path(report_dir)

    def _temporal_holdout(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured out-of-time test window when one is available."""
        data_config = self.training_config.get("data", {})
        start_date = data_config.get("test_start_date")
        end_date = data_config.get("test_end_date")
        if not start_date and not end_date:
            return frame.copy()
        if "snapshot_date" not in frame:
            raise ValueError(
                "Training config defines a temporal test window, but validation data "
                "does not contain snapshot_date"
            )

        snapshot_dates = pd.to_datetime(frame["snapshot_date"])
        mask = pd.Series(True, index=frame.index)
        if start_date:
            mask &= snapshot_dates >= pd.Timestamp(start_date)
        if end_date:
            mask &= snapshot_dates <= pd.Timestamp(end_date)
        holdout = frame.loc[mask].copy()
        if holdout.empty:
            raise ValueError(
                f"No validation rows exist in temporal test window {start_date} to {end_date}"
            )
        return holdout

    @staticmethod
    def _prepare_features(frame: pd.DataFrame, model_entry: Dict[str, Any]) -> pd.DataFrame:
        feature_names = model_entry.get("metadata", {}).get("feature_names")
        if not feature_names:
            raise ValueError("Model metadata does not contain feature_names")

        missing = sorted(set(feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Validation data is missing model features: {missing}")

        features = frame.loc[:, feature_names].copy()
        # Legacy artifacts predate persisted preprocessing metadata and were
        # trained with pandas categorical codes. Reproduce that transformation
        # for their full temporal holdout. New artifacts should store an sklearn
        # pipeline or explicit category vocabulary in model metadata.
        for column in features.select_dtypes(exclude=[np.number]).columns:
            features[column] = pd.Categorical(features[column]).codes
        return features.fillna(0)

    def _business_metrics(
        self,
        frame: pd.DataFrame,
        probabilities: np.ndarray,
        labels: pd.Series,
    ) -> Dict[str, Any]:
        retention = self.decision_config["retention"]
        clv_config = self.decision_config["clv"]["simple"]
        decision = self.decision_config["decision"]
        variable_cost = retention["costs"]["variable_cost_model"]

        monthly_revenue = (
            frame["monthly_charges"].astype(float)
            if "monthly_charges" in frame
            else float(clv_config.get("monthly_revenue", 60.0))
        )
        clv = (
            monthly_revenue
            * float(clv_config["default_lifetime_months"])
            * float(clv_config["margin"])
        )
        success_rate = float(retention.get("success_rate", 0.40))
        cost_tp = float(variable_cost["cost_true_positive"])
        cost_fp = float(variable_cost["cost_false_positive"])

        scores = pd.DataFrame(
            {
                "probability": probabilities,
                "label": labels.astype(int).to_numpy(),
                "clv": np.asarray(clv) if isinstance(clv, pd.Series) else clv,
            },
            index=frame.index,
        )
        scores["expected_cost"] = (
            scores["probability"] * cost_tp + (1 - scores["probability"]) * cost_fp
        )
        scores["expected_value"] = (
            scores["probability"] * (success_rate * scores["clv"] - cost_tp)
            - (1 - scores["probability"]) * cost_fp
        )

        thresholds = decision["probability_thresholds"]
        eligible = scores[
            (scores["expected_value"] > float(decision.get("min_expected_value", 0.0)))
            & (scores["probability"] >= float(thresholds["min_churn_probability"]))
            & (scores["probability"] <= float(thresholds["max_churn_probability"]))
        ].sort_values("expected_value", ascending=False)

        budget = self.decision_config.get("budget", {}).get("max_monthly_budget")
        if budget is not None:
            eligible = eligible[eligible["expected_cost"].cumsum() <= float(budget)]

        eligible["realized_value"] = np.where(
            eligible["label"] == 1,
            success_rate * eligible["clv"] - cost_tp,
            -cost_fp,
        )
        return {
            "expected_profit": float(eligible["expected_value"].sum()),
            "estimated_holdout_profit": float(eligible["realized_value"].sum()),
            "targeted_customers": int(len(eligible)),
            "campaign_cost": float(eligible["expected_cost"].sum()),
        }

    def evaluate(
        self,
        model_entry: Dict[str, Any],
        frame: pd.DataFrame,
        label_column: str,
    ) -> EvaluationMetrics:
        if label_column not in frame:
            raise ValueError(f"Validation label column not found: {label_column}")
        labels = frame[label_column].astype(int)
        if labels.nunique() < 2:
            raise ValueError("Validation labels must contain both classes")

        model = joblib.load(model_entry["model_path"])
        features = self._prepare_features(frame, model_entry)
        probabilities = np.asarray(model.predict_proba(features))[:, 1]
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Model produced invalid probabilities")

        business = self._business_metrics(frame, probabilities, labels)
        return EvaluationMetrics(
            roc_auc=float(roc_auc_score(labels, probabilities)),
            pr_auc=float(average_precision_score(labels, probabilities)),
            brier_score=float(brier_score_loss(labels, probabilities)),
            **business,
        )

    def _quality_checks(
        self,
        candidate: EvaluationMetrics,
        champion: Optional[EvaluationMetrics],
    ) -> Dict[str, bool]:
        checks = {
            "minimum_roc_auc": candidate.roc_auc >= float(self.policy.get("min_roc_auc", 0.0)),
            "minimum_pr_auc": candidate.pr_auc >= float(self.policy.get("min_pr_auc", 0.0)),
            "maximum_brier_score": candidate.brier_score
            <= float(self.policy.get("max_brier_score", 1.0)),
            "minimum_estimated_holdout_profit": candidate.estimated_holdout_profit
            >= float(self.policy.get("min_estimated_holdout_profit", float("-inf"))),
        }
        if champion is not None:
            baseline = champion.roc_auc
            relative_gain = (candidate.roc_auc - baseline) / max(abs(baseline), 1e-12)
            checks["roc_auc_improvement"] = relative_gain >= float(
                self.policy.get("min_improvement_over_production", 0.0)
            )
            allowed_profit_regression = float(
                self.policy.get("max_estimated_holdout_profit_regression", 0.0)
            )
            checks["business_value_not_regressed"] = candidate.estimated_holdout_profit >= (
                champion.estimated_holdout_profit - allowed_profit_regression
            )
        return checks

    def run(
        self,
        candidate_version: str,
        frame: pd.DataFrame,
        label_column: str = "churn_label_next_period",
        promote: bool = False,
    ) -> Dict[str, Any]:
        candidate_entry = self.registry.get_model(candidate_version)
        if candidate_entry is None:
            raise ValueError(f"Candidate model not found: {candidate_version}")
        if candidate_entry["status"] == "production":
            raise ValueError("Candidate is already the production model")

        holdout = self._temporal_holdout(frame)
        champion_entry = self.registry.get_production_model()
        candidate_metrics = self.evaluate(candidate_entry, holdout, label_column)
        champion_metrics = (
            self.evaluate(champion_entry, holdout, label_column) if champion_entry else None
        )
        checks = self._quality_checks(candidate_metrics, champion_metrics)
        approved = all(checks.values())

        report: Dict[str, Any] = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_version": candidate_version,
            "champion_version": champion_entry["model_version"] if champion_entry else None,
            "candidate_metrics": asdict(candidate_metrics),
            "champion_metrics": asdict(champion_metrics) if champion_metrics else None,
            "checks": checks,
            "approved": approved,
            "promotion_requested": promote,
            "promoted": False,
            "validation": {
                "rows": len(holdout),
                "start_date": (
                    str(pd.to_datetime(holdout["snapshot_date"]).min().date())
                    if "snapshot_date" in holdout
                    else None
                ),
                "end_date": (
                    str(pd.to_datetime(holdout["snapshot_date"]).max().date())
                    if "snapshot_date" in holdout
                    else None
                ),
            },
        }

        if promote and approved:
            report["promotion"] = self.registry.promote_to_production(
                candidate_version,
                reason="champion-challenger quality gate passed",
            )
            report["promoted"] = True

        ensure_dir(str(self.report_dir))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"promotion_{candidate_version}_{timestamp}.json"
        report["report_path"] = str(report_path)
        save_json(report, str(report_path))
        logger.info("Promotion decision saved to %s", report_path)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate model for production")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--label-column", default="churn_label_next_period")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--decision-config", default="configs/decision.yaml")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    validation_path = Path(args.validation_data)
    frame = (
        pd.read_parquet(validation_path)
        if validation_path.suffix.lower() == ".parquet"
        else pd.read_csv(validation_path)
    )
    gate = ModelPromotionGate(
        load_config(args.training_config),
        load_config(args.decision_config),
    )
    report = gate.run(args.candidate_version, frame, args.label_column, args.promote)
    print(f"approved={report['approved']} promoted={report['promoted']}")
    if not report["approved"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise SystemExit(f"Promotion gate failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
