from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.promotion import ModelPromotionGate
from src.models.registry import ModelRegistry


class SignalModel:
    """Small deterministic model used to exercise promotion policy behavior."""

    def __init__(self, feature: str, direction: float = 1.0):
        self.feature = feature
        self.direction = direction

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = self.direction * frame[self.feature].to_numpy()
        probability = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probability, probability])


@pytest.fixture
def validation_frame() -> pd.DataFrame:
    signal = np.linspace(-2.0, 2.0, 80)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": np.sin(np.arange(80) * 2.7),
            "monthly_charges": np.full(80, 100.0),
            "churn_label_next_period": (signal > 0).astype(int),
        }
    )


@pytest.fixture
def configs():
    training = {
        "registry": {
            "promotion": {
                "min_roc_auc": 0.70,
                "min_pr_auc": 0.70,
                "max_brier_score": 0.25,
                "min_improvement_over_production": 0.01,
                "min_estimated_holdout_profit": 0.0,
                "max_estimated_holdout_profit_regression": 0.0,
            }
        }
    }
    decision = {
        "clv": {
            "simple": {
                "monthly_revenue": 60.0,
                "default_lifetime_months": 24,
                "margin": 0.30,
            }
        },
        "retention": {
            "success_rate": 0.40,
            "costs": {
                "variable_cost_model": {
                    "cost_true_positive": 50.0,
                    "cost_false_positive": 300.0,
                }
            },
        },
        "decision": {
            "min_expected_value": 0.0,
            "probability_thresholds": {
                "min_churn_probability": 0.30,
                "max_churn_probability": 0.90,
            },
        },
        "budget": {"max_monthly_budget": 50_000.0},
    }
    return training, decision


def register_model(
    registry: ModelRegistry,
    directory: Path,
    version: str,
    model: SignalModel,
    feature_names: list[str],
) -> None:
    model_path = directory / f"{version}.pkl"
    joblib.dump(model, model_path)
    registry.register_model(
        str(model_path),
        version,
        metadata={"feature_names": feature_names, "metrics": {}},
    )


def test_approved_candidate_is_promoted_and_reported(
    tmp_path: Path,
    validation_frame: pd.DataFrame,
    configs,
):
    registry = ModelRegistry(str(tmp_path / "registry"))
    register_model(registry, tmp_path, "champion", SignalModel("noise"), ["noise"])
    register_model(registry, tmp_path, "challenger", SignalModel("signal"), ["signal"])
    registry.promote_to_production("champion", reason="test baseline")

    gate = ModelPromotionGate(
        *configs,
        registry=registry,
        report_dir=str(tmp_path / "reports"),
    )
    report = gate.run("challenger", validation_frame, promote=True)

    assert report["approved"] is True
    assert report["promoted"] is True
    assert all(report["checks"].values())
    assert registry.get_production_model()["model_version"] == "challenger"
    assert Path(report["report_path"]).exists()


def test_rejected_candidate_cannot_replace_production(
    tmp_path: Path,
    validation_frame: pd.DataFrame,
    configs,
):
    registry = ModelRegistry(str(tmp_path / "registry"))
    register_model(registry, tmp_path, "champion", SignalModel("signal"), ["signal"])
    register_model(
        registry,
        tmp_path,
        "bad-challenger",
        SignalModel("signal", direction=-1.0),
        ["signal"],
    )
    registry.promote_to_production("champion", reason="test baseline")

    gate = ModelPromotionGate(
        *configs,
        registry=registry,
        report_dir=str(tmp_path / "reports"),
    )
    report = gate.run("bad-challenger", validation_frame, promote=True)

    assert report["approved"] is False
    assert report["promoted"] is False
    assert registry.get_production_model()["model_version"] == "champion"


def test_unknown_promotion_does_not_archive_production(tmp_path: Path):
    registry = ModelRegistry(str(tmp_path / "registry"))
    register_model(registry, tmp_path, "champion", SignalModel("signal"), ["signal"])
    registry.promote_to_production("champion", reason="test baseline")

    with pytest.raises(ValueError, match="not found"):
        registry.promote_to_production("missing", reason="should fail")

    assert registry.get_production_model()["model_version"] == "champion"


def test_validation_schema_must_match_model_metadata(
    tmp_path: Path,
    validation_frame: pd.DataFrame,
    configs,
):
    registry = ModelRegistry(str(tmp_path / "registry"))
    register_model(registry, tmp_path, "challenger", SignalModel("signal"), ["missing"])
    gate = ModelPromotionGate(
        *configs,
        registry=registry,
        report_dir=str(tmp_path / "reports"),
    )

    with pytest.raises(ValueError, match="missing model features"):
        gate.run("challenger", validation_frame)


def test_configured_temporal_holdout_excludes_training_rows(tmp_path: Path, configs):
    training, decision = configs
    training["data"] = {
        "test_start_date": "2024-10-01",
        "test_end_date": "2024-12-31",
    }
    gate = ModelPromotionGate(
        training,
        decision,
        registry=ModelRegistry(str(tmp_path / "registry")),
    )
    frame = pd.DataFrame(
        {
            "snapshot_date": ["2024-09-01", "2024-10-01", "2024-12-01", "2025-01-01"],
            "signal": [0.0, 1.0, 2.0, 3.0],
        }
    )

    holdout = gate._temporal_holdout(frame)

    assert holdout["signal"].tolist() == [1.0, 2.0]
