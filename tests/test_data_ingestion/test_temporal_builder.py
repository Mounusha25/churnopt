"""
Unit tests for the temporal dataset builder.

The builder turns one row per customer into monthly snapshots. These tests check the properties
that matter for a leak-free churn model: snapshots stay inside the observation window, move forward
one month at a time, never look past the end of the window, and carry the label only on the
snapshot exactly `prediction_horizon_months` before churn.

Run with: python -m pytest tests/test_data_ingestion/test_temporal_builder.py -v
"""

import pandas as pd
import pytest

from src.data_ingestion.temporal_builder import TemporalDatasetBuilder

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2022-12-01")
HORIZON = 1


@pytest.fixture
def config(tmp_path):
    """Config in the shape of configs/data.yaml, writing output under a temp directory."""
    return {
        "temporal": {
            "start_date": "2020-01-01",
            "end_date": "2022-12-01",
            "prediction_horizon_months": HORIZON,
        },
        "paths": {"datasets_output": str(tmp_path / "datasets")},
    }


@pytest.fixture
def builder(config):
    return TemporalDatasetBuilder(config)


def make_customer(customer_id, tenure, churn):
    """One cleaned-customer row, with the columns the builder reads."""
    return pd.Series(
        {
            "customer_id": customer_id,
            "tenure": tenure,
            "churn": churn,
            "monthly_charges": 50.0,
            "total_charges": 50.0 * tenure,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
        }
    )


@pytest.fixture
def customers():
    """A small cleaned dataset: two active customers and two who churn."""
    return pd.DataFrame(
        [
            make_customer("C001", 12, 0),
            make_customer("C002", 24, 0),
            make_customer("C003", 18, 1),
            make_customer("C004", 30, 1),
        ]
    )


class TestInitialization:
    def test_reads_dates_and_horizon_from_config(self, builder):
        assert builder.start_date == START
        assert builder.end_date == END
        assert builder.prediction_horizon == HORIZON

    def test_missing_temporal_config_raises(self):
        with pytest.raises(KeyError):
            TemporalDatasetBuilder({"temporal": {}})


class TestExpandCustomer:
    def test_active_customer_gets_monthly_snapshots_with_no_churn_label(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("A", 12, 0), START, END)

        assert len(snaps) > 0
        assert (snaps["customer_id"] == "A").all()
        assert (snaps["churn_label_next_period"] == 0).all()
        assert snaps["months_until_churn"].isna().all()

    def test_snapshots_are_month_starts_one_month_apart(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("A", 12, 0), START, END)

        dates = pd.DatetimeIndex(snaps["snapshot_date"])
        assert dates.is_month_start.all()
        assert dates.is_unique
        assert dates.is_monotonic_increasing
        months = (dates.year * 12 + dates.month).to_numpy()
        assert (months[1:] - months[:-1] == 1).all()

    def test_tenure_grows_one_month_per_snapshot(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("A", 12, 0), START, END)

        tenure = snaps["tenure_months"].to_numpy()
        assert (tenure[1:] - tenure[:-1] == 1).all()

    def test_snapshots_stay_inside_the_observation_window(self, builder):
        for churn in (0, 1):
            snaps = builder.expand_customer_to_snapshots(make_customer("A", 30, churn), START, END)
            assert snaps["snapshot_date"].min() >= START
            assert snaps["snapshot_date"].max() <= END

    def test_active_customer_is_observed_through_the_end_of_the_window(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("A", 12, 0), START, END)

        assert snaps["snapshot_date"].max() == END

    def test_churned_customer_has_no_snapshot_at_or_after_churn(self, builder):
        """No future leakage: the last snapshot comes before the end of the window."""
        snaps = builder.expand_customer_to_snapshots(make_customer("C", 24, 1), START, END)

        assert snaps["snapshot_date"].max() < END

    def test_churn_label_only_on_snapshot_exactly_one_horizon_before_churn(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("C", 24, 1), START, END)

        positives = snaps[snaps["churn_label_next_period"] == 1]
        assert len(positives) <= 1
        assert (positives["months_until_churn"] == HORIZON).all()
        # snapshots further from churn are labelled 0 and know how far away churn is
        earlier = snaps[snaps["churn_label_next_period"] == 0]
        assert (earlier["months_until_churn"] > HORIZON).all()

    def test_static_features_are_copied_to_every_snapshot(self, builder):
        snaps = builder.expand_customer_to_snapshots(make_customer("A", 12, 0), START, END)

        assert (snaps["monthly_charges"] == 50.0).all()
        assert (snaps["contract"] == "Month-to-month").all()
        assert (snaps["payment_method"] == "Electronic check").all()

    def test_longer_tenure_never_gives_fewer_snapshots(self, builder):
        short = builder.expand_customer_to_snapshots(make_customer("A", 3, 0), START, END)
        long_ = builder.expand_customer_to_snapshots(make_customer("A", 24, 0), START, END)

        assert len(long_) >= len(short) >= 1


class TestBuildTemporalDataset:
    def test_has_multiple_snapshots_per_customer(self, builder, customers):
        df = builder.build_temporal_dataset(customers)

        assert set(df["customer_id"]) == set(customers["customer_id"])
        assert len(df) > len(customers)

    def test_no_duplicate_customer_date_pairs(self, builder, customers):
        df = builder.build_temporal_dataset(customers)

        assert not df.duplicated(["customer_id", "snapshot_date"]).any()

    def test_sorted_by_customer_then_date(self, builder, customers):
        df = builder.build_temporal_dataset(customers)

        expected = df.sort_values(["customer_id", "snapshot_date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected)

    def test_labels_are_binary_and_active_customers_never_positive(self, builder, customers):
        df = builder.build_temporal_dataset(customers)

        assert df["churn_label_next_period"].isin([0, 1]).all()
        active = df[df["customer_id"].isin(["C001", "C002"])]
        assert (active["churn_label_next_period"] == 0).all()


class TestRun:
    def test_run_returns_and_saves_both_tables(self, builder, config, customers, tmp_path):
        result = builder.run(customers)

        assert set(result) == {"temporal", "customers"}
        out = tmp_path / "datasets"
        assert (out / "monthly_customer_snapshots.parquet").exists()
        assert (out / "customers_master.csv").exists()

    def test_customer_master_has_one_row_per_customer(self, builder, customers):
        result = builder.run(customers)

        master = result["customers"]
        assert len(master) == len(customers)
        assert master["customer_id"].is_unique
        assert (master["first_snapshot_date"] <= master["last_snapshot_date"]).all()
