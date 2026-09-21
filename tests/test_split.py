import pandas as pd

from src.preprocessing.split import check_temporal_leakage, temporal_split


def _transactions(dates, fraud_flags):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(dates),
        "is_fraud": fraud_flags,
    })


def test_check_temporal_leakage_detecte_un_chevauchement():
    train = _transactions(["2024-01-01", "2024-06-01"], [0, 0])
    test = _transactions(["2024-03-01", "2024-08-01"], [0, 1])  # 2024-03-01 < 2024-06-01
    assert check_temporal_leakage(train, test) is False


def test_check_temporal_leakage_valide_un_split_propre():
    train = _transactions(["2024-01-01", "2024-06-01"], [0, 0])
    test = _transactions(["2024-07-01", "2024-08-01"], [0, 1])
    assert check_temporal_leakage(train, test) is True


def test_temporal_split_respecte_le_cutoff():
    df = _transactions(
        ["2024-06-01", "2024-06-15", "2024-07-01", "2024-07-15"],
        [0, 1, 0, 1],
    )
    train, test = temporal_split(df, date_cutoff="2024-07-01")

    assert (train["timestamp"] < pd.Timestamp("2024-07-01")).all()
    assert (test["timestamp"] >= pd.Timestamp("2024-07-01")).all()
    assert len(train) + len(test) == len(df)
