"""
Monitoring module — detects performance degradation and distribution drift
to trigger retraining in the Continuous Training pipeline.
"""

import pandas as pd
from scipy import stats


def check_performance_drop(
    current_f1: float,
    baseline_f1: float,
    threshold_pct: float = 0.05
) -> bool:
    """
    Return True if F1 score dropped by more than threshold_pct relative
    to the baseline (e.g. previous evaluation window).
    """
    drop = (baseline_f1 - current_f1) / (baseline_f1 + 1e-9)
    triggered = drop > threshold_pct
    print(
        f"[monitor] F1 baseline={baseline_f1:.4f} current={current_f1:.4f} "
        f"drop={drop*100:.2f}% trigger={triggered}"
    )
    return triggered


def check_volatility_spike(
    df: pd.DataFrame,
    window: int = 7,
    multiplier: float = 2.0
) -> bool:
    """
    Return True if the latest rolling volatility exceeds multiplier times
    the long-term average volatility.
    """
    daily_return = df["price_usd"].pct_change(1)
    recent_vol = daily_return.rolling(window).std().iloc[-1]
    long_term_vol = daily_return.std()
    triggered = recent_vol > multiplier * long_term_vol
    print(
        f"[monitor] Volatility recent={recent_vol:.6f} "
        f"long_term={long_term_vol:.6f} "
        f"ratio={recent_vol/long_term_vol:.2f}x trigger={triggered}"
    )
    return triggered


def check_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = "feature",
    p_threshold: float = 0.05
) -> bool:
    """
    Kolmogorov-Smirnov test to detect distribution shift between a
    reference window and the current window for a given feature.
    """
    stat, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
    triggered = p_value < p_threshold
    print(
        f"[monitor] KS test '{feature_name}': stat={stat:.4f} "
        f"p={p_value:.4f} trigger={triggered}"
    )
    return triggered


def should_retrain(
    current_f1: float,
    baseline_f1: float,
    df: pd.DataFrame,
    reference_window: int = 90,
    f1_threshold_pct: float = 0.05,
    volatility_multiplier: float = 2.0,
    f1_floor: float = 0.60
) -> bool:
    """
    Aggregate trigger: returns True if any retraining condition is met.

    Conditions:
      1. F1 score below absolute floor
      2. F1 dropped by more than threshold vs baseline
      3. Volatility spike detected
      4. Feature drift detected on daily return
    """
    triggers = []

    # Condition 1 — absolute F1 floor
    if current_f1 < f1_floor:
        print(f"[monitor] F1 {current_f1:.4f} below floor {f1_floor} → RETRAIN")
        triggers.append("f1_floor")

    # Condition 2 — relative performance drop
    if check_performance_drop(current_f1, baseline_f1, f1_threshold_pct):
        triggers.append("f1_drop")

    # Condition 3 — volatility spike
    if check_volatility_spike(df, multiplier=volatility_multiplier):
        triggers.append("volatility_spike")

    # Condition 4 — feature drift on 1-day return
    if "return_1d" in df.columns and len(df) > reference_window * 2:
        reference = df["return_1d"].iloc[-reference_window * 2:-reference_window]
        current = df["return_1d"].iloc[-reference_window:]
        if check_feature_drift(reference, current, "return_1d"):
            triggers.append("feature_drift")

    if triggers:
        print(f"[monitor] Retraining triggered by: {triggers}")
        return True

    print("[monitor] No retraining needed.")
    return False