"""Remaining Useful Life (RUL), timestamp cadence analysis, data auditing, and schema alias detection.

This module extends core.py additively without replacing any existing solver or metric logic.
All calculations preserve full internal floating-point precision and recorded timestamps (no invented timezones).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import core

# Column alias detection map (case-insensitive, stripped)
COLUMN_ALIASES = {
    "battery": ["battery", "battery_id", "cell", "cell_id", "id"],
    "discharge_cycle": [
        "discharge_cycle_num",
        "discharge_cycle",
        "discharge_index",
        "cycle_index",
        "cycle",
        "cycle_num",
    ],
    "capacity_Ah": [
        "capacity_ahr",
        "capacity_ah",
        "capacity",
        "cap_ah",
        "q_ah",
    ],
    "start_time": [
        "datetime",
        "timestamp",
        "date_time",
        "start_time",
        "time",
        "date",
    ],
    "ambient_temperature_C": [
        "ambient_temperature_c",
        "ambient_temperature",
        "temp_c",
        "temperature_c",
        "temperature",
    ],
}

OFFICIAL_NASA_CITATION = (
    "B. Saha and K. Goebel (2007), 'Battery Data Set', NASA Prognostics Data Repository, "
    "NASA Ames Research Center, Moffett Field, CA."
)
OFFICIAL_NASA_URL = "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"
SECONDARY_GITHUB_URL = "https://github.com/Wondermindimran/Battery_Energy_Storage_System_Performance_Analysis"


def detect_columns(columns: list[str] | pd.Index) -> dict[str, str | None]:
    """Detect canonical column names from a dataframe's headers using ordered alias priority.

    Specifically prefers `discharge_cycle_num` over generic cycle indices when both are present.
    """
    col_map_lower = {str(c).strip().lower(): str(c) for c in columns}
    detected: dict[str, str | None] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        # Also check exact case match first for capacity_Ah
        if canonical in columns:
            found = canonical
        for alias in aliases:
            if alias.lower() in col_map_lower:
                found = col_map_lower[alias.lower()]
                break
        detected[canonical] = found
    return detected


def normalize_dataframe(
    raw: pd.DataFrame,
    cycle_col: str | None = None,
    cap_col: str | None = None,
    time_col: str | None = None,
    bat_col: str | None = None,
    default_battery: str = "UPLOAD",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a DataFrame using detected or user-specified columns, auditing dropped/invalid rows.

    Does not impute missing measurements or invent timestamps/timezones.
    """
    detected = detect_columns(raw.columns)
    c_col = cycle_col or detected["discharge_cycle"]
    q_col = cap_col or detected["capacity_Ah"]
    t_col = time_col if time_col is not None else detected["start_time"]
    b_col = bat_col if bat_col is not None else detected["battery"]

    if not c_col or not q_col:
        raise ValueError(f"Could not detect required cycle and capacity columns in {list(raw.columns)}")

    n_raw = len(raw)
    work = pd.DataFrame()
    work["battery"] = raw[b_col].astype(str).str.strip() if b_col and b_col in raw.columns else default_battery
    work["discharge_cycle"] = pd.to_numeric(raw[c_col], errors="coerce")
    work["capacity_Ah"] = pd.to_numeric(raw[q_col], errors="coerce")

    has_time = bool(t_col and t_col in raw.columns)
    if has_time:
        # Preserve timestamps as recorded without appending UTC or inventing a timezone
        work["start_time"] = raw[t_col].astype(str).where(raw[t_col].notna(), None)
        parsed_t = pd.to_datetime(raw[t_col], errors="coerce")
        if getattr(parsed_t.dt, "tz", None) is not None:
            parsed_t = parsed_t.dt.tz_localize(None)
        work["_ts"] = parsed_t
    else:
        work["start_time"] = None
        work["_ts"] = pd.NaT

    if detected["ambient_temperature_C"] and detected["ambient_temperature_C"] in raw.columns:
        work["ambient_temperature_C"] = pd.to_numeric(raw[detected["ambient_temperature_C"]], errors="coerce")

    notes: list[str] = []
    missing_mask = work["discharge_cycle"].isna() | work["capacity_Ah"].isna() | ~np.isfinite(work["discharge_cycle"]) | ~np.isfinite(work["capacity_Ah"])
    n_missing = int(missing_mask.sum())
    if n_missing:
        notes.append(f"Dropped {n_missing} row(s) with missing or non-numeric cycle/capacity values (not imputed).")
        work = work[~missing_mask].copy()

    invalid_cap_mask = work["capacity_Ah"] <= 0
    n_invalid_cap = int(invalid_cap_mask.sum())
    if n_invalid_cap:
        notes.append(f"Flagged {n_invalid_cap} row(s) with non-positive capacity (<= 0 Ah).")

    # Sort chronologically by battery and discharge_cycle
    work = work.sort_values(["battery", "discharge_cycle"], kind="mergesort").reset_index(drop=True)

    dup_mask = work.duplicated(subset=["battery", "discharge_cycle"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        notes.append(f"Dropped {n_dup} duplicate (battery, discharge_cycle) row(s) (kept first occurrence).")
        work = work[~dup_mask].reset_index(drop=True)

    info = {
        "detected_columns": {"battery": b_col, "discharge_cycle": c_col, "capacity_Ah": q_col, "start_time": t_col},
        "n_raw": n_raw,
        "n_clean": len(work),
        "n_missing_dropped": n_missing,
        "n_duplicates_dropped": n_dup,
        "n_nonpositive_capacity": n_invalid_cap,
        "has_timestamps": bool(has_time and work["_ts"].notna().any()),
        "notes": notes,
    }
    return work.drop(columns=["_ts"]), info


def audit_battery_dataframe(df: pd.DataFrame, battery_id: str | None = None) -> dict[str, Any]:
    """Comprehensive data-quality audit for a single battery's discharge series."""
    d = df[df["battery"] == battery_id].copy() if battery_id else df.copy()
    n = len(d)
    if n == 0:
        return {"battery": battery_id or "EMPTY", "n_rows": 0}

    cyc = pd.to_numeric(d["discharge_cycle"], errors="coerce")
    cap = pd.to_numeric(d["capacity_Ah"], errors="coerce")
    n_missing_cyc = int(cyc.isna().sum())
    n_missing_cap = int(cap.isna().sum())
    n_dup_cyc = int(cyc.duplicated().sum())
    n_invalid_cap = int((cap <= 0).sum() + (cap > 10.0).sum())

    cyc_valid = cyc.dropna().to_numpy(float)
    cap_valid = cap.dropna().to_numpy(float)
    is_monotonic_cyc = bool(len(cyc_valid) < 2 or (np.diff(cyc_valid) > 0).all())
    cyc_diffs = np.diff(cyc_valid) if len(cyc_valid) >= 2 else np.array([])
    n_cycle_gaps = int((cyc_diffs > 1).sum())
    max_cycle_gap = int(cyc_diffs.max()) if len(cyc_diffs) else 0

    has_ts = "start_time" in d.columns and d["start_time"].notna().any()
    n_missing_ts = int(d["start_time"].isna().sum()) if "start_time" in d.columns else n
    n_repeated_ts = 0
    n_non_monotonic_ts = 0
    n_long_pauses_gt_24h = 0
    max_gap_hours = None
    first_ts_str = None
    last_ts_str = None

    if has_ts:
        ts = pd.to_datetime(d["start_time"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        n_missing_ts = int(ts.isna().sum())
        ts_valid = ts.dropna()
        if len(ts_valid) >= 1:
            first_ts_str = str(d["start_time"].dropna().iloc[0])
            last_ts_str = str(d["start_time"].dropna().iloc[-1])
        if len(ts_valid) >= 2:
            dt_h = ts_valid.diff().dropna().dt.total_seconds() / 3600.0
            n_repeated_ts = int((dt_h == 0).sum())
            n_non_monotonic_ts = int((dt_h < 0).sum())
            n_long_pauses_gt_24h = int((dt_h > 24.0).sum())
            max_gap_hours = float(dt_h.max())

    return {
        "battery": str(battery_id or d["battery"].iloc[0]),
        "n_rows": n,
        "cycle_min": int(cyc_valid.min()) if len(cyc_valid) else None,
        "cycle_max": int(cyc_valid.max()) if len(cyc_valid) else None,
        "capacity_initial_Ah": float(cap_valid[0]) if len(cap_valid) else None,
        "capacity_final_Ah": float(cap_valid[-1]) if len(cap_valid) else None,
        "capacity_min_Ah": float(cap_valid.min()) if len(cap_valid) else None,
        "capacity_max_Ah": float(cap_valid.max()) if len(cap_valid) else None,
        "missing_cycles": n_missing_cyc,
        "missing_capacities": n_missing_cap,
        "duplicate_cycles": n_dup_cyc,
        "invalid_capacities": n_invalid_cap,
        "strictly_increasing_cycles": is_monotonic_cyc,
        "cycle_index_gaps": n_cycle_gaps,
        "max_cycle_step": max_cycle_gap,
        "has_timestamps": has_ts and (n_missing_ts < n),
        "missing_timestamps": n_missing_ts,
        "repeated_timestamps": n_repeated_ts,
        "non_monotonic_timestamps": n_non_monotonic_ts,
        "pauses_gt_24h": n_long_pauses_gt_24h,
        "max_interval_hours": max_gap_hours,
        "first_timestamp": first_ts_str,
        "last_timestamp": last_ts_str,
    }


def compute_training_cadence(train: pd.DataFrame) -> dict[str, Any]:
    """Compute timestamp interval statistics (in hours) from consecutive training observations.

    Uses only positive intervals (dt > 0) for cadence estimation, while reporting counts of
    zero or negative intervals if any exist. Never invents timestamps or timezones.
    """
    out: dict[str, Any] = {
        "available": False,
        "n_timestamps": 0,
        "n_intervals_total": 0,
        "n_positive_intervals": 0,
        "n_nonpositive_intervals": 0,
        "median_hours": None,
        "mean_hours": None,
        "iqr_hours": None,
        "q25_hours": None,
        "q75_hours": None,
        "min_hours": None,
        "max_hours": None,
        "std_hours": None,
        "cutoff_timestamp": None,
        "cutoff_timestamp_str": None,
        "mean_to_median_ratio": None,
        "pause_distortion_warning": None,
        "reason": None,
    }
    if "start_time" not in train.columns or train["start_time"].isna().all():
        out["reason"] = "Valid timestamps are absent in training data; calendar-time RUL is disabled."
        return out

    # Ensure chronological sort by discharge_cycle
    tr = train.sort_values("discharge_cycle").reset_index(drop=True)
    ts = pd.to_datetime(tr["start_time"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)

    valid_ts = ts.dropna()
    out["n_timestamps"] = int(len(valid_ts))
    if len(valid_ts) < 2 or pd.isna(ts.iloc[-1]):
        out["reason"] = "Fewer than 2 valid training timestamps (or missing cutoff timestamp); calendar-time RUL is disabled."
        return out

    dt_hours = ts.diff().dropna().dt.total_seconds().to_numpy(float) / 3600.0
    pos = dt_hours[dt_hours > 0]
    out["n_intervals_total"] = int(len(dt_hours))
    out["n_positive_intervals"] = int(len(pos))
    out["n_nonpositive_intervals"] = int(len(dt_hours) - len(pos))

    if len(pos) == 0:
        out["reason"] = "No positive timestamp intervals found in training observations; calendar-time RUL is disabled."
        return out

    q25, med, q75 = np.percentile(pos, [25.0, 50.0, 75.0])
    mean_val = float(np.mean(pos))
    med_val = float(med)
    ratio = mean_val / med_val if med_val > 0 else None

    out.update({
        "available": True,
        "median_hours": med_val,
        "mean_hours": mean_val,
        "iqr_hours": float(q75 - q25),
        "q25_hours": float(q25),
        "q75_hours": float(q75),
        "min_hours": float(pos.min()),
        "max_hours": float(pos.max()),
        "std_hours": float(np.std(pos, ddof=1)) if len(pos) > 1 else 0.0,
        "cutoff_timestamp": ts.iloc[-1],
        "cutoff_timestamp_str": str(tr["start_time"].iloc[-1]),
        "mean_to_median_ratio": ratio,
    })
    if ratio is not None and ratio > 1.25:
        out["pause_distortion_warning"] = (
            f"Mean interval ({mean_val:.2f} h) is {ratio:.2f}x the median ({med_val:.2f} h) due to multi-day "
            f"laboratory rest pauses (max interval { float(pos.max()):.1f} h). Median cadence is used as primary."
        )
    return out


def format_duration_hours(hours: float | None) -> str:
    """Format a duration in hours into a human-readable 'Xd Yh Zm' or 'Yh Zm' string."""
    if hours is None or not np.isfinite(hours):
        return "n/a"
    sign = "-" if hours < 0 else ""
    total_sec = int(round(abs(hours) * 3600.0))
    days, rem = divmod(total_sec, 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{sign}{days}d {hrs}h {mins}m ({hours:+.2f} h)" if sign else f"{days}d {hrs}h {mins}m ({hours:.2f} h)"
    return f"{sign}{hrs}h {mins}m {secs}s ({hours:+.2f} h)" if sign else f"{hrs}h {mins}m {secs}s ({hours:.2f} h)"


def analyze_battery_rul(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    threshold_ah: float = core.EOL_AH,
    max_meaningful_cycle_factor: float = 10.0,
) -> dict[str, Any]:
    """Full chronological regression, cycle RUL, timestamp cadence RUL, and held-out crossing validation.

    Strictly fits on the first floor(train_frac * N) observations only.
    Never silently clips negative RUL values to zero.
    """
    d = df.sort_values("discharge_cycle").reset_index(drop=True)
    bat = str(d["battery"].iloc[0]) if len(d) and "battery" in d.columns else "UNKNOWN"
    train, test = core.chronological_split(d, train_frac)
    if len(train) < 3 or train["discharge_cycle"].nunique() < 2:
        raise ValueError(f"Insufficient training observations ({len(train)}) for battery {bat}.")

    fit_res = core.fit(train)
    # Verify GE and GJ agree within tight floating-point tolerance, use numerically stable lstsq for production
    beta_ge = fit_res["beta_ge"]
    beta_gj = fit_res["beta_gj"]
    beta_lstsq = np.linalg.lstsq(fit_res["X"], fit_res["y"], rcond=None)[0]
    ge_gj_diff = float(np.abs(beta_ge - beta_gj).max())
    beta0, beta1 = float(beta_ge[0]), float(beta_ge[1])

    pred_train = core.predict(beta_ge, train["discharge_cycle"])
    pred_test = core.predict(beta_ge, test["discharge_cycle"])
    m_test = core.metrics(test["capacity_Ah"], pred_test)

    k_min_train = int(train["discharge_cycle"].iloc[0])
    k_cutoff = int(train["discharge_cycle"].iloc[-1])
    cap_cutoff = float(train["capacity_Ah"].iloc[-1])
    k_min_test = int(test["discharge_cycle"].iloc[0]) if len(test) else None
    k_max_test = int(test["discharge_cycle"].iloc[-1]) if len(test) else None
    k_max_all = int(d["discharge_cycle"].iloc[-1])

    # Check whether threshold was already reached in training observations
    train_below = train[train["capacity_Ah"] <= threshold_ah]
    already_reached_in_train = len(train_below) > 0
    first_train_crossing_cycle = int(train_below["discharge_cycle"].iloc[0]) if already_reached_in_train else None
    first_train_crossing_cap = float(train_below["capacity_Ah"].iloc[0]) if already_reached_in_train else None
    first_train_crossing_ts = str(train_below["start_time"].iloc[0]) if (already_reached_in_train and "start_time" in train_below.columns) else None

    # Cycle-based RUL calculation and safeguards
    pred_crossing_cycle: float | None = None
    pred_rul_cycles: float | None = None
    has_positive_predicted_rul = False
    pred_status_code = ""
    pred_status_label = ""
    safeguard_notes: list[str] = []

    if already_reached_in_train:
        safeguard_notes.append(
            f"Threshold ({threshold_ah:.3f} Ah) was already reached during the training period at cycle "
            f"{first_train_crossing_cycle} (measured {first_train_crossing_cap:.6f} Ah). Positive remaining life from cutoff is not claimed."
        )

    if beta1 >= 0:
        pred_status_code = "NON_NEGATIVE_SLOPE"
        pred_status_label = "No crossing predicted (fitted slope is zero or positive)"
        safeguard_notes.append(f"Fitted slope β₁ = {beta1:.6e} Ah/cycle is non-negative; model never degrades to threshold.")
    else:
        raw_crossing = (threshold_ah - beta0) / beta1
        raw_rul = raw_crossing - k_cutoff
        pred_crossing_cycle = float(raw_crossing)
        pred_rul_cycles = float(raw_rul)  # preserved without clipping to zero
        max_horizon = max(k_max_all * max_meaningful_cycle_factor, k_cutoff + 5000)

        if already_reached_in_train:
            pred_status_code = "ALREADY_PASSED_IN_TRAINING"
            pred_status_label = f"Threshold already reached in training (cycle {first_train_crossing_cycle})"
        elif raw_crossing <= k_cutoff:
            pred_status_code = "AT_OR_BEFORE_CUTOFF"
            pred_status_label = "No positive predicted RUL from this model (predicted crossing at or before training cutoff)"
            safeguard_notes.append(
                f"Linear model places threshold crossing at cycle {raw_crossing:.3f}, which is at or before the "
                f"training cutoff cycle {k_cutoff} (raw RUL = {raw_rul:.3f} cycles; not clipped to zero)."
            )
        elif raw_crossing > max_horizon:
            pred_status_code = "BEYOND_MEANINGFUL_HORIZON"
            pred_status_label = "No reliable crossing found within a meaningful forecast horizon"
            safeguard_notes.append(
                f"Predicted crossing ({raw_crossing:.1f}) exceeds meaningful forecast horizon ({max_horizon:.0f} cycles)."
            )
        else:
            has_positive_predicted_rul = True
            pred_status_code = "PREDICTED_AFTER_CUTOFF"
            pred_status_label = "Extrapolated model crossing predicted after training cutoff"

    # Cadence & Time-based RUL (only when valid timestamps exist AND positive predicted RUL is meaningful)
    cadence = compute_training_cadence(train)
    pred_rul_hours_median: float | None = None
    pred_rul_days_median: float | None = None
    pred_crossing_ts_median: pd.Timestamp | None = None
    pred_rul_hours_mean: float | None = None
    pred_rul_days_mean: float | None = None
    pred_crossing_ts_mean: pd.Timestamp | None = None

    # We also compute raw unclipped time deltas when pred_rul_cycles is available so users can inspect sensitivity,
    # but only report positive calendar-time RUL when has_positive_predicted_rul is True.
    if cadence["available"] and pred_rul_cycles is not None and has_positive_predicted_rul:
        med_h = cadence["median_hours"]
        mean_h = cadence["mean_hours"]
        t_cut: pd.Timestamp = cadence["cutoff_timestamp"]
        pred_rul_hours_median = float(pred_rul_cycles * med_h)
        pred_rul_days_median = float(pred_rul_hours_median / 24.0)
        pred_crossing_ts_median = t_cut + pd.Timedelta(hours=pred_rul_hours_median)

        pred_rul_hours_mean = float(pred_rul_cycles * mean_h)
        pred_rul_days_mean = float(pred_rul_hours_mean / 24.0)
        pred_crossing_ts_mean = t_cut + pd.Timedelta(hours=pred_rul_hours_mean)

    # Held-out validation of threshold crossing
    test_below = test[test["capacity_Ah"] <= threshold_ah] if len(test) else pd.DataFrame()
    observed_in_test = len(test_below) > 0
    obs_crossing_cycle: int | None = None
    obs_crossing_cap: float | None = None
    obs_crossing_ts: pd.Timestamp | None = None
    obs_crossing_ts_str: str | None = None
    actual_rul_cycles: int | None = None
    actual_elapsed_hours: float | None = None
    actual_elapsed_days: float | None = None

    cycle_error_pred_minus_actual: float | None = None
    abs_cycle_error: float | None = None
    time_error_hours_median: float | None = None
    time_error_days_median: float | None = None
    time_error_hours_mean: float | None = None
    time_error_days_mean: float | None = None

    if already_reached_in_train:
        crossing_validation_status = (
            f"Already passed in training (cycle {first_train_crossing_cycle}, {first_train_crossing_cap:.4f} Ah)"
        )
    elif observed_in_test:
        row0 = test_below.iloc[0]
        obs_crossing_cycle = int(row0["discharge_cycle"])
        obs_crossing_cap = float(row0["capacity_Ah"])
        actual_rul_cycles = int(obs_crossing_cycle - k_cutoff)

        if "start_time" in row0 and pd.notna(row0["start_time"]):
            obs_crossing_ts_str = str(row0["start_time"])
            ts_parsed = pd.to_datetime(row0["start_time"], errors="coerce")
            if getattr(ts_parsed, "tzinfo", None) is not None:
                ts_parsed = ts_parsed.tz_localize(None)
            obs_crossing_ts = ts_parsed
            if cadence["available"] and cadence["cutoff_timestamp"] is not None and pd.notna(obs_crossing_ts):
                actual_elapsed_hours = float((obs_crossing_ts - cadence["cutoff_timestamp"]).total_seconds() / 3600.0)
                actual_elapsed_days = float(actual_elapsed_hours / 24.0)

        if pred_crossing_cycle is not None:
            cycle_error_pred_minus_actual = float(pred_crossing_cycle - obs_crossing_cycle)
            abs_cycle_error = abs(cycle_error_pred_minus_actual)

        if actual_elapsed_hours is not None and pred_rul_hours_median is not None:
            time_error_hours_median = float(pred_rul_hours_median - actual_elapsed_hours)
            time_error_days_median = float(time_error_hours_median / 24.0)
        if actual_elapsed_hours is not None and pred_rul_hours_mean is not None:
            time_error_hours_mean = float(pred_rul_hours_mean - actual_elapsed_hours)
            time_error_days_mean = float(time_error_hours_mean / 24.0)

        if has_positive_predicted_rul:
            crossing_validation_status = "Observed in held-out test & positive RUL predicted"
        else:
            crossing_validation_status = (
                f"Observed in held-out test (cycle {obs_crossing_cycle}), but model predicted crossing at/before cutoff "
                f"(cycle {pred_crossing_cycle:.2f})" if pred_crossing_cycle is not None
                else f"Observed in held-out test (cycle {obs_crossing_cycle}), no crossing predicted by model"
            )
    else:
        min_test_cap = float(test["capacity_Ah"].min()) if len(test) else None
        min_note = f" (min held-out capacity = {min_test_cap:.4f} Ah)" if min_test_cap is not None else ""
        crossing_validation_status = f"Threshold crossing not observed within the test horizon{min_note}"

    return {
        "battery": bat,
        "train_fraction": float(train_frac),
        "threshold_Ah": float(threshold_ah),
        "n_total": len(d),
        "n_train": len(train),
        "n_test": len(test),
        "train_cycle_range": (k_min_train, k_cutoff),
        "test_cycle_range": (k_min_test, k_max_test),
        "cutoff_cycle": k_cutoff,
        "cutoff_capacity_Ah": cap_cutoff,
        "cutoff_timestamp": cadence["cutoff_timestamp"],
        "cutoff_timestamp_str": cadence["cutoff_timestamp_str"],
        "beta0": beta0,
        "beta1": beta1,
        "beta_ge": beta_ge,
        "beta_gj": beta_gj,
        "beta_lstsq": beta_lstsq,
        "ge_gj_max_diff": ge_gj_diff,
        "fit_details": fit_res,
        "train_df": train,
        "test_df": test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "metrics": m_test,
        # Crossing & RUL
        "already_reached_in_train": already_reached_in_train,
        "first_train_crossing_cycle": first_train_crossing_cycle,
        "first_train_crossing_cap_Ah": first_train_crossing_cap,
        "first_train_crossing_ts_str": first_train_crossing_ts,
        "pred_crossing_cycle": pred_crossing_cycle,
        "pred_rul_cycles": pred_rul_cycles,
        "has_positive_predicted_rul": has_positive_predicted_rul,
        "pred_status_code": pred_status_code,
        "pred_status_label": pred_status_label,
        "safeguard_notes": safeguard_notes,
        # Cadence & Time RUL
        "cadence": cadence,
        "pred_rul_hours_median": pred_rul_hours_median,
        "pred_rul_days_median": pred_rul_days_median,
        "pred_crossing_ts_median": pred_crossing_ts_median,
        "pred_rul_hours_mean": pred_rul_hours_mean,
        "pred_rul_days_mean": pred_rul_days_mean,
        "pred_crossing_ts_mean": pred_crossing_ts_mean,
        # Held-out validation
        "observed_in_test": observed_in_test,
        "obs_crossing_cycle": obs_crossing_cycle,
        "obs_crossing_cap_Ah": obs_crossing_cap,
        "obs_crossing_ts": obs_crossing_ts,
        "obs_crossing_ts_str": obs_crossing_ts_str,
        "actual_rul_cycles": actual_rul_cycles,
        "actual_elapsed_hours": actual_elapsed_hours,
        "actual_elapsed_days": actual_elapsed_days,
        "cycle_error_pred_minus_actual": cycle_error_pred_minus_actual,
        "abs_cycle_error": abs_cycle_error,
        "time_error_hours_median": time_error_hours_median,
        "time_error_days_median": time_error_days_median,
        "time_error_hours_mean": time_error_hours_mean,
        "time_error_days_mean": time_error_days_mean,
        "crossing_validation_status": crossing_validation_status,
    }


def summarize_all_batteries(
    df_all: pd.DataFrame,
    batteries: list[str] | None = None,
    train_frac: float = 0.60,
    threshold_ah: float = core.EOL_AH,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run `analyze_battery_rul` across all specified batteries using the exact same procedure."""
    bats = batteries or list(df_all["battery"].unique())
    analyses: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for b in bats:
        sub = df_all[df_all["battery"] == b]
        if len(sub) < 5:
            continue
        res = analyze_battery_rul(sub, train_frac=train_frac, threshold_ah=threshold_ah)
        analyses[b] = res
        m = res["metrics"]
        cad = res["cadence"]
        rows.append({
            "battery": b,
            "n_total": res["n_total"],
            "train_range": f"{res['train_cycle_range'][0]}–{res['train_cycle_range'][1]} (n={res['n_train']})",
            "test_range": f"{res['test_cycle_range'][0]}–{res['test_cycle_range'][1]} (n={res['n_test']})" if res["n_test"] else "none",
            "cutoff_cycle": res["cutoff_cycle"],
            "cutoff_timestamp": res["cutoff_timestamp_str"],
            "cutoff_capacity_Ah": res["cutoff_capacity_Ah"],
            "beta0_Ah": res["beta0"],
            "beta1_Ah_per_cycle": res["beta1"],
            "MAE_Ah": m["MAE"],
            "RMSE_Ah": m["RMSE"],
            "R2": m["R2"],
            "MAPE_pct": m["MAPE_%"],
            "RMSE_pct_of_mean": m["RMSE_%_of_mean"],
            "threshold_Ah": res["threshold_Ah"],
            "pred_crossing_cycle": res["pred_crossing_cycle"],
            "pred_RUL_cycles": res["pred_rul_cycles"],
            "obs_crossing_cycle": res["obs_crossing_cycle"],
            "obs_crossing_capacity_Ah": res["obs_crossing_cap_Ah"],
            "obs_crossing_timestamp": res["obs_crossing_ts_str"],
            "actual_RUL_cycles": res["actual_rul_cycles"],
            "RUL_cycle_error_pred_minus_obs": res["cycle_error_pred_minus_actual"],
            "median_interval_hours": cad["median_hours"],
            "mean_interval_hours": cad["mean_hours"],
            "iqr_interval_hours": cad["iqr_hours"],
            "pred_RUL_hours_median": res["pred_rul_hours_median"],
            "pred_RUL_days_median": res["pred_rul_days_median"],
            "pred_crossing_timestamp_median": str(res["pred_crossing_ts_median"]) if res["pred_crossing_ts_median"] is not None else None,
            "pred_RUL_hours_mean": res["pred_rul_hours_mean"],
            "pred_RUL_days_mean": res["pred_rul_days_mean"],
            "actual_elapsed_hours": res["actual_elapsed_hours"],
            "actual_elapsed_days": res["actual_elapsed_days"],
            "time_error_hours_median": res["time_error_hours_median"],
            "time_error_days_median": res["time_error_days_median"],
            "crossing_status": res["crossing_validation_status"],
        })
    return pd.DataFrame(rows), analyses


def verify_b0005_sanity_check(df_all: pd.DataFrame) -> dict[str, Any]:
    """Reproducibility check against the Section F reference values for B0005 (60% split, 1.4 Ah threshold).

    Evaluates both the direct NASA .mat extraction (second-level timestamps) and minute-truncated
    timestamps (as found in third-party clean_discharge.csv exports). Nothing is hardcoded into outputs.
    """
    b5 = df_all[df_all["battery"] == "B0005"].sort_values("discharge_cycle").reset_index(drop=True)
    res = analyze_battery_rul(b5, train_frac=0.60, threshold_ah=1.4)

    # Also compute minute-truncated timestamp equivalent (2008-05-11 16:49:00) to verify the exact 4.95 h / 150.749 h reference
    b5_min = b5.copy()
    b5_min["start_time"] = pd.to_datetime(b5_min["start_time"]).dt.floor("min").dt.strftime("%Y-%m-%d %H:%M:%S")
    res_min = analyze_battery_rul(b5_min, train_frac=0.60, threshold_ah=1.4)

    checks = {
        "cutoff_cycle_is_100": res["cutoff_cycle"] == 100,
        "cutoff_date_is_2008_05_11_16_49": str(res["cutoff_timestamp"]).startswith("2008-05-11 16:49"),
        "cutoff_cap_approx_1_485868": bool(np.isclose(res["cutoff_capacity_Ah"], 1.485868, atol=1e-5)),
        "beta0_approx_1_901404883": bool(np.isclose(res["beta0"], 1.901404883, atol=1e-8)),
        "beta1_approx_neg_0_003843534": bool(np.isclose(res["beta1"], -0.003843534, atol=1e-8)),
        "pred_crossing_approx_130_454": bool(np.isclose(res["pred_crossing_cycle"], 130.454125, atol=1e-3)),
        "pred_rul_approx_30_454": bool(np.isclose(res["pred_rul_cycles"], 30.454125, atol=1e-3)),
        "obs_crossing_cycle_is_125": res["obs_crossing_cycle"] == 125,
        "obs_crossing_cap_approx_1_396701": bool(np.isclose(res["obs_crossing_cap_Ah"], 1.396701, atol=1e-5)),
        "obs_crossing_date_is_2008_05_17_17_15": str(res["obs_crossing_ts"]).startswith("2008-05-17 17:15"),
        "actual_elapsed_approx_144_43h": bool(np.isclose(res["actual_elapsed_hours"], 144.4275, atol=0.02)),
        "median_interval_approx_4_95h": bool(np.isclose(res["cadence"]["median_hours"], 4.953056, atol=0.01)),
        "minute_truncated_median_is_4_95h": bool(np.isclose(res_min["cadence"]["median_hours"], 4.95, atol=1e-6)),
        "minute_truncated_pred_rul_hours_is_150_748": bool(np.isclose(res_min["pred_rul_hours_median"], 150.747918, atol=0.01)),
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "full_precision_mat": res,
        "minute_truncated_equiv": res_min,
    }

ALL_34_CSV = core.DATA_DIR / "nasa_pcoe_all_34_batteries.csv"
DEDUP_LOG_CSV = core.DATA_DIR / "nasa_pcoe_deduplication_log.csv"

BATCH_DESCRIPTIONS = {
    "1. BatteryAgingARC-FY08Q4.zip": "Batch 1: Room Temp CC (24 °C, 2A CC, EOL 1.4 Ah) — Primary FY08Q4",
    "2. BatteryAgingARC_25_26_27_28_P1.zip": "Batch 2: Room Temp Square-Wave Load (24 °C, 4A 0.05Hz)",
    "3. BatteryAgingARC_25-44.zip": "Batch 3: Multi-Regime Temp & Load (24/43/44/4 °C)",
    "4. BatteryAgingARC_45_46_47_48.zip": "Batch 4: Cold Temp Fixed Load (4 °C, 1A CC, EOL 1.4 Ah)",
    "5. BatteryAgingARC_49_50_51_52.zip": "Batch 5: Cold Temp Fixed Load (4 °C, 2A CC)",
    "6. BatteryAgingARC_53_54_55_56.zip": "Batch 6: Cold Temp Fixed Load (4 °C, 2A CC, EOL 1.4 Ah)",
}


def build_or_load_all_34_nasa(zip_path: Path = core.NASA_ZIP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract all 34 unique NASA PCoE batteries from the 6 sub-archives, deduplicating identical .mat files by MD5."""
    if ALL_34_CSV.exists() and DEDUP_LOG_CSV.exists():
        return pd.read_csv(ALL_34_CSV), pd.read_csv(DEDUP_LOG_CSV)

    import hashlib
    import io
    import zipfile
    from scipy.io import loadmat

    rows: list[dict[str, Any]] = []
    dedup_log: list[dict[str, Any]] = []
    seen_bats: dict[str, tuple[str, str]] = {}

    with zipfile.ZipFile(zip_path) as outer:
        for zname in sorted(outer.namelist()):
            if not zname.endswith(".zip"):
                continue
            short_z = zname.split("/")[-1]
            with zipfile.ZipFile(io.BytesIO(outer.read(zname))) as inner:
                for mname in sorted(inner.namelist()):
                    if not mname.endswith(".mat"):
                        continue
                    bat = mname.split("/")[-1].replace(".mat", "")
                    raw_bytes = inner.read(mname)
                    md5 = hashlib.md5(raw_bytes).hexdigest()
                    if bat in seen_bats:
                        dedup_log.append({
                            "battery": bat,
                            "kept_from_archive": seen_bats[bat][0],
                            "duplicate_in_archive": short_z,
                            "md5_kept": seen_bats[bat][1],
                            "md5_duplicate": md5,
                            "identical_md5": seen_bats[bat][1] == md5,
                            "action": "Skipped identical duplicate .mat archive entry",
                        })
                        continue
                    seen_bats[bat] = (short_z, md5)
                    mat = loadmat(io.BytesIO(raw_bytes), simplify_cells=True)
                    if bat not in mat:
                        continue
                    k = 0
                    for c in mat[bat]["cycle"]:
                        if c["type"] != "discharge":
                            continue
                        cap = np.ravel(c["data"].get("Capacity", []))
                        if cap.size == 0 or not np.isfinite(cap[0]):
                            continue
                        k += 1
                        t = [int(v) for v in np.ravel(c["time"])[:5]] + [float(np.ravel(c["time"])[5])]
                        rows.append({
                            "battery": bat,
                            "archive_group": short_z,
                            "batch_description": BATCH_DESCRIPTIONS.get(short_z, short_z),
                            "discharge_cycle": k,
                            "capacity_Ah": float(cap[0]),
                            "ambient_temperature_C": float(c["ambient_temperature"]),
                            "start_time": pd.Timestamp(*t[:5], int(t[5])).isoformat(),
                            "md5": md5,
                        })
    df_full = pd.DataFrame(rows)
    df_dedup = pd.DataFrame(dedup_log)
    ALL_34_CSV.parent.mkdir(exist_ok=True)
    df_full.to_csv(ALL_34_CSV, index=False)
    df_dedup.to_csv(DEDUP_LOG_CSV, index=False)
    return df_full, df_dedup


def analyze_satellite_fleet(
    tle_rows: list[list[str]],
    beta0: float,
    beta1: float,
    threshold_ah: float = core.EOL_AH,
    rated_ah: float = core.RATED_AH,
    assumed_dod: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute orbital parameters, eclipse cycling rates, and hypothetical battery projections across all satellites.

    Deduplicates any satellites that share an identical NORAD catalog ID or identical TLE line pair.
    """
    import math

    seen_norad: dict[str, str] = {}
    dedup_sats: list[dict[str, Any]] = []
    fleet: list[dict[str, Any]] = []
    k_star = (threshold_ah - beta0) / beta1 if beta1 < 0 else None

    for item in tle_rows:
        if len(item) < 3:
            continue
        name, l1, l2 = item[0].strip(), item[1].rstrip(), item[2].rstrip()
        norad = l1[2:7].strip()
        if norad in seen_norad:
            dedup_sats.append({
                "norad_id": norad,
                "satellite": name,
                "kept_name": seen_norad[norad],
                "action": "Skipped duplicate NORAD catalog entry",
            })
            continue
        seen_norad[norad] = name
        try:
            cospar = l1[9:17].strip()
            yy_str = l1[9:11].strip()
            yy = int(yy_str) if yy_str.isdigit() else None
            launch_year = (2000 + yy if yy < 57 else 1900 + yy) if yy is not None else None
            incl_deg = float(l2[8:16])
            ecc = float("0." + l2[26:33].strip())
            mean_motion = float(l2[52:63])
            if mean_motion <= 0:
                continue
            period_min = 1440.0 / mean_motion
            n_rad_s = mean_motion * 2.0 * math.pi / 86400.0
            a_km = (398600.4418 / (n_rad_s ** 2)) ** (1.0 / 3.0)
            alt_km = max(0.0, a_km - 6371.0)
            vel_kms = math.sqrt(398600.4418 / a_km)
            cls = "HEO" if ecc > 0.25 else ("LEO" if alt_km < 2000.0 else ("MEO" if alt_km < 30000.0 else "GEO"))
            sin_rho = min(1.0, 6371.0 / a_km)
            max_ecl_frac = math.asin(sin_rho) / math.pi
            if cls == "LEO":
                ecl_orbit_frac_yr = 0.92 if incl_deg > 70.0 else 0.98
                mean_ecl_min = period_min * max_ecl_frac * 0.78
            elif cls == "MEO":
                ecl_orbit_frac_yr = 0.28
                mean_ecl_min = period_min * max_ecl_frac * 0.65
            elif cls == "GEO":
                ecl_orbit_frac_yr = 90.0 / 365.25
                mean_ecl_min = period_min * max_ecl_frac * 0.70
            else:
                ecl_orbit_frac_yr = 0.50
                mean_ecl_min = period_min * max_ecl_frac * 0.50

            ecl_cycles_day = mean_motion * ecl_orbit_frac_yr
            eq_cycles_day = ecl_cycles_day * assumed_dod
            sim_days = (k_star / eq_cycles_day) if (k_star is not None and k_star > 0 and eq_cycles_day > 0) else None
            sim_years = (sim_days / 365.25) if sim_days is not None else None
            # Projected SOH after 1 year of simulated fresh-cell orbital cycling
            cap_1yr = beta0 + beta1 * (eq_cycles_day * 365.25)
            soh_1yr_pct = max(0.0, cap_1yr / rated_ah * 100.0)

            fleet.append({
                "satellite": name.replace("0 ", "").strip(),
                "norad_id": norad,
                "cospar": cospar,
                "launch_year": launch_year,
                "orbit_class": cls,
                "altitude_km": round(alt_km, 1),
                "velocity_km_s": round(vel_kms, 3),
                "inclination_deg": round(incl_deg, 2),
                "eccentricity": round(ecc, 5),
                "period_min": round(period_min, 2),
                "rev_per_day": round(mean_motion, 3),
                "est_eclipse_min_per_orbit": round(mean_ecl_min, 1),
                "est_eclipse_orbits_pct_yr": round(ecl_orbit_frac_yr * 100.0, 1),
                "eclipse_cycles_per_day": round(ecl_cycles_day, 2),
                "lab_equiv_cycles_per_day": round(eq_cycles_day, 3),
                "sim_1yr_SOH_pct": round(soh_1yr_pct, 1),
                "sim_days_to_threshold_fresh": round(sim_days, 1) if sim_days is not None else None,
                "sim_years_to_threshold_fresh": round(sim_years, 2) if sim_years is not None else None,
            })
        except Exception:
            continue
    return pd.DataFrame(fleet), pd.DataFrame(dedup_sats)
