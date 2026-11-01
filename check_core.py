"""Automated checks. Run:  python check_core.py   (exits non-zero on failure)"""
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import core
import rul_analysis

# 1. Solvers on a hand-checkable system: 2b0 + b1 = 5, b0 + 3b1 = 10  ->  b0 = 1, b1 = 3
A, b = np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([5.0, 10.0])
for solver in (core.gaussian_elimination, core.gauss_jordan):
    x, steps = solver(A, b)
    assert np.allclose(x, [1, 3]), (solver.__name__, x)
    assert len(steps) > 2
# pivoting path: zero leading entry forces a row swap
x, steps = core.gaussian_elimination(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([2.0, 3.0]))
assert np.allclose(x, [1, 2]) and any("↔" in s["op"] for s in steps)
try:
    core.gauss_jordan(np.ones((2, 2)), np.ones(2)); raise AssertionError("singular not detected")
except np.linalg.LinAlgError:
    pass

# 2. Every NASA battery and many training fractions: GE == GJ == numpy == sklearn, and no leakage
df_all = core.load_nasa()
assert set(df_all.battery) == set(core.NASA_BATTERIES)
for bat in core.NASA_BATTERIES:
    df = df_all[df_all.battery == bat].reset_index(drop=True)
    for f in (0.3, 0.6, 0.9):
        tr, te = core.chronological_split(df, f)
        F = core.fit(tr)
        assert np.allclose(F["beta_ge"], F["beta_gj"], rtol=1e-9, atol=1e-12), (bat, f)
        assert np.allclose(F["beta_ge"], np.linalg.lstsq(F["X"], F["y"], rcond=None)[0], rtol=1e-7)
        sk = LinearRegression().fit(tr[["discharge_cycle"]], tr.capacity_Ah)
        assert np.allclose(F["beta_ge"], [sk.intercept_, sk.coef_[0]], rtol=1e-7)
        assert tr.discharge_cycle.max() < te.discharge_cycle.min()
        assert len(tr) + len(te) == len(df) and F["X"].shape[0] == len(tr)
        poisoned = df.copy(); poisoned.loc[te.index, "capacity_Ah"] = -1e6
        assert np.array_equal(core.fit(core.chronological_split(poisoned, f)[0])["beta_ge"], F["beta_ge"]), "leakage"

# 3. Metrics: hand values and safe undefined cases
m = core.metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
assert np.isclose(m["MAE"], 1 / 3) and np.isclose(m["RMSE"], np.sqrt(1 / 3)) and np.isclose(m["R2"], 0.5)
assert np.isclose(m["MAPE_%"], 100 / 9) and np.isclose(m["RMSE_%_of_mean"], np.sqrt(1 / 3) / 2 * 100)
assert core.metrics([], [])["MAE"] is None
m0 = core.metrics([0.0, 0.0], [0.1, -0.1])
assert m0["R2"] is None and m0["MAPE_%"] is None and m0["RMSE_%_of_mean"] is None
assert core.threshold_cycle([2.0, 0.01], 1.4) is None and np.isclose(core.threshold_cycle([2.0, -0.01], 1.4), 60)

# 4. Upload cleaning & column alias detection (prefers discharge_cycle_num over cycle_index)
clean, notes = core.clean_upload(pd.DataFrame({"c": [3, 1, "x", 2, 2, 1], "q": [1.8, 2.0, 1.9, None, 1.85, 1.7]}), "c", "q")
assert list(clean.discharge_cycle) == [1, 2, 3] and list(clean.capacity_Ah) == [2.0, 1.85, 1.8] and len(notes) == 2

alias_raw = pd.DataFrame({
    "battery_id": ["B99", "B99", "B99"],
    "cycle_index": [10, 20, 30],
    "discharge_cycle_num": [1, 2, 3],
    "capacity_ahr": [1.9, 1.8, 1.7],
    "datetime": ["2008-05-01 00:00:00", "2008-05-01 05:00:00", "2008-05-01 10:00:00"],
})
det = rul_analysis.detect_columns(alias_raw.columns)
assert det["discharge_cycle"] == "discharge_cycle_num" and det["capacity_Ah"] == "capacity_ahr" and det["start_time"] == "datetime"
norm_df, norm_info = rul_analysis.normalize_dataframe(alias_raw)
assert list(norm_df["discharge_cycle"]) == [1, 2, 3] and norm_info["has_timestamps"]

# 5. Calendar-time conversion: 730.5 cycles at 2/day = 365.25 d = exactly 1 Julian year
t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T = core.cycles_to_time(730.5, 2.0, t0)
assert (T["years"], T["days"], T["hours"], T["minutes"], T["seconds_part"]) == (1, 0, 0, 0, 0) and (T["date"] - t0).total_seconds() == 31557600
T = core.cycles_to_time(1.5, 1.0, t0)  # 1.5 days
assert (T["years"], T["days"], T["hours"]) == (0, 1, 12)
assert core.cycles_to_time(10, 0, t0) is None and core.cycles_to_time(-1, 1, t0) is None and core.cycles_to_time(float("nan"), 1, t0) is None

# 6. Data validation: every NASA battery passes; corrupted data is caught
for bat in core.NASA_BATTERIES:
    bad = [c for c, ok, _ in core.validate_dataset(df_all[df_all.battery == bat], nasa=True) if not ok]
    assert not bad, (bat, bad)
d = df_all[df_all.battery == "B0005"].copy()
d.loc[d.index[10], "capacity_Ah"] = 5.0          # spike
assert any(not ok for _, ok, _ in core.validate_dataset(d, nasa=True))
d = df_all[df_all.battery == "B0005"].copy(); d.loc[d.index[3], "capacity_Ah"] = -1.0
assert any(not ok for _, ok, _ in core.validate_dataset(d, nasa=True))
l1 = "1 25544U 98067A   26301.46699108  .00005372  00000+0  10526-3 0  9998"
assert core.tle_checksum_ok(l1) and not core.tle_checksum_ok(l1[:20] + "9" + l1[21:])

# 7. The CSV the app reads reproduces exactly from the raw NASA archive (skipped if the archive is absent)
if core.NASA_ZIP.exists():
    rebuilt = core.build_nasa_csv(out_csv=Path(tempfile.mkdtemp()) / "x.csv")
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), df_all.reset_index(drop=True), check_dtype=False)
    assert rebuilt.groupby("battery").size().to_dict() == {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}

# 8. B0005 Section F sanity check & threshold/RUL arithmetic + timestamp cadence
sanity = rul_analysis.verify_b0005_sanity_check(df_all)
assert sanity["all_passed"], sanity["checks"]

# 9. Safeguards: non-negative slope, already passed in training, crossing before cutoff (B0006), not observed in test (B0007), missing timestamps
_, analyses = rul_analysis.summarize_all_batteries(df_all, core.NASA_BATTERIES, train_frac=0.60, threshold_ah=1.4)
# B0006: predicted crossing ~98.878 <= cutoff 100 -> negative RUL preserved (not clipped to 0), has_positive_predicted_rul is False
r6 = analyses["B0006"]
assert np.isclose(r6["pred_crossing_cycle"], 98.877674, atol=1e-4)
assert np.isclose(r6["pred_rul_cycles"], -1.122326, atol=1e-4) and not r6["has_positive_predicted_rul"]
assert r6["pred_status_code"] == "AT_OR_BEFORE_CUTOFF" and r6["obs_crossing_cycle"] == 109
# B0007: threshold not reached in test horizon (min test cap ~ 1.400455 Ah)
r7 = analyses["B0007"]
assert r7["has_positive_predicted_rul"] and not r7["observed_in_test"] and r7["obs_crossing_cycle"] is None
assert "not observed within the test horizon" in r7["crossing_validation_status"]
# Non-negative slope safeguard
upward_df = pd.DataFrame({"battery": "UP", "discharge_cycle": np.arange(1, 11), "capacity_Ah": np.linspace(1.5, 1.9, 10)})
r_up = rul_analysis.analyze_battery_rul(upward_df, train_frac=0.6, threshold_ah=1.4)
assert r_up["pred_status_code"] == "NON_NEGATIVE_SLOPE" and r_up["pred_crossing_cycle"] is None and not r_up["has_positive_predicted_rul"]
assert not r_up["cadence"]["available"] and r_up["pred_rul_hours_median"] is None
# Already passed in training safeguard
passed_df = df_all[df_all["battery"] == "B0005"].copy()
r_pass = rul_analysis.analyze_battery_rul(passed_df, train_frac=0.6, threshold_ah=1.80)
assert r_pass["already_reached_in_train"] and r_pass["pred_status_code"] == "ALREADY_PASSED_IN_TRAINING" and not r_pass["has_positive_predicted_rul"]

# 10. Exported reproducible outputs into outputs/ (without replacing source CSV)
out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(exist_ok=True)
summary_table, _ = rul_analysis.summarize_all_batteries(df_all, core.NASA_BATTERIES, train_frac=0.60, threshold_ah=1.4)
summary_table.to_csv(out_dir / "nasa_all_batteries_rul_summary_60_40.csv", index=False)
audit_table = pd.DataFrame([rul_analysis.audit_battery_dataframe(df_all, b) for b in core.NASA_BATTERIES])
audit_table.to_csv(out_dir / "nasa_data_quality_audit.csv", index=False)
b5_res = analyses["B0005"]
b5_full = df_all[df_all["battery"] == "B0005"].sort_values("discharge_cycle").reset_index(drop=True).copy()
b5_full["split"] = np.where(b5_full["discharge_cycle"] <= b5_res["cutoff_cycle"], "train", "test")
b5_full["predicted_capacity_Ah"] = core.predict(b5_res["beta_ge"], b5_full["discharge_cycle"])
b5_full["residual_Ah"] = b5_full["capacity_Ah"] - b5_full["predicted_capacity_Ah"]
b5_full.to_csv(out_dir / "B0005_predictions_and_residuals_60_40.csv", index=False)

# 11. All 34 unique NASA batteries deduplication (4 identical MD5 files skipped) + satellite fleet simulation
df34, dedup_log = rul_analysis.build_or_load_all_34_nasa()
assert df34["battery"].nunique() == 34 and len(df34) == 2769
skipped_dups = dedup_log[dedup_log["identical_md5"] == True]
assert len(skipped_dups) == 4 and set(skipped_dups["battery"]) == {"B0025", "B0026", "B0027", "B0028"}
summary_34, _ = rul_analysis.summarize_all_batteries(df34, sorted(df34["battery"].unique()), train_frac=0.60, threshold_ah=1.4)
assert len(summary_34) == 33  # B0052 has 4 rows (<5 minimum for 60/40 split) and is safely skipped
summary_34.to_csv(out_dir / "nasa_all_34_batteries_rul_summary.csv", index=False)

tle_lines = [l.rstrip() for p in sorted(core.DATA_DIR.glob("tle_*.txt")) for l in p.read_text().splitlines() if l.strip()]
sats = [tle_lines[i:i + 3] for i in range(0, len(tle_lines) - 2, 3)
        if tle_lines[i + 1].startswith("1 ") and tle_lines[i + 2].startswith("2 ")]
fleet_df, sat_dedup_df = rul_analysis.analyze_satellite_fleet(
    sats + sats[:3], b5_res["beta0"], b5_res["beta1"], threshold_ah=1.4
)
assert len(fleet_df) > 100 and fleet_df["norad_id"].is_unique and len(sat_dedup_df) >= 3
assert {"orbit_class", "altitude_km", "eclipse_cycles_per_day", "sim_days_to_threshold_fresh"}.issubset(fleet_df.columns)
fleet_df.to_csv(out_dir / "satellite_fleet_simulation_B0005.csv", index=False)

print("All checks passed.")
