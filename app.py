"""Matrix-Based Lithium-Ion Battery Degradation Forecasting and Remaining Useful Life Estimation Using NASA Experimental Data.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

import core
import rul_analysis

st.set_page_config(
    page_title="Matrix-Based Lithium-Ion Battery Degradation Forecasting & RUL Estimation",
    layout="wide",
)

# Muted, print-style palette (no neon, no gradients).
C = {"train": "#2F4B63", "test": "#B07A1E", "pred": "#B5543A", "ok": "#4F7A5A", "grid": "#EAEAEA", "ref": "#787774"}
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&family=Newsreader:wght@500;600&display=swap');
:root { --bg:#FBFBFA; --surface:#FFFFFF; --alt:#F7F6F3; --line:#EAEAEA; --text:#2F3437; --mute:#787774; }
html, body, [class*="css"], .stMarkdown, .stText, button, input { font-family:'Geist','Helvetica Neue',sans-serif !important; color:var(--text); }
.stApp { background:var(--bg); }
[data-testid="stSidebar"] { background:var(--alt); border-right:1px solid var(--line); }
.block-container { padding-top:2.2rem; max-width:1400px; }
code, .mono { font-family:'Geist Mono','SF Mono',monospace !important; }
.hero { padding:6px 0 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
.hero h1 { margin:0; font-family:'Newsreader',Georgia,serif !important; font-weight:600; font-size:32px; letter-spacing:-.02em; line-height:1.15; color:#111; }
.hero p { margin:10px 0 0; color:var(--mute); font-size:14px; line-height:1.6; max-width:940px; }
.status { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 0; }
.chip { font-family:'Geist Mono',monospace; font-size:11.5px; padding:4px 10px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--text); }
.chip b { color:var(--mute); font-weight:500; margin-right:6px; }
.tag { display:inline-block; padding:2px 10px; border-radius:9999px; font-size:10.5px; letter-spacing:.06em; text-transform:uppercase; margin-right:6px; }
.t-meas { background:#E1F3FE; color:#1F6C9F; } .t-pred { background:#FDEBEC; color:#9F2F2D; }
.t-hyp { background:#FBF3DB; color:#956400; } .t-real { background:#EDF3EC; color:#346538; }
.card { border:1px solid var(--line); border-radius:10px; padding:16px 18px; background:var(--surface); height:100%; }
.card .k { color:var(--mute); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; }
.card .v { font-family:'Geist Mono',monospace; font-size:21px; font-weight:500; margin-top:6px; color:#111; }
.card .s { color:var(--mute); font-size:11.5px; margin-top:4px; line-height:1.4; }
div[data-testid="stTabs"] button p { font-size:14px; font-weight:500; }
h4 { font-family:'Newsreader',Georgia,serif !important; font-weight:600; letter-spacing:-.01em; }
/* motion: transform + opacity only, staggered, disabled for reduced-motion users */
@keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
@keyframes fade { from { opacity:0; } to { opacity:1; } }
@keyframes sweep { from { transform:scaleX(0); } to { transform:scaleX(1); } }
.hero { animation:rise .7s cubic-bezier(.16,1,.3,1) both; position:relative; }
.hero::after { content:""; position:absolute; left:0; bottom:-1px; height:2px; width:120px; background:#2F4B63; transform-origin:left; animation:sweep 1.1s .3s cubic-bezier(.16,1,.3,1) both; }
.chip { animation:rise .6s cubic-bezier(.16,1,.3,1) both; }
.chip:nth-child(2) { animation-delay:.05s } .chip:nth-child(3) { animation-delay:.1s } .chip:nth-child(4) { animation-delay:.15s } .chip:nth-child(5) { animation-delay:.2s }
.card { animation:rise .6s cubic-bezier(.16,1,.3,1) both; transition:transform .2s, box-shadow .2s, border-color .2s; }
.card:hover { transform:translateY(-2px); box-shadow:0 2px 10px rgba(0,0,0,.05); border-color:#D5D5D1; }
[data-testid="stColumn"]:nth-child(2) .card { animation-delay:.06s } [data-testid="stColumn"]:nth-child(3) .card { animation-delay:.12s }
[data-testid="stColumn"]:nth-child(4) .card { animation-delay:.18s } [data-testid="stColumn"]:nth-child(5) .card { animation-delay:.24s }
[data-testid="stColumn"]:nth-child(6) .card { animation-delay:.3s }
[data-testid="stTabsContent"], [role="tabpanel"] { animation:fade .45s ease both; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { animation:rise .6s .05s cubic-bezier(.16,1,.3,1) both; }
button[kind], [data-testid="stDownloadButton"] button { transition:transform .15s; } button:active { transform:scale(.98); }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)


def card(col, k, v, s=""):
    col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def fmt(v, d=4, suf=""):
    return "undefined" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}{suf}"


def style(fig, title, ytitle="Capacity (Ah)", xtitle="Sequential discharge-cycle number k"):
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=15)), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FFFFFF", font=dict(family="Geist, Helvetica Neue, sans-serif", size=12, color="#2F3437"),
                      xaxis_title=xtitle, yaxis_title=ytitle, legend=dict(orientation="h", y=-0.22),
                      margin=dict(l=10, r=10, t=50, b=10), height=430)
    fig.update_xaxes(gridcolor=C["grid"], linecolor="#CFCFCB", zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], linecolor="#CFCFCB", zeroline=False)
    return fig


# --------------------------------------------------------------------------- data
@st.cache_data
def nasa():
    return core.load_nasa()


@st.cache_data
def nasa_all_34():
    return rul_analysis.build_or_load_all_34_nasa()


CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP={}&FORMAT=tle"
TLE_GROUPS = {"Space stations": "stations", "Brightest (visual)": "visual", "Weather": "weather",
              "Science": "science", "Earth resources": "resource", "GPS": "gps-ops", "Geostationary": "geo"}


def _fetch(group: str):
    """One CelesTrak group. Returns (text, fetched_utc, source, round_trip_ms). Falls back to the last cached file."""
    cache = core.DATA_DIR / f"tle_{group}.txt"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(CELESTRAK.format(group), headers={"User-Agent": "battery-forecast-research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            txt = r.read().decode()
        ms = (time.perf_counter() - t0) * 1000
        if "\n1 " not in txt:
            raise ValueError(txt[:80])
        cache.write_text(txt)
        return txt, datetime.now(timezone.utc), "live", ms
    except Exception:
        if not cache.exists():
            return "", None, "unavailable", None
        return cache.read_text(), datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc), "cached", None


@st.cache_data(ttl=7200, show_spinner="Fetching orbital elements from CelesTrak...")  # CelesTrak: max one download per group per 2 h
def load_tles(group_keys: tuple):
    with ThreadPoolExecutor(max_workers=6) as ex:  # groups downloaded in parallel
        results = list(ex.map(_fetch, group_keys))
    rows, seen, rejected = [], set(), 0
    for txt, _, _, _ in results:
        lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
        for i in range(0, len(lines) - 2, 3):
            l1, l2 = lines[i + 1], lines[i + 2]
            if not (l1.startswith("1 ") and l2.startswith("2 ")):
                continue
            if not (core.tle_checksum_ok(l1) and core.tle_checksum_ok(l2)) or l1[2:7] != l2[2:7]:
                rejected += 1  # corrupt element set: dropped, never repaired
            elif l1[2:7] not in seen:
                seen.add(l1[2:7]); rows.append(lines[i:i + 3])
    times = [t for _, t, _, _ in results if t]
    srcs = {s for _, _, s, _ in results}
    ms = [m for _, _, _, m in results if m is not None]
    return {"rows": rows, "rejected": rejected, "fetched": min(times).isoformat() if times else None,
            "source": "live" if srcs == {"live"} else ("unavailable" if srcs == {"unavailable"} else ("cached" if srcs == {"cached"} else "mixed")),
            "ms": max(ms) if ms else None}


try:
    ALL = nasa()
    ALL_34, DEDUP_34 = nasa_all_34()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Controls")
    src = st.radio(
        "Data source",
        ["NASA PCoE (Primary 4: B0005–B0018)", "NASA PCoE (All 34 Unique Cells)", "Upload CSV"],
        index=0,
    )
    notes: list[str] = []
    provenance_label = "Official NASA Ames PCoE .mat extraction (data/nasa_pcoe_discharge_capacity.csv)"
    upload_info = None
    active_pool = ALL
    active_bats = core.NASA_BATTERIES
    if src == "NASA PCoE (Primary 4: B0005–B0018)":
        bat = st.selectbox("Battery", core.NASA_BATTERIES, help="18650 Li-ion cells, NASA Ames PCoE, ~24 °C ambient")
        df = ALL[ALL.battery == bat].sort_values("discharge_cycle").reset_index(drop=True)
    elif src == "NASA PCoE (All 34 Unique Cells)":
        batch_opts = ["All 6 NASA sub-archives (34 unique cells, 4 identical duplicates skipped)"] + sorted(ALL_34["batch_description"].unique())
        batch_sel = st.selectbox("Experimental archive batch", batch_opts)
        sub_34 = ALL_34 if batch_sel.startswith("All ") else ALL_34[ALL_34["batch_description"] == batch_sel]
        avail_34 = sorted(sub_34["battery"].unique())
        bat = st.selectbox("Battery (34 unique across NASA PCoE)", avail_34)
        df = ALL_34[ALL_34["battery"] == bat].sort_values("discharge_cycle").reset_index(drop=True)
        active_pool = sub_34
        active_bats = avail_34
        provenance_label = f"Official NASA Ames PCoE Full Archive ({bat}, MD5-deduplicated across 6 sub-zips)"
    else:
        up = st.file_uploader("CSV with cycle, capacity (Ah), and optional timestamp/battery columns", type="csv")
        if up is None:
            st.info("Upload a CSV to continue."); st.stop()
        raw = pd.read_csv(up)
        det = rul_analysis.detect_columns(raw.columns)
        cols_list = list(raw.columns)
        c_idx = cols_list.index(det["discharge_cycle"]) if det["discharge_cycle"] in cols_list else 0
        q_idx = cols_list.index(det["capacity_Ah"]) if det["capacity_Ah"] in cols_list else min(1, len(cols_list) - 1)
        t_opts = ["<none>"] + cols_list
        t_idx = t_opts.index(det["start_time"]) if det["start_time"] in cols_list else 0
        b_opts = ["<single battery: UPLOAD>"] + cols_list
        b_idx = b_opts.index(det["battery"]) if det["battery"] in cols_list else 0

        cyc_col = st.selectbox("Discharge cycle column (prefers discharge_cycle_num)", cols_list, index=c_idx)
        cap_col = st.selectbox("Capacity column (Ah)", cols_list, index=q_idx)
        time_sel = st.selectbox("Timestamp column (optional, as recorded)", t_opts, index=t_idx)
        bat_sel = st.selectbox("Battery ID column (optional)", b_opts, index=b_idx)

        norm_df, upload_info = rul_analysis.normalize_dataframe(
            raw,
            cycle_col=cyc_col,
            cap_col=cap_col,
            time_col=None if time_sel == "<none>" else time_sel,
            bat_col=None if bat_sel.startswith("<") else bat_sel,
        )
        notes = upload_info["notes"]
        avail_bats = list(norm_df["battery"].unique())
        bat = st.selectbox("Select battery from uploaded CSV", avail_bats)
        df = norm_df[norm_df["battery"] == bat].sort_values("discharge_cycle").reset_index(drop=True)
        active_pool = norm_df
        active_bats = avail_bats
        provenance_label = f"User-uploaded CSV ({up.name}) — third-party / external processed file (not labelled as original NASA .mat)"

    frac = st.slider("Training fraction (chronological)", 0.20, 0.95, 0.60, 0.05)
    st.markdown("---")
    show_thr = st.toggle("End-of-life threshold & RUL analysis", value=True)
    thr = st.number_input(
        "Experimental EOL threshold (Ah)", 0.1, 5.0, core.EOL_AH, 0.05,
        help="Default 1.4 Ah = criterion used in the NASA PCoE experiment (30 % fade from 2.0 Ah rated capacity), not a universal limit for all batteries.",
    )
    groups = st.multiselect("Live satellite groups (CelesTrak, for Orbital View)", list(TLE_GROUPS),
                            default=["Space stations", "Science", "Weather", "GPS"])

for n in notes:
    st.sidebar.warning(n)

train, test = core.chronological_split(df, frac)
if len(train) < 3 or train["discharge_cycle"].nunique() < 2:
    st.error(f"Insufficient training data: {len(train)} rows (need ≥3 with ≥2 distinct cycles). Increase the training fraction.")
    st.stop()

# Run full RUL, cadence, and crossing validation analysis for the selected battery and across all batteries
RUL = rul_analysis.analyze_battery_rul(df, train_frac=frac, threshold_ah=thr)
F = RUL["fit_details"]
beta = RUL["beta_ge"]
pred_test = RUL["pred_test"]
M = RUL["metrics"]
k_star = RUL["pred_crossing_cycle"]

ALL_SUMMARY_DF, ALL_ANALYSES = rul_analysis.summarize_all_batteries(
    active_pool,
    active_bats,
    train_frac=frac,
    threshold_ah=thr,
)

# --------------------------------------------------------------------------- header
ORB = load_tles(tuple(TLE_GROUPS[g] for g in groups))
if ORB["fetched"]:
    age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(ORB["fetched"])).total_seconds() / 60
    age_txt = f"{age_min:.0f} min ago" if age_min < 120 else f"{age_min / 60:.1f} h ago"
else:
    age_txt = "n/a"
src_txt = {"live": "live download", "cached": "cached file", "mixed": "live + cached", "unavailable": "offline"}[ORB["source"]]
ms_txt = f"{ORB['ms']:.0f} ms" if ORB["ms"] is not None else "n/a"
st.markdown(f"""<div class="hero"><h1>Matrix-Based Lithium-Ion Battery Degradation Forecasting and Remaining Useful Life Estimation Using NASA Experimental Data</h1>
<p>Linear regression on chronological laboratory discharge-capacity fade (<span class="mono">Q̂(k) = β₀ + β₁k</span>), normal equations solved transparently via
Gaussian elimination (REF) and Gauss–Jordan elimination (RREF), cycle-based and recorded-timestamp cadence RUL estimation, and held-out threshold-crossing validation.</p>
<div class="status">
<span class="chip"><b>DATA SOURCE</b>{provenance_label}</span>
<span class="chip"><b>SELECTED CELL</b>{bat} ({len(df)} discharge cycles)</span>
<span class="chip"><b>SPLIT</b>{len(train)} train / {len(test)} test ({int(round(frac*100))}% / {int(round((1-frac)*100))}%)</span>
<span class="chip"><b>EOL THRESHOLD</b>{thr:.2f} Ah (experimental criterion)</span>
<span class="chip"><b>ORBIT CONTEXT</b>CelesTrak {src_txt} ({len(ORB['rows'])} objects, {ms_txt})</span></div>
<p style="margin-top:14px"><span class="tag t-meas">Measured (NASA lab)</span><span class="tag t-pred">Model prediction / extrapolation</span>
<span class="tag t-real">Recorded timestamp cadence</span><span class="tag t-hyp">Hypothetical orbital scenario</span></p></div>""",
            unsafe_allow_html=True)

cols = st.columns(6)
card(cols[0], "Battery & Range", bat, f"Train k={RUL['train_cycle_range'][0]}–{RUL['train_cycle_range'][1]} | Test k={RUL['test_cycle_range'][0]}–{RUL['test_cycle_range'][1]}")
card(cols[1], "Fitted Slope β₁", f"{beta[1]:.6f}", f"Ah/cycle · β₀ = {beta[0]:.4f} Ah")
card(cols[2], "Held-out MAE / RMSE", f"{fmt(M['MAE'], 4)} / {fmt(M['RMSE'], 4)}", f"Ah ({fmt(M['RMSE_%_of_mean'], 2, '%')} of test mean)")
card(cols[3], "Held-out R² / MAPE", f"{fmt(M['R2'], 3)} / {fmt(M['MAPE_%'], 2, '%')}", "test set exclusively")
if RUL["has_positive_predicted_rul"]:
    card(cols[4], "Predicted RUL (Cycles)", f"{RUL['pred_rul_cycles']:.2f} cyc", f"crosses {thr:.2f} Ah at k ≈ {RUL['pred_crossing_cycle']:.2f}")
else:
    raw_s = f"raw Δk = {RUL['pred_rul_cycles']:.2f} (k̂={RUL['pred_crossing_cycle']:.2f})" if RUL["pred_crossing_cycle"] is not None else "β₁ ≥ 0"
    card(cols[4], "Predicted RUL (Cycles)", "No +RUL", raw_s)
if RUL["observed_in_test"]:
    err_s = f"cycle error (pred−obs): {RUL['cycle_error_pred_minus_actual']:+.2f}" if RUL["cycle_error_pred_minus_actual"] is not None else "no model crossing"
    card(cols[5], "Observed Test Crossing", f"Cycle {RUL['obs_crossing_cycle']}", f"actual RUL = {RUL['actual_rul_cycles']} cyc ({err_s})")
elif RUL["already_reached_in_train"]:
    card(cols[5], "Observed Crossing", f"Train k={RUL['first_train_crossing_cycle']}", "already reached in training")
else:
    card(cols[5], "Observed Test Crossing", "Not reached", f"min test cap = {test.capacity_Ah.min():.4f} Ah" if len(test) else "no test set")

for n in M["notes"]:
    st.warning(n)

tabs = st.tabs([
    "Forecast & RUL",
    "RUL & Cadence Validation",
    "Data & Quality Audit",
    "Matrix Solver",
    "Residuals & Checks",
    "Cross-Battery",
    "Orbital View",
    "Methodology & Paper",
    "Results & Export",
])

# --------------------------------------------------------------------------- 1. Forecast & RUL
with tabs[0]:
    kmax = float(df.discharge_cycle.max())
    if show_thr and k_star is not None and k_star > 0:
        kmax = max(kmax, min(float(k_star) * 1.08, kmax * 2.5))
    kk = np.linspace(float(df.discharge_cycle.min()), kmax, 300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.discharge_cycle, y=train.capacity_Ah, mode="markers",
        name=f"Measured — training (k={RUL['train_cycle_range'][0]}–{RUL['train_cycle_range'][1]})",
        marker=dict(color=C["train"], size=7, line=dict(width=.5, color="#001")),
    ))
    fig.add_trace(go.Scatter(
        x=test.discharge_cycle, y=test.capacity_Ah, mode="markers",
        name=f"Measured — held-out test (k={RUL['test_cycle_range'][0]}–{RUL['test_cycle_range'][1]})",
        marker=dict(color=C["test"], size=7, symbol="diamond", line=dict(width=.5, color="#001")),
    ))
    fig.add_trace(go.Scatter(
        x=kk, y=core.predict(beta, kk), mode="lines",
        name=f"Regression fit Q̂(k) = {beta[0]:.4f} {'−' if beta[1]<0 else '+'} {abs(beta[1]):.6f}k",
        line=dict(color=C["pred"], width=2.8, dash="dash"),
    ))
    fig.add_vline(
        x=RUL["cutoff_cycle"] + 0.5, line=dict(color="#787774", dash="dot"),
        annotation_text=f"training cutoff (k={RUL['cutoff_cycle']})", annotation_position="top",
    )
    if show_thr:
        fig.add_hline(
            y=thr, line=dict(color=C["ok"], dash="dot"),
            annotation_text=f"experimental EOL threshold {thr:.2f} Ah (30% fade of 2.0 Ah rated)",
            annotation_position="bottom left",
        )
        if k_star is not None and 0 < k_star <= kmax * 1.2:
            fig.add_trace(go.Scatter(
                x=[k_star], y=[thr], mode="markers",
                name=f"Predicted crossing (k̂ ≈ {k_star:.2f} — model estimate)",
                marker=dict(color=C["pred"], size=13, symbol="x", line=dict(width=2, color="#111")),
            ))
        if RUL["observed_in_test"]:
            fig.add_trace(go.Scatter(
                x=[RUL["obs_crossing_cycle"]], y=[RUL["obs_crossing_cap_Ah"]], mode="markers",
                name=f"First observed test crossing (k = {RUL['obs_crossing_cycle']}, {RUL['obs_crossing_cap_Ah']:.4f} Ah)",
                marker=dict(color=C["ok"], size=13, symbol="star", line=dict(width=1.5, color="#111")),
            ))
    st.plotly_chart(style(fig, f"{bat}: Measured Discharge Capacity vs. Sequential Discharge Cycle & Linear Regression Forecast"))

    c1, c2 = st.columns([1.15, 1.35])
    with c1:
        st.markdown("#### Fitted Linear Regression Model (Training Data Only)")
        st.latex(rf"\hat Q(k) = {beta[0]:.9f} \;{'-' if beta[1] < 0 else '+'}\; {abs(beta[1]):.9f}\,k \quad\text{{(Ah)}}")
        st.markdown(f"""
- **Intercept ($\\beta_0$):** `{beta[0]:.9f} Ah` (estimated initial capacity parameter at $k=0$)
- **Regression slope ($\\beta_1$):** `{beta[1]:.9f} Ah / discharge cycle` (`{beta[1]*1000:.4f} mAh/cycle`)
- **Training range:** cycles `{RUL['train_cycle_range'][0]}` to `{RUL['train_cycle_range'][1]}` (`{RUL['n_train']}` observations; cutoff timestamp `{RUL['cutoff_timestamp_str'] or 'n/a'}`; cutoff capacity `{RUL['cutoff_capacity_Ah']:.6f} Ah`)
- **Held-out test range:** cycles `{RUL['test_cycle_range'][0]}` to `{RUL['test_cycle_range'][1]}` (`{RUL['n_test']}` observations)
- **Solver verification:** $|\\beta_{{\\text{{GE}}}} - \\beta_{{\\text{{GJ}}}}|_\\infty = {RUL['ge_gj_max_diff']:.3e}$
""")
    with c2:
        st.markdown("#### End-of-Life Threshold & Remaining Useful Life Summary")
        st.markdown(f"**Status:** `{RUL['crossing_validation_status']}`")
        for note in RUL["safeguard_notes"]:
            st.warning(note)
        cad = RUL["cadence"]
        if RUL["has_positive_predicted_rul"]:
            st.markdown(f"""
- **Experimental threshold:** `{thr:.3f} Ah` (criterion used in this NASA experiment; not a universal battery limit)
- **Predicted crossing cycle (extrapolated model estimate):** `k̂ = {RUL['pred_crossing_cycle']:.3f}`
- **Predicted RUL from cutoff (`k = {RUL['cutoff_cycle']}`):** `{RUL['pred_rul_cycles']:.3f} discharge cycles`
""")
            if cad["available"]:
                st.markdown(f"""
- **Observed training cadence (`n = {cad['n_positive_intervals']}` positive intervals):** median `{cad['median_hours']:.3f} h` | mean `{cad['mean_hours']:.3f} h` | IQR `{cad['iqr_hours']:.3f} h` (range `{cad['min_hours']:.2f}–{cad['max_hours']:.2f} h`)
- **Primary time-based RUL (median cadence):** `{RUL['pred_rul_hours_median']:.2f} hours` (`{RUL['pred_rul_days_median']:.3f} days`, or `{rul_analysis.format_duration_hours(RUL['pred_rul_hours_median'])}`) → predicted timestamp `{RUL['pred_crossing_ts_median']}`
- **Sensitivity time-based RUL (mean cadence):** `{RUL['pred_rul_hours_mean']:.2f} hours` (`{RUL['pred_rul_days_mean']:.3f} days`) → predicted timestamp `{RUL['pred_crossing_ts_mean']}`
""")
                if cad["pause_distortion_warning"]:
                    st.caption(cad["pause_distortion_warning"])
            else:
                st.info(cad["reason"])
        else:
            st.info(
                f"No positive predicted RUL is reported from cutoff k={RUL['cutoff_cycle']} "
                f"({RUL['pred_status_label']}). Unclipped arithmetic crossing: "
                + (f"k̂ = {RUL['pred_crossing_cycle']:.3f}, raw Δk = {RUL['pred_rul_cycles']:.3f} cycles." if RUL["pred_crossing_cycle"] is not None else "n/a (non-negative slope).")
            )

        if RUL["observed_in_test"]:
            act_t_str = (
                f", elapsed time from cutoff `{RUL['actual_elapsed_hours']:.3f} h` (`{RUL['actual_elapsed_days']:.3f} d`, `{rul_analysis.format_duration_hours(RUL['actual_elapsed_hours'])}`)"
                if RUL["actual_elapsed_hours"] is not None else ""
            )
            err_t_str = (
                f" | median-cadence time error (pred − actual): `{RUL['time_error_hours_median']:+.3f} h` (`{RUL['time_error_days_median']:+.3f} d`, `{rul_analysis.format_duration_hours(RUL['time_error_hours_median'])}`)"
                if RUL["time_error_hours_median"] is not None else ""
            )
            st.markdown(f"""
- **First observed held-out crossing (measured event):** cycle `{RUL['obs_crossing_cycle']}` (`{RUL['obs_crossing_cap_Ah']:.6f} Ah`) at `{RUL['obs_crossing_ts_str'] or 'n/a'}` → actual RUL = `{RUL['actual_rul_cycles']} cycles`{act_t_str}
- **Threshold-crossing validation error (predicted − observed):** `{RUL['cycle_error_pred_minus_actual']:+.3f} cycles`{err_t_str}
""")
        st.caption(
            "Note: Time-based RUL reflects the laboratory test schedule recorded in the dataset timestamps and must never be "
            "interpreted as a spacecraft battery-life guarantee."
        )

# --------------------------------------------------------------------------- 2. RUL & Cadence Validation (All Batteries)
with tabs[1]:
    st.markdown("#### Consistent Multi-Battery Regression, RUL, Cadence & Validation Summary")
    st.caption(
        f"All batteries below are evaluated with the exact same chronological split (`{int(round(frac*100))}%` training / "
        f"`{int(round((1-frac)*100))}%` held-out testing) and experimental threshold (`{thr:.2f} Ah`). "
        "No values are copied from prior runs or hardcoded."
    )
    st.dataframe(ALL_SUMMARY_DF.style.format(precision=5, na_rep="—"), width="stretch", hide_index=True)

    st.markdown("#### Timestamp Interval Cadence & Pause Sensitivity (`" + bat + "`)")
    cad = RUL["cadence"]
    if cad["available"]:
        ts_tr = pd.to_datetime(train["start_time"], errors="coerce")
        if getattr(ts_tr.dt, "tz", None) is not None:
            ts_tr = ts_tr.dt.tz_localize(None)
        dt_tr = ts_tr.diff().dt.total_seconds() / 3600.0
        fig_cad = go.Figure()
        fig_cad.add_trace(go.Bar(
            x=train["discharge_cycle"].iloc[1:], y=dt_tr.iloc[1:],
            name="Training inter-discharge interval (hours)", marker_color=C["train"],
        ))
        fig_cad.add_hline(y=cad["median_hours"], line=dict(color=C["ok"], width=2),
                          annotation_text=f"Median = {cad['median_hours']:.2f} h (primary)")
        fig_cad.add_hline(y=cad["mean_hours"], line=dict(color=C["pred"], dash="dash", width=2),
                          annotation_text=f"Mean = {cad['mean_hours']:.2f} h (distorted by rest pauses)")
        st.plotly_chart(style(
            fig_cad,
            f"{bat}: Recorded Training Inter-Discharge Time Intervals (Hours) — Why Median Cadence is Primary",
            ytitle="Interval between consecutive discharges (hours)",
            xtitle="Training discharge-cycle number k",
        ))
    else:
        st.info(cad["reason"])

    if src.startswith("NASA PCoE"):
        st.markdown("#### Reproducibility Check: B0005 Reference Verification (60% Split, 1.4 Ah Threshold)")
        sanity = rul_analysis.verify_b0005_sanity_check(ALL)
        r_mat = sanity["full_precision_mat"]
        r_min = sanity["minute_truncated_equiv"]
        st.success("All B0005 reference checks passed from live recalculation." if sanity["all_passed"] else "Discrepancy detected in B0005 sanity check.")
        ref_table = pd.DataFrame([
            {"Quantity": "Training cutoff cycle", "Reference (Section F)": "100", "Direct NASA .mat (seconds)": f"{r_mat['cutoff_cycle']}", "Minute-truncated timestamp equiv.": f"{r_min['cutoff_cycle']}"},
            {"Quantity": "Cycle 100 timestamp", "Reference (Section F)": "2008-05-11 16:49:00", "Direct NASA .mat (seconds)": str(r_mat["cutoff_timestamp"]), "Minute-truncated timestamp equiv.": str(r_min["cutoff_timestamp"])},
            {"Quantity": "Capacity at cycle 100 (Ah)", "Reference (Section F)": "≈ 1.485868", "Direct NASA .mat (seconds)": f"{r_mat['cutoff_capacity_Ah']:.6f}", "Minute-truncated timestamp equiv.": f"{r_min['cutoff_capacity_Ah']:.6f}"},
            {"Quantity": "Fitted intercept β₀ (Ah)", "Reference (Section F)": "≈ 1.901404883", "Direct NASA .mat (seconds)": f"{r_mat['beta0']:.9f}", "Minute-truncated timestamp equiv.": f"{r_min['beta0']:.9f}"},
            {"Quantity": "Fitted slope β₁ (Ah/cycle)", "Reference (Section F)": "≈ -0.003843534", "Direct NASA .mat (seconds)": f"{r_mat['beta1']:.9f}", "Minute-truncated timestamp equiv.": f"{r_min['beta1']:.9f}"},
            {"Quantity": "Predicted crossing at 1.4 Ah (cycle)", "Reference (Section F)": "≈ 130.454", "Direct NASA .mat (seconds)": f"{r_mat['pred_crossing_cycle']:.6f}", "Minute-truncated timestamp equiv.": f"{r_min['pred_crossing_cycle']:.6f}"},
            {"Quantity": "Predicted RUL (cycles)", "Reference (Section F)": "≈ 30.454", "Direct NASA .mat (seconds)": f"{r_mat['pred_rul_cycles']:.6f}", "Minute-truncated timestamp equiv.": f"{r_min['pred_rul_cycles']:.6f}"},
            {"Quantity": "First observed test crossing cycle & capacity", "Reference (Section F)": "cycle 125, ≈ 1.396701 Ah", "Direct NASA .mat (seconds)": f"cycle {r_mat['obs_crossing_cycle']}, {r_mat['obs_crossing_cap_Ah']:.6f} Ah", "Minute-truncated timestamp equiv.": f"cycle {r_min['obs_crossing_cycle']}, {r_min['obs_crossing_cap_Ah']:.6f} Ah"},
            {"Quantity": "Observed crossing timestamp", "Reference (Section F)": "2008-05-17 17:15:00", "Direct NASA .mat (seconds)": str(r_mat["obs_crossing_ts"]), "Minute-truncated timestamp equiv.": str(r_min["obs_crossing_ts"])},
            {"Quantity": "Actual elapsed time from cutoff to observed crossing", "Reference (Section F)": "≈ 144.433 h (6d 26m)", "Direct NASA .mat (seconds)": f"{r_mat['actual_elapsed_hours']:.4f} h ({rul_analysis.format_duration_hours(r_mat['actual_elapsed_hours'])})", "Minute-truncated timestamp equiv.": f"{r_min['actual_elapsed_hours']:.4f} h ({rul_analysis.format_duration_hours(r_min['actual_elapsed_hours'])})"},
            {"Quantity": "Median positive training interval (hours)", "Reference (Section F)": "≈ 4.95 h", "Direct NASA .mat (seconds)": f"{r_mat['cadence']['median_hours']:.6f} h (4h 57m 11s)", "Minute-truncated timestamp equiv.": f"{r_min['cadence']['median_hours']:.6f} h (4h 57m 00s)"},
            {"Quantity": "Median-cadence time RUL forecast", "Reference (Section F)": "≈ 150.749 h (6d 6h 45m)", "Direct NASA .mat (seconds)": f"{r_mat['pred_rul_hours_median']:.3f} h ({rul_analysis.format_duration_hours(r_mat['pred_rul_hours_median'])})", "Minute-truncated timestamp equiv.": f"{r_min['pred_rul_hours_median']:.3f} h ({rul_analysis.format_duration_hours(r_min['pred_rul_hours_median'])})"},
            {"Quantity": "Time forecast error (predicted − observed)", "Reference (Section F)": "≈ +6h 19m", "Direct NASA .mat (seconds)": f"{r_mat['time_error_hours_median']:+.3f} h ({rul_analysis.format_duration_hours(r_mat['time_error_hours_median'])})", "Minute-truncated timestamp equiv.": f"{r_min['time_error_hours_median']:+.3f} h ({rul_analysis.format_duration_hours(r_min['time_error_hours_median'])})"},
        ])
        st.dataframe(ref_table, width="stretch", hide_index=True)
        st.caption(
            "Explanation of the 11-second timestamp precision difference: the original NASA `.mat` files store `time` vectors "
            "to the second (e.g. `2008-05-11 16:49:58`), giving a median interval of `4.953056 h` (`150.841 h` RUL). "
            "Third-party `clean_discharge.csv` exports truncate seconds to `:00` (`2008-05-11 16:49:00`), giving `4.950000 h` (`150.748 h` RUL)."
        )

# --------------------------------------------------------------------------- 3. Data & Quality Audit
with tabs[2]:
    st.markdown("#### Dataset Provenance & Integrity Audit")
    st.markdown(f"""
- **Active dataset source:** `{provenance_label}`
- **Official NASA dataset reference:** [{rul_analysis.OFFICIAL_NASA_URL}]({rul_analysis.OFFICIAL_NASA_URL})
- **Original repository citation:** {rul_analysis.OFFICIAL_NASA_CITATION}
- **Secondary processed reference (for comparison only):** [{rul_analysis.SECONDARY_GITHUB_URL}]({rul_analysis.SECONDARY_GITHUB_URL})
- **Column alias detection rules:** automatically inspects headers for `battery_id`/`battery`, `discharge_cycle_num`/`discharge_cycle`/`discharge_index`/`cycle_index` (preferring sequential discharge event count `discharge_cycle_num` over general charge+discharge cycle numbers), `datetime`/`timestamp`/`date_time`/`start_time`, and `capacity_ahr`/`capacity_Ah`/`capacity_ah`/`capacity`.
- **Precision & timezone policy:** full double-precision (`float64`) capacities are preserved internally; timestamps are parsed as recorded without inventing or appending `UTC`.
""")
    audit_target = active_pool
    audit_bats = active_bats
    audit_df = pd.DataFrame([rul_analysis.audit_battery_dataframe(audit_target, b) for b in audit_bats])
    st.dataframe(audit_df, width="stretch", hide_index=True)

    st.markdown("#### Deduplication of Identical NASA Archive Files & Multi-Batch Overview")
    st.caption(
        "Across the 6 official `.zip` sub-archives in `5. Battery Data Set.zip` (38 `.mat` files total), "
        "4 files (`B0025.mat`–`B0028.mat`) appear identically in both `2. BatteryAgingARC_25_26_27_28_P1.zip` and "
        "`3. BatteryAgingARC_25-44.zip` (verified by exact MD5 hash). Those 4 identical duplicates are skipped automatically, "
        "leaving all **34 unique NASA batteries** (`2,769` discharge observations) available in the simulation."
    )
    st.dataframe(DEDUP_34, width="stretch", hide_index=True)

    st.markdown("#### Automated Structural & Physical Sanity Checks")
    sets = {b_: audit_target[audit_target.battery == b_].sort_values("discharge_cycle") for b_ in audit_bats}
    rows_v = []
    for name_, d_ in sets.items():
        for chk, ok_, det in core.validate_dataset(d_, nasa=(src == "NASA PCoE (Primary 4: B0005–B0018)")):
            rows_v.append({"dataset": name_, "check": chk, "result": "PASS" if ok_ else "FAIL", "detail": det})
    vdf = pd.DataFrame(rows_v)
    n_fail = int((vdf.result == "FAIL").sum())
    (st.success if n_fail == 0 else st.warning)(
        f"{len(vdf) - n_fail} of {len(vdf)} structural/physical checks passed across {len(sets)} battery series."
    )
    st.dataframe(vdf, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- 4. Matrix Solver
def mat_df(Mx, aug=False):
    cols = [f"β{j}" for j in range(Mx.shape[1] - (1 if aug else 0))] + (["| rhs"] if aug else [])
    return pd.DataFrame(Mx, columns=cols, index=[f"R{i+1}" for i in range(Mx.shape[0])])


with tabs[3]:
    st.markdown("#### 1 · Training data in matrix form (training rows only)")
    a, b, c = st.columns([2, 1, 2])
    with a:
        st.latex(r"X=\begin{bmatrix}1&k_1\\1&k_2\\\vdots&\vdots\\1&k_n\end{bmatrix},\;\; \mathbf y=\begin{bmatrix}C_1\\C_2\\\vdots\\C_n\end{bmatrix}")
        st.dataframe(pd.DataFrame(F["X"], columns=["1 (intercept)", "k (cycle)"]).assign(**{"y = C (Ah)": F["y"]}),
                     height=260, width="stretch")
    with b:
        st.latex(r"X^TX\,\beta = X^T\mathbf y")
        st.markdown(f"n = **{len(train)}** training rows")
        st.markdown(f"cond(XᵀX) = **{np.linalg.cond(F['XtX']):.3e}**")
    with c:
        XtX, Xty = F["XtX"], F["Xty"]
        st.latex(rf"\begin{{bmatrix}}{XtX[0,0]:.0f}&{XtX[0,1]:.0f}\\{XtX[1,0]:.0f}&{XtX[1,1]:.0f}\end{{bmatrix}}"
                 rf"\begin{{bmatrix}}\beta_0\\\beta_1\end{{bmatrix}}=\begin{{bmatrix}}{Xty[0]:.6f}\\{Xty[1]:.6f}\end{{bmatrix}}")
        st.markdown("Augmented matrix **[XᵀX | Xᵀy]**")
        st.dataframe(mat_df(np.column_stack([XtX, Xty]), True).style.format("{:.8g}"), width="stretch")

    st.markdown("#### 2 · Interactive elimination")
    g1, g2 = st.columns(2)
    for col, key, title in [(g1, "ge", "Gaussian elimination → REF → back-substitution"),
                            (g2, "gj", "Gauss–Jordan elimination → RREF")]:
        with col:
            steps = F[f"steps_{key}"]
            st.markdown(f"**{title}**")
            i = st.slider("Step", 0, len(steps) - 1, len(steps) - 1, key=f"s_{key}")
            s = steps[i]
            st.markdown(f"`{s['phase']}` — **{s['op']}**")
            st.dataframe(mat_df(s["matrix"], True).style.format("{:.10g}"), width="stretch")
            with st.expander("All row operations"):
                for j, s in enumerate(steps):
                    st.markdown(f"{j}. `{s['phase']}` {s['op']}")

    st.markdown("#### 3 · Coefficient comparison")
    b_np = np.linalg.lstsq(F["X"], F["y"], rcond=None)[0]
    sk = LinearRegression().fit(train[["discharge_cycle"]], train.capacity_Ah)
    b_sk = np.array([sk.intercept_, sk.coef_[0]])
    comp = pd.DataFrame({"Gaussian (REF)": F["beta_ge"], "Gauss–Jordan (RREF)": F["beta_gj"],
                         "|GE − GJ|": np.abs(F["beta_ge"] - F["beta_gj"]),
                         "numpy lstsq (verify)": b_np, "scikit-learn (verify)": b_sk,
                         "max |GE − verify|": np.maximum(np.abs(F["beta_ge"] - b_np), np.abs(F["beta_ge"] - b_sk))},
                        index=["β₀ intercept (Ah)", "β₁ slope (Ah/cycle)"])
    st.dataframe(comp.style.format("{:.12g}"), width="stretch")
    st.caption("numpy/scikit-learn are used as independent numerical verification; the transparent step-by-step solvers in core.py match to machine precision.")

# --------------------------------------------------------------------------- 5. Residuals & Checks
with tabs[4]:
    res_tr = train.capacity_Ah - core.predict(beta, train.discharge_cycle)
    res_te = test.capacity_Ah - pred_test
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.discharge_cycle, y=res_tr, mode="markers", name="Training residual", marker=dict(color=C["train"])))
    fig.add_trace(go.Scatter(x=test.discharge_cycle, y=res_te, mode="markers", name="Held-out residual", marker=dict(color=C["test"], symbol="diamond")))
    fig.add_hline(y=0, line=dict(color="#787774"))
    a, b = st.columns(2)
    a.plotly_chart(style(fig, "Residuals (measured − predicted) vs cycle", "Residual (Ah)"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pred_test, y=res_te, mode="markers", name="Held-out", marker=dict(color=C["test"])))
    fig2.add_hline(y=0, line=dict(color="#787774"))
    b.plotly_chart(style(fig2, "Held-out residuals vs predicted value", "Residual (Ah)", "Predicted capacity (Ah)"))
    st.caption("Structured (non-random) residual patterns, e.g. the capacity-regeneration jumps after rest periods in NASA "
               "data, indicate the straight-line model misses real electrochemical behaviour.")

    st.markdown("#### Training-window sensitivity experiment")
    rows = []
    for f in np.round(np.arange(0.20, 0.951, 0.05), 2):
        tr, te = core.chronological_split(df, f)
        if len(tr) < 3 or len(te) < 2:
            continue
        r_s = rul_analysis.analyze_battery_rul(df, train_frac=float(f), threshold_ah=thr)
        m = r_s["metrics"]
        rows.append({
            "train_fraction": f, "n_train": len(tr), "n_test": len(te),
            "beta0": r_s["beta0"], "beta1": r_s["beta1"],
            "MAE_Ah": m["MAE"], "RMSE_Ah": m["RMSE"], "R2": m["R2"], "MAPE_%": m["MAPE_%"], "RMSE_%": m["RMSE_%_of_mean"],
            "pred_crossing_cycle": r_s["pred_crossing_cycle"], "pred_RUL_cycles": r_s["pred_rul_cycles"],
            "obs_crossing_cycle": r_s["obs_crossing_cycle"], "cycle_error": r_s["cycle_error_pred_minus_actual"],
        })
    sens = pd.DataFrame(rows)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sens.train_fraction, y=sens.RMSE_Ah, mode="lines+markers", name="Held-out RMSE (Ah)", line=dict(color=C["pred"])))
    fig3.add_trace(go.Scatter(x=sens.train_fraction, y=sens.MAE_Ah, mode="lines+markers", name="Held-out MAE (Ah)", line=dict(color=C["train"])))
    fig3.add_vline(x=frac, line=dict(color=C["ok"], dash="dot"), annotation_text="current")
    st.plotly_chart(style(fig3, "Held-out error vs training fraction", "Error (Ah)", "Training fraction"))
    st.dataframe(sens.style.format(precision=5, na_rep="—"), width="stretch")

    st.markdown("#### Automated integrity checks")
    checks = []
    tol = 1e-9 * max(1.0, np.abs(F["beta_ge"]).max())
    checks.append(("Gaussian and Gauss–Jordan coefficients agree (|Δ| ≤ 1e-9 rel.)", np.allclose(F["beta_ge"], F["beta_gj"], rtol=1e-9, atol=tol)))
    checks.append(("Elimination matches numpy lstsq and scikit-learn (rtol 1e-6)", np.allclose(F["beta_ge"], b_np, rtol=1e-6, atol=1e-10) and np.allclose(F["beta_ge"], b_sk, rtol=1e-6, atol=1e-10)))
    checks.append(("Chronological split: every training cycle precedes every test cycle", len(test) == 0 or train.discharge_cycle.max() < test.discharge_cycle.min()))
    checks.append(("Train ∪ test = all rows, train ∩ test = ∅", len(train) + len(test) == len(df) and not set(train.index) & set(test.index)))
    checks.append(("Design matrix has exactly the training rows", F["X"].shape[0] == len(train)))
    perturbed = df.copy(); perturbed.loc[test.index, "capacity_Ah"] = 999.0
    checks.append(("No leakage: replacing all test capacities with 999 leaves β unchanged",
                   np.array_equal(core.fit(core.chronological_split(perturbed, frac)[0])["beta_ge"], beta)))
    for name, ok in checks:
        st.markdown(f"{'**PASS**' if ok else '**FAIL**'} — {name}")

# --------------------------------------------------------------------------- 6. Cross-Battery
with tabs[5]:
    st.markdown(f"Model trained on **all {len(df)} cycles of {bat}**, then tested on every other battery in the selected pool without refitting.")
    b_full = core.fit(df)["beta_ge"]
    crit = st.slider("Flag 'poor transfer' when RMSE % of mean test capacity exceeds", 1.0, 20.0, 5.0, 0.5)
    rows, fig = [], go.Figure()
    kx = np.linspace(1, active_pool.discharge_cycle.max(), 200)
    fig.add_trace(go.Scatter(x=kx, y=core.predict(b_full, kx),
                             name=f"Model from {bat}", line=dict(color=C["pred"], width=3, dash="dash")))
    palette = ["#2F4B63", "#B07A1E", "#4F7A5A", "#7A6A9A", "#8C564B", "#5C6B73", "#A47148", "#3D5A80"]
    for i, ob in enumerate([b_ for b_ in active_bats if b_ != bat]):
        o = active_pool[active_pool.battery == ob]
        m = core.metrics(o.capacity_Ah, core.predict(b_full, o.discharge_cycle))
        poor = (m["R2"] is None or m["R2"] < 0) or (m["RMSE_%_of_mean"] or 0) > crit
        rows.append({"test battery": ob, "n": m["n"], "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                     "MAPE_%": m["MAPE_%"], "RMSE_%": m["RMSE_%_of_mean"], "transfer": "POOR" if poor else "acceptable"})
        if i < 12:  # plot up to 12 series clearly on the chart, while listing all in the table below
            fig.add_trace(go.Scatter(x=o.discharge_cycle, y=o.capacity_Ah, mode="markers", name=f"{ob} measured",
                                     marker=dict(color=palette[i % len(palette)], size=5)))
    xb = pd.DataFrame(rows)
    st.plotly_chart(style(fig, "Cross-battery transfer"))
    st.dataframe(xb.style.format(precision=4, na_rep="undefined"), width="stretch")
    if len(xb) and (xb.transfer != "acceptable").any():
        st.warning("Performance transfers poorly to: " + ", ".join(xb[xb.transfer != "acceptable"]["test battery"]) +
                   ". Cells differ in initial capacity, discharge cut-off voltage (B0005 2.7 V, B0006 2.5 V, B0007 2.2 V, "
                   "B0018 2.5 V), ambient temperature (24 °C vs 43 °C vs 4 °C), and fade rate — a single-cell linear fit is not a general model.")

# --------------------------------------------------------------------------- 7. Orbital View (preserved existing feature + fleet table)
with tabs[6]:
    payload = {"tles": ORB["rows"], "beta": [float(beta[0]), float(beta[1])], "rated": core.RATED_AH, "threshold": float(thr),
               "battery": bat, "rmse": M["RMSE"], "max_cycle": int(df.discharge_cycle.max()),
               "fetched": ORB["fetched"], "source": ORB["source"], "fetch_ms": ORB["ms"]}
    html = (Path(__file__).parent / "globe.html").read_text().replace("__PAYLOAD__", json.dumps(payload))
    components.html(html, height=800)
    st.caption(f"Positions: CelesTrak GP data ({src_txt}, fetched {age_txt}) propagated in the browser with SGP4 (satellite.js). "
               "Illumination and eclipse counts are geometric models (low-precision solar ephemeris, cylindrical Earth shadow). "
               "Battery numbers in the satellite card are a **hypothetical scenario** that applies the NASA lab regression above "
               "under the stated depth-of-discharge assumption — they are not satellite telemetry and not validated RUL.")

    st.markdown("#### All Applicable Satellites — Fleet Orbital & Hypothetical Battery Simulation Table")
    st.caption(
        f"Every unique satellite loaded from the selected CelesTrak groups (`{len(ORB['rows'])}` unique NORAD objects; "
        "identical/duplicate NORAD entries across groups are skipped automatically) evaluated with the current "
        f"`{bat}` linear regression (`β₀ = {beta[0]:.4f} Ah`, `β₁ = {beta[1]:.6f} Ah/cycle`, threshold `{thr:.2f} Ah`). "
        "**Hypothetical mission scenario only — not actual spacecraft telemetry.**"
    )
    dod_fleet = st.slider("Assumed depth of discharge (DOD) per eclipse for fleet comparison (%)", 5, 100, 25, 5) / 100.0
    fleet_df, sat_dedup_df = rul_analysis.analyze_satellite_fleet(
        ORB["rows"], float(beta[0]), float(beta[1]), threshold_ah=float(thr), rated_ah=core.RATED_AH, assumed_dod=dod_fleet
    )
    if len(fleet_df):
        f_col1, f_col2 = st.columns(2)
        cls_counts = fleet_df.groupby("orbit_class", as_index=False).agg(
            satellites=("norad_id", "count"),
            mean_altitude_km=("altitude_km", "mean"),
            mean_eclipse_cycles_day=("eclipse_cycles_per_day", "mean"),
            mean_lab_equiv_cycles_day=("lab_equiv_cycles_per_day", "mean"),
            mean_sim_years_to_threshold=("sim_years_to_threshold_fresh", "mean"),
        )
        with f_col1:
            fig_fl1 = go.Figure()
            for cls_name, c_hex in [("LEO", "#2F4B63"), ("MEO", "#B07A1E"), ("GEO", "#7A6A9A"), ("HEO", "#B5543A")]:
                sub_f = fleet_df[fleet_df["orbit_class"] == cls_name]
                if len(sub_f):
                    fig_fl1.add_trace(go.Scatter(
                        x=sub_f["altitude_km"], y=sub_f["eclipse_cycles_per_day"], mode="markers",
                        name=f"{cls_name} (n={len(sub_f)})", text=sub_f["satellite"],
                        marker=dict(color=c_hex, size=7, opacity=0.85),
                    ))
            st.plotly_chart(style(
                fig_fl1, "All Loaded Satellites: Eclipse Cycles/Day vs. Orbital Altitude",
                ytitle="Estimated eclipse cycles / day", xtitle="Orbital altitude (km)",
            ))
        with f_col2:
            fig_fl2 = go.Figure()
            for cls_name, c_hex in [("LEO", "#2F4B63"), ("MEO", "#B07A1E"), ("GEO", "#7A6A9A"), ("HEO", "#B5543A")]:
                sub_f = fleet_df[fleet_df["orbit_class"] == cls_name]
                if len(sub_f):
                    fig_fl2.add_trace(go.Scatter(
                        x=sub_f["inclination_deg"], y=sub_f["sim_1yr_SOH_pct"], mode="markers",
                        name=f"{cls_name} (n={len(sub_f)})", text=sub_f["satellite"],
                        marker=dict(color=c_hex, size=7, opacity=0.85),
                    ))
            st.plotly_chart(style(
                fig_fl2, f"Hypothetical 1-Year Projected SOH (%) vs. Inclination (DOD={int(dod_fleet*100)}%)",
                ytitle="Hypothetical 1-yr SOH (%)", xtitle="Orbital inclination (deg)",
            ))
        st.markdown("**Summary by Orbit Regime (LEO / MEO / GEO / HEO)**")
        st.dataframe(cls_counts.style.format(precision=2, na_rep="—"), width="stretch", hide_index=True)
        st.markdown(f"**Complete Applicable Satellite Fleet Table ({len(fleet_df)} Unique Satellites)**")
        st.dataframe(fleet_df.style.format(precision=2, na_rep="—"), height=360, width="stretch", hide_index=True)
        st.download_button(
            "Download all-satellites fleet simulation CSV",
            fleet_df.to_csv(index=False),
            f"satellite_fleet_simulation_{bat}.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- 8. Methodology & Research Summary
with tabs[7]:
    paper_path = Path(__file__).parent / "research_summary.md"
    if paper_path.exists():
        st.markdown(paper_path.read_text())
    else:
        st.info("research_summary.md not found.")

# --------------------------------------------------------------------------- 9. Results & Export
with tabs[8]:
    out = df.copy()
    out["split"] = np.where(out.index.isin(train.index), "train", "test")
    out["predicted_capacity_Ah"] = core.predict(beta, out.discharge_cycle)
    out["residual_Ah"] = out.capacity_Ah - out.predicted_capacity_Ah
    out["threshold_Ah"] = thr
    out["value_type"] = "capacity_Ah = measured observation ; predicted_capacity_Ah = linear regression output"

    cad = RUL["cadence"]
    summary = pd.DataFrame([{
        "battery": bat,
        "data_source": provenance_label,
        "train_fraction": frac,
        "n_total": RUL["n_total"],
        "n_train": RUL["n_train"],
        "n_test": RUL["n_test"],
        "train_cycle_start": RUL["train_cycle_range"][0],
        "cutoff_cycle": RUL["cutoff_cycle"],
        "cutoff_timestamp": RUL["cutoff_timestamp_str"],
        "cutoff_capacity_Ah": RUL["cutoff_capacity_Ah"],
        "beta0_GE_Ah": F["beta_ge"][0],
        "beta1_GE_Ah_per_cycle": F["beta_ge"][1],
        "beta0_GJ_Ah": F["beta_gj"][0],
        "beta1_GJ_Ah_per_cycle": F["beta_gj"][1],
        "max_abs_diff_GE_GJ": RUL["ge_gj_max_diff"],
        "MAE_Ah": M["MAE"],
        "RMSE_Ah": M["RMSE"],
        "R2": M["R2"],
        "MAPE_pct": M["MAPE_%"],
        "RMSE_pct_of_mean": M["RMSE_%_of_mean"],
        "threshold_Ah": thr,
        "pred_crossing_cycle": RUL["pred_crossing_cycle"],
        "pred_RUL_cycles": RUL["pred_rul_cycles"],
        "pred_status": RUL["pred_status_label"],
        "obs_crossing_cycle": RUL["obs_crossing_cycle"],
        "obs_crossing_capacity_Ah": RUL["obs_crossing_cap_Ah"],
        "obs_crossing_timestamp": RUL["obs_crossing_ts_str"],
        "actual_RUL_cycles": RUL["actual_rul_cycles"],
        "RUL_cycle_error_pred_minus_obs": RUL["cycle_error_pred_minus_actual"],
        "median_training_interval_hours": cad["median_hours"],
        "mean_training_interval_hours": cad["mean_hours"],
        "iqr_training_interval_hours": cad["iqr_hours"],
        "pred_RUL_hours_median": RUL["pred_rul_hours_median"],
        "pred_RUL_days_median": RUL["pred_rul_days_median"],
        "pred_crossing_timestamp_median": str(RUL["pred_crossing_ts_median"]) if RUL["pred_crossing_ts_median"] is not None else None,
        "pred_RUL_hours_mean": RUL["pred_rul_hours_mean"],
        "pred_RUL_days_mean": RUL["pred_rul_days_mean"],
        "actual_elapsed_hours": RUL["actual_elapsed_hours"],
        "actual_elapsed_days": RUL["actual_elapsed_days"],
        "time_error_hours_median": RUL["time_error_hours_median"],
        "time_error_days_median": RUL["time_error_days_median"],
        "crossing_validation_status": RUL["crossing_validation_status"],
    }])
    st.markdown("#### Per-Cycle Predictions & Residuals (`" + bat + "`)")
    st.dataframe(out, height=300, width="stretch")
    a, b, c = st.columns(3)
    a.download_button("Download per-cycle predictions CSV", out.to_csv(index=False), f"{bat}_predictions_rul.csv", "text/csv")
    b.download_button("Download selected battery RUL summary CSV", summary.to_csv(index=False), f"{bat}_rul_summary.csv", "text/csv")
    c.download_button("Download all-batteries comparison CSV", ALL_SUMMARY_DF.to_csv(index=False), "all_batteries_rul_validation.csv", "text/csv")
    st.markdown("#### Selected Battery Full Summary Record")
    st.dataframe(summary.T.rename(columns={0: "value"}).astype(str), width="stretch")
