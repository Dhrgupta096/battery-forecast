"""Core numerics: data loading, chronological split, hand-written elimination solvers, metrics.

Everything the dashboard reports is computed here so it can be checked independently
(see check_core.py). No library solver is used inside the elimination routines;
numpy / scikit-learn solvers are only used afterwards as *independent verification*.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
NASA_CSV = DATA_DIR / "nasa_pcoe_discharge_capacity.csv"
NASA_ZIP = DATA_DIR / "nasa_battery.zip"
NASA_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_AH = 2.0  # NASA PCoE README: 18650 Li-ion cells, rated capacity 2 Ah
EOL_AH = 1.4    # NASA PCoE README: experiments stopped at 30 % fade (2 Ah -> 1.4 Ah)

SOURCE = (
    "NASA Ames Prognostics Center of Excellence (PCoE) Battery Data Set — "
    "B. Saha and K. Goebel (2007), 'Battery Data Set', NASA Ames Prognostics Data Repository, "
    "NASA Ames Research Center, Moffett Field, CA. Archive: "
    "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
)


# --------------------------------------------------------------------------- data
def build_nasa_csv(zip_path: Path = NASA_ZIP, out_csv: Path = NASA_CSV) -> pd.DataFrame:
    """Extract per-discharge-cycle capacity (Ah) from the official NASA .mat files.

    Preprocessing (nothing else is done): keep only records whose type == 'discharge',
    number them 1..N in recorded order (= discharge_cycle), copy the 'Capacity' field (Ah),
    ambient temperature (degC) and start timestamp. Records with no capacity value are
    dropped and counted -- never filled in.
    """
    from scipy.io import loadmat

    rows = []
    with zipfile.ZipFile(zip_path) as outer:
        inner_name = next(n for n in outer.namelist() if "FY08Q4" in n and n.endswith(".zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            for bat in NASA_BATTERIES:
                mat_name = next(n for n in inner.namelist() if n.endswith(f"{bat}.mat"))
                mat = loadmat(io.BytesIO(inner.read(mat_name)), simplify_cells=True)
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
                        "discharge_cycle": k,
                        "capacity_Ah": float(cap[0]),
                        "ambient_temperature_C": float(c["ambient_temperature"]),
                        "start_time": pd.Timestamp(*t[:5], int(t[5])).isoformat(),
                    })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_nasa() -> pd.DataFrame:
    if not NASA_CSV.exists():
        if not NASA_ZIP.exists():
            raise FileNotFoundError("NASA data not found. Run: python fetch_data.py")
        return build_nasa_csv()
    return pd.read_csv(NASA_CSV)


def clean_upload(raw: pd.DataFrame, cycle_col: str, cap_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Validate user CSV. Non-numeric / missing rows are dropped and reported, never imputed."""
    notes = []
    df = pd.DataFrame({
        "discharge_cycle": pd.to_numeric(raw[cycle_col], errors="coerce"),
        "capacity_Ah": pd.to_numeric(raw[cap_col], errors="coerce"),
    })
    bad = df.isna().any(axis=1) | ~np.isfinite(df).all(axis=1)
    if bad.any():
        notes.append(f"Dropped {int(bad.sum())} row(s) with missing/non-numeric values (not imputed).")
    df = df[~bad]
    dup = df["discharge_cycle"].duplicated(keep="first")
    if dup.any():
        notes.append(f"Dropped {int(dup.sum())} duplicate cycle index row(s) (kept first occurrence).")
        df = df[~dup]
    df = df.sort_values("discharge_cycle").reset_index(drop=True)
    df.insert(0, "battery", "UPLOAD")
    return df, notes


# --------------------------------------------------------------------------- split
def chronological_split(df: pd.DataFrame, train_frac: float):
    """First floor(frac*n) cycles train, the rest test. Never shuffled."""
    df = df.sort_values("discharge_cycle").reset_index(drop=True)
    n_train = int(np.floor(train_frac * len(df)))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def design_matrix(cycles) -> np.ndarray:
    k = np.asarray(cycles, dtype=float)
    return np.column_stack([np.ones_like(k), k])


def normal_equations(train: pd.DataFrame):
    X = design_matrix(train["discharge_cycle"])
    y = train["capacity_Ah"].to_numpy(float)
    return X, y, X.T @ X, X.T @ y


# --------------------------------------------------------------------------- solvers
def _fmt(v: float) -> str:
    return f"{v:.6g}"


def gaussian_elimination(A, b, tol: float = 1e-12):
    """Forward elimination to REF (partial pivoting) then back-substitution.

    Returns (x, steps). Each step: {"phase", "op", "matrix"} where matrix is the
    augmented matrix *after* the operation (back-substitution steps carry the solved value).
    """
    M = np.column_stack([np.array(A, float), np.array(b, float)])
    n = M.shape[0]
    steps = [{"phase": "Start", "op": "Augmented matrix [XᵀX | Xᵀy]", "matrix": M.copy()}]
    for col in range(n):
        p = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[p, col]) < tol:
            raise np.linalg.LinAlgError("Singular normal equations (need ≥2 distinct cycle values).")
        if p != col:
            M[[col, p]] = M[[p, col]]
            steps.append({"phase": "REF", "op": f"R{col+1} ↔ R{p+1}  (partial pivot)", "matrix": M.copy()})
        for r in range(col + 1, n):
            f = M[r, col] / M[col, col]
            if f == 0:
                continue
            M[r] = M[r] - f * M[col]
            M[r, col] = 0.0  # exact zero below pivot
            steps.append({"phase": "REF", "op": f"R{r+1} ← R{r+1} − ({_fmt(f)})·R{col+1}", "matrix": M.copy()})
    steps.append({"phase": "REF", "op": "Row-echelon form reached (upper triangular)", "matrix": M.copy()})
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = M[i, -1] - M[i, i + 1:n] @ x[i + 1:]
        x[i] = s / M[i, i]
        known = "".join(f" − ({_fmt(M[i, j])})·β{j}" for j in range(i + 1, n))
        steps.append({"phase": "Back-substitution",
                      "op": f"β{i} = ({_fmt(M[i, -1])}{known}) / {_fmt(M[i, i])} = {x[i]:.10g}",
                      "matrix": M.copy()})
    return x, steps


def gauss_jordan(A, b, tol: float = 1e-12):
    """Full reduction to RREF (partial pivoting). Solution is read from the last column."""
    M = np.column_stack([np.array(A, float), np.array(b, float)])
    n = M.shape[0]
    steps = [{"phase": "Start", "op": "Augmented matrix [XᵀX | Xᵀy]", "matrix": M.copy()}]
    for col in range(n):
        p = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[p, col]) < tol:
            raise np.linalg.LinAlgError("Singular normal equations (need ≥2 distinct cycle values).")
        if p != col:
            M[[col, p]] = M[[p, col]]
            steps.append({"phase": "RREF", "op": f"R{col+1} ↔ R{p+1}  (partial pivot)", "matrix": M.copy()})
        piv = M[col, col]
        if piv != 1.0:
            M[col] = M[col] / piv
            M[col, col] = 1.0
            steps.append({"phase": "RREF", "op": f"R{col+1} ← R{col+1} / ({_fmt(piv)})  (make pivot 1)", "matrix": M.copy()})
        for r in range(n):
            if r == col or M[r, col] == 0:
                continue
            f = M[r, col]
            M[r] = M[r] - f * M[col]
            M[r, col] = 0.0
            steps.append({"phase": "RREF", "op": f"R{r+1} ← R{r+1} − ({_fmt(f)})·R{col+1}", "matrix": M.copy()})
    steps.append({"phase": "RREF", "op": "Reduced row-echelon form: [I | β]", "matrix": M.copy()})
    return M[:, -1].copy(), steps


# --------------------------------------------------------------------------- metrics
def fit(train: pd.DataFrame):
    """Fit on training rows only. Returns beta (GE), beta (GJ) and both step logs."""
    X, y, XtX, Xty = normal_equations(train)
    b_ge, s_ge = gaussian_elimination(XtX, Xty)
    b_gj, s_gj = gauss_jordan(XtX, Xty)
    return {"X": X, "y": y, "XtX": XtX, "Xty": Xty,
            "beta_ge": b_ge, "steps_ge": s_ge, "beta_gj": b_gj, "steps_gj": s_gj}


def predict(beta, cycles) -> np.ndarray:
    return design_matrix(cycles) @ np.asarray(beta)


def metrics(y_true, y_pred) -> dict:
    """Held-out metrics. Undefined quantities are returned as None with a reason, never 0."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    out = {"n": int(y.size), "MAE": None, "RMSE": None, "R2": None, "MAPE_%": None, "RMSE_%_of_mean": None, "notes": []}
    if y.size == 0:
        out["notes"].append("No held-out observations.")
        return out
    e = y - p
    out["MAE"] = float(np.mean(np.abs(e)))
    out["RMSE"] = float(np.sqrt(np.mean(e ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if y.size < 2 or ss_tot == 0:
        out["notes"].append("R² undefined (test targets have zero variance or n<2).")
    else:
        out["R2"] = 1 - float(np.sum(e ** 2)) / ss_tot
    nz = y != 0
    if nz.any():
        out["MAPE_%"] = float(np.mean(np.abs(e[nz] / y[nz])) * 100)
        if (~nz).any():
            out["notes"].append(f"MAPE excludes {int((~nz).sum())} zero-capacity observation(s).")
    else:
        out["notes"].append("MAPE undefined (all observed values are zero).")
    if y.mean() != 0:
        out["RMSE_%_of_mean"] = out["RMSE"] / abs(float(y.mean())) * 100
    else:
        out["notes"].append("RMSE % undefined (mean observed capacity is zero).")
    return out


def threshold_cycle(beta, threshold_ah: float):
    """Cycle k where b0 + b1 k = threshold. None if the fitted line never declines to it."""
    b0, b1 = beta
    if b1 >= 0:
        return None
    return (threshold_ah - b0) / b1


# --------------------------------------------------------------------------- cycles -> calendar time
def cycles_to_time(k: float, cycles_per_day: float, start):
    """Convert a cycle count to elapsed time and a UTC date under an *assumed* cycles/day (Julian year = 365.25 d).

    Returns None if the inputs are not meaningful. Elapsed time is split into years/days/hours/min/sec.
    """
    from datetime import timedelta
    if k is None or not np.isfinite(k) or k < 0 or not cycles_per_day or cycles_per_day <= 0:
        return None
    secs = k / cycles_per_day * 86400.0
    y, r = divmod(secs, 31557600.0)
    d, r = divmod(r, 86400.0)
    h, r = divmod(r, 3600.0)
    m, s = divmod(r, 60.0)
    try:
        date = start + timedelta(seconds=secs)
    except OverflowError:
        date = None
    return {"seconds": secs, "years": int(y), "days": int(d), "hours": int(h), "minutes": int(m), "seconds_part": int(s), "date": date}


# --------------------------------------------------------------------------- data validation
def validate_dataset(df: pd.DataFrame, nasa: bool = False) -> list[tuple[str, bool, str]]:
    """Structural and physical sanity checks on one battery's table. Returns (check, passed, detail)."""
    out = []
    n = len(df)
    out.append(("Has at least 5 rows", n >= 5, f"{n} rows"))
    cyc, cap = df["discharge_cycle"].to_numpy(float), df["capacity_Ah"].to_numpy(float)
    out.append(("No missing / non-finite values", bool(np.isfinite(cyc).all() and np.isfinite(cap).all()), ""))
    out.append(("Cycle index strictly increasing, no duplicates", bool(n < 2 or (np.diff(cyc) > 0).all()), ""))
    out.append(("Capacity strictly positive", bool((cap > 0).all()), f"min {np.nanmin(cap):.4f} Ah" if n else ""))
    out.append(("Capacity physically plausible (< 10 Ah for a single cell)", bool((cap < 10).all()), f"max {np.nanmax(cap):.4f} Ah" if n else ""))
    if n >= 2:
        jump = np.abs(np.diff(cap))
        out.append(("No single-cycle jump larger than 25 % of rated capacity (spike check)",
                    bool((jump <= 0.25 * RATED_AH).all()), f"largest jump {jump.max():.4f} Ah"))
        gaps = np.diff(cyc)
        out.append(("Cycle index contiguous (no gaps)", bool((gaps == 1).all()), f"largest gap {int(gaps.max())}"))
    if nasa and n:
        out.append(("Starts near rated capacity (within 1.0 Ah of 2 Ah)", abs(cap[0] - RATED_AH) < 1.0, f"first = {cap[0]:.4f} Ah"))
        out.append(("Net capacity fade over the test (last < first)", bool(cap[-1] < cap[0]), f"{cap[0]:.3f} to {cap[-1]:.3f} Ah"))
        if "ambient_temperature_C" in df:
            t = df["ambient_temperature_C"]
            out.append(("Ambient temperature recorded and within 0-60 °C", bool(t.between(0, 60).all()), f"{t.min():.1f} to {t.max():.1f} °C"))
    return out


def tle_checksum_ok(line: str) -> bool:
    """NORAD two-line element checksum: digits add their value, '-' adds 1, sum mod 10 equals the last character."""
    if len(line) < 69 or not line[68].isdigit():
        return False
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10 == int(line[68])
