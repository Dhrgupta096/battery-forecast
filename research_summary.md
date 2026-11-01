# Matrix-Based Lithium-Ion Battery Degradation Forecasting and Remaining Useful Life Estimation Using NASA Experimental Data

## 1. Title and Research Question
**Title:** Matrix-Based Lithium-Ion Battery Degradation Forecasting and Remaining Useful Life Estimation Using NASA Experimental Data

**Research Question:** To what extent can ordinary least-squares linear regression—formulated explicitly via the normal equations and solved transparently through Gaussian elimination (Row-Echelon Form, REF) and Gauss–Jordan elimination (Reduced Row-Echelon Form, RREF)—forecast lithium-ion discharge capacity fade and estimate cycle-based and timestamp-cadence-based Remaining Useful Life (RUL) under chronological holdout validation on the NASA Ames Prognostics Center of Excellence (PCoE) aging dataset?

---

## 2. Problem Motivation
Lithium-ion batteries degrade across repeated charge–discharge cycling due to loss of lithium inventory, active material degradation, and solid-electrolyte interphase (SEI) growth. In both terrestrial energy storage and aerospace power subsystems, estimating when a cell's discharge capacity will cross a defined End-of-Life (EOL) threshold is critical for operational planning. Before deploying complex nonlinear or machine-learning architectures, a rigorous, transparent baseline is needed whose mathematical steps, matrix conditioning, chronological holdout errors, and failure modes can be inspected directly.

---

## 3. NASA Dataset Provenance and Citation
- **Primary Source Used in This Project:** Official NASA Ames Prognostics Center of Excellence (PCoE) MATLAB archive (`data/nasa_battery.zip` → `1. BatteryAgingARC-FY08Q4.zip` → `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`), extracted without modification into `data/nasa_pcoe_discharge_capacity.csv` (636 total discharge observations).
- **Official Dataset Page:** [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- **Original Repository Citation:** B. Saha and K. Goebel (2007), *"Battery Data Set"*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- **Secondary / Third-Party Processed Representation (Distinguished from Official Source):** Public cleaned CSV extractions such as [https://github.com/Wondermindimran/Battery_Energy_Storage_System_Performance_Analysis](https://github.com/Wondermindimran/Battery_Energy_Storage_System_Performance_Analysis) (`clean_discharge.csv`) may be loaded via the application's CSV upload interface. Third-party exports are explicitly distinguished from the original NASA `.mat` files (notably, `clean_discharge.csv` truncates timestamps to the minute, whereas the original `.mat` files record timestamps to the second).

---

## 4. Experimental Protocol, Data Extraction, and Preprocessing
### 4.1 Experimental Conditions (NASA PCoE Room-Temperature Group)
Four commercial 18650 lithium-ion rechargeable cells (`B0005`, `B0006`, `B0007`, `B0018`), rated at $2.0\text{ Ah}$, were cycled at an ambient temperature of $24\,^\circ\text{C}$:
- **Charge profile:** Constant current (CC) at $1.5\text{ A}$ until cell voltage reached $4.2\text{ V}$, followed by constant voltage (CV) until charge current dropped to $20\text{ mA}$.
- **Discharge profile:** Constant current (CC) at $2.0\text{ A}$ down to cell-specific cut-off voltages: $2.7\text{ V}$ (`B0005`), $2.5\text{ V}$ (`B0006`), $2.2\text{ V}$ (`B0007`), and $2.5\text{ V}$ (`B0018`).
- **Reported experimental EOL criterion:** $30\%$ fade from the $2.0\text{ Ah}$ rated capacity, corresponding to $1.4\text{ Ah}$.

### 4.2 Extraction and Preprocessing Steps
1. **Operation Filtering:** Each `.mat` structure contains interleaved `charge`, `discharge`, and `impedance` records. Only records with `type == 'discharge'` are selected.
2. **Sequential Discharge Indexing ($k$):** Discharge events are numbered sequentially $k = 1, 2, \dots, N$ in recorded chronological order for each battery (`discharge_cycle` / `discharge_cycle_num`). General cycle indices that count charge and impedance operations are not confused with $k$.
3. **Full Precision and Recorded Timestamps:** Measured capacity (`Capacity`, in $\text{Ah}$) is retained in full IEEE-754 double precision (`float64`) internally. Each discharge start vector `[YYYY, MM, DD, hh, mm, ss]` is preserved exactly as recorded (`YYYY-MM-DDTHH:MM:SS`) without inventing or appending a timezone.
4. **No Imputation or Smoothing:** Records missing a finite capacity value are dropped and counted (0 dropped in `B0005`, `B0006`, `B0007`, `B0018`). No outlier removal, interpolation, or smoothing is applied.

---

## 5. Variables, Units, and Data-Quality Audit
| Variable | Symbol | Units | Description |
| :--- | :--- | :--- | :--- |
| Battery identifier | — | categorical | `B0005`, `B0006`, `B0007`, `B0018` (or uploaded cell ID) |
| Sequential discharge-cycle number | $k$ | dimensionless (cycles) | $1, 2, \dots, N$ counting discharge events only |
| Measured discharge capacity | $Q(k)$ | $\text{Ah}$ | Integrated discharge capacity per discharge cycle |
| Predicted discharge capacity | $\hat{Q}(k)$ | $\text{Ah}$ | Linear regression model output |
| Recorded start timestamp | $t_k$ | ISO-8601 local/recorded | Experimental start time of discharge cycle $k$ |
| Inter-discharge time interval | $\Delta t_k$ | $\text{hours}$ | $t_k - t_{k-1}$ for consecutive training discharges |
| Ambient temperature | $T_{\text{amb}}$ | $^\circ\text{C}$ | Recorded chamber temperature ($24.0\,^\circ\text{C}$) |

### Data-Quality Audit Summary Across Selected NASA Cells
| Battery | Total Discharge Cycles ($N$) | Initial Capacity ($\text{Ah}$) | Final Capacity ($\text{Ah}$) | Missing / Duplicate / Invalid Rows | Non-Monotonic Cycles or Timestamps | Pauses $> 24\text{ h}$ (Full Series) | Max Interval ($\text{hours}$) | First Recorded Timestamp | Last Recorded Timestamp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0005` | 168 | 1.856487 | 1.325079 | 0 / 0 / 0 | 0 / 0 | 4 | 310.396 | `2008-04-02T15:25:41` | `2008-05-27T20:45:42` |
| `B0006` | 168 | 2.035338 | 1.185675 | 0 / 0 / 0 | 0 / 0 | 4 | 310.396 | `2008-04-02T15:25:41` | `2008-05-27T20:45:42` |
| `B0007` | 168 | 1.891052 | 1.432455 | 0 / 0 / 0 | 0 / 0 | 4 | 310.396 | `2008-04-02T15:25:41` | `2008-05-27T20:45:42` |
| `B0018` | 132 | 1.855005 | 1.341051 | 0 / 0 / 0 | 0 / 0 | 7 | 244.691 | `2008-07-07T15:15:28` | `2008-08-20T08:37:19` |

---

## 6. Mathematical Formulation: Linear Regression and Matrix Elimination
### 6.1 Linear Degradation Model and Normal Equations
We model discharge capacity as an affine function of the sequential discharge-cycle number $k$:
$$\hat{Q}(k) = \beta_0 + \beta_1 k$$
where $\beta_0$ ($\text{Ah}$) is the fitted intercept and $\beta_1$ ($\text{Ah/cycle}$) is the fitted capacity change per discharge cycle. Given $n_{\text{train}}$ chronological training observations $\{(k_i, Q_i)\}_{i=1}^{n_{\text{train}}}$, the design matrix $X \in \mathbb{R}^{n_{\text{train}} \times 2}$ and target vector $\mathbf{y} \in \mathbb{R}^{n_{\text{train}}}$ are:
$$X = \begin{bmatrix} 1 & k_1 \\ 1 & k_2 \\ \vdots & \vdots \\ 1 & k_{n_{\text{train}}} \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} Q_1 \\ Q_2 \\ \vdots \\ Q_{n_{\text{train}}} \end{bmatrix}, \qquad \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}$$
Minimizing the sum of squared training residuals $\|\mathbf{y} - X\boldsymbol{\beta}\|_2^2$ yields the $2 \times 2$ normal equations:
$$X^T X \boldsymbol{\beta} = X^T \mathbf{y}$$

### 6.2 Role of Gaussian Elimination (REF) and Gauss–Jordan Elimination (RREF)
To provide full pedagogical and numerical transparency, the augmented matrix $[X^T X \mid X^T \mathbf{y}]$ is solved via two separate hand-written algorithms in `core.py`:
1. **Gaussian Elimination to Row-Echelon Form (REF):** Applies partial pivoting ($R_i \leftrightarrow R_p$) to select the largest-magnitude pivot in each column, eliminates sub-diagonal entries to produce an upper-triangular matrix, and solves for $\beta_1, \beta_0$ via back-substitution.
2. **Gauss–Jordan Elimination to Reduced Row-Echelon Form (RREF):** Applies partial pivoting, normalizes each pivot row to unity ($R_i \leftarrow R_i / \text{pivot}$), and eliminates both above- and below-diagonal entries until $[I \mid \boldsymbol{\beta}]$ is obtained directly.

Both methods are verified against each other ($|\boldsymbol{\beta}_{\text{GE}} - \boldsymbol{\beta}_{\text{GJ}}|_\infty \le 10^{-15}$) and cross-checked against QR/SVD least-squares (`numpy.linalg.lstsq` and `scikit-learn`).

---

## 7. Chronological Holdout Protocol and Evaluation Metrics
### 7.1 Chronological Split (Zero Data Leakage)
Time-series observations are never shuffled. For a training fraction $f$ (default $f = 0.60$), the first $n_{\text{train}} = \lfloor f \cdot N \rfloor$ discharge cycles form the training set ($k \in [1, k_{\text{cut}}]$), and the remaining $n_{\text{test}} = N - n_{\text{train}}$ cycles form the held-out test set ($k \in [k_{\text{cut}}+1, N]$). Test observations are never used to fit coefficients, select the threshold, or tune parameters.

### 7.2 Held-Out Capacity Prediction Error Metrics
All accuracy metrics are computed exclusively on the $n_{\text{test}}$ held-out observations:
$$\text{MAE} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} |Q_i - \hat{Q}_i|\;\text{(Ah)}, \qquad \text{RMSE} = \sqrt{\frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} (Q_i - \hat{Q}_i)^2}\;\text{(Ah)}$$
$$R^2 = 1 - \frac{\sum_{i=1}^{n_{\text{test}}} (Q_i - \hat{Q}_i)^2}{\sum_{i=1}^{n_{\text{test}}} (Q_i - \bar{Q}_{\text{test}})^2}, \qquad \text{MAPE} = \frac{100}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} \left|\frac{Q_i - \hat{Q}_i}{Q_i}\right|\;(\%), \qquad \text{RMSE}_{\%\text{mean}} = \frac{100 \cdot \text{RMSE}}{\bar{Q}_{\text{test}}}\;(\%)$$

---

## 8. Remaining Useful Life (RUL) and Recorded-Timestamp Cadence Methodology
### 8.1 Cycle-Based RUL and Safeguards
For a configured EOL threshold $Q_{\text{thresh}}$ (default $1.4\text{ Ah}$) and a fitted model with $\beta_1 < 0$:
$$\hat{k}_{\text{cross}} = \frac{Q_{\text{thresh}} - \beta_0}{\beta_1}, \qquad \widehat{\text{RUL}}_{\text{cycles}} = \hat{k}_{\text{cross}} - k_{\text{cut}}$$
Explicit safeguards prevent misleading outputs:
- If $\beta_1 \ge 0$, no crossing is predicted.
- If $Q_i \le Q_{\text{thresh}}$ already occurred in the training window, that condition is reported rather than claiming positive remaining life.
- If $\hat{k}_{\text{cross}} \le k_{\text{cut}}$ (as occurs for `B0006` at $f=0.60$, where $\hat{k}_{\text{cross}} = 98.878 < 100$), the raw negative $\widehat{\text{RUL}}_{\text{cycles}}$ ($-1.122$ cycles) is preserved without clipping to zero, and the model reports *"No positive predicted RUL from this model"*.

### 8.2 Time-Based RUL from Recorded Experimental Timestamps
Rather than assuming an arbitrary 1 cycle/day rate, we compute consecutive positive training time intervals $\Delta t_i = t_i - t_{i-1} > 0$ (in hours) from the recorded timestamps:
- Because laboratory experiments include multi-day pauses between test batches (up to $310.4\text{ h}$ in `B0005`–`B0007` and $244.7\text{ h}$ in `B0018`), the sample mean interval ($\approx 9.47\text{ h}$ for `B0005`–`B0007`) isnearly double the typical cycling cadence.
- Therefore, the **median positive training interval** $\Delta t_{\text{med}}$ is used as the primary cadence estimator, with the **mean interval** $\Delta t_{\text{mean}}$ reported alongside as a pause-sensitivity comparison:
$$\widehat{\text{RUL}}_{\text{hours, med}} = \widehat{\text{RUL}}_{\text{cycles}} \times \Delta t_{\text{med}}, \qquad \widehat{\text{RUL}}_{\text{days, med}} = \frac{\widehat{\text{RUL}}_{\text{hours, med}}}{24}, \qquad \hat{t}_{\text{cross, med}} = t_{k_{\text{cut}}} + \widehat{\text{RUL}}_{\text{hours, med}}$$

---

## 9. Empirical Results Across All Four NASA Batteries (60% Train / 40% Test, $1.4\text{ Ah}$ Threshold)

### 9.1 Regression Coefficients and Held-Out Capacity Error Metrics
| Battery | Train Cycles ($n_{\text{train}}$) | Test Cycles ($n_{\text{test}}$) | $\beta_0$ ($\text{Ah}$) | $\beta_1$ ($\text{Ah/cycle}$) | $|\boldsymbol{\beta}_{\text{GE}} - \boldsymbol{\beta}_{\text{GJ}}|_\infty$ | Held-Out MAE ($\text{Ah}$) | Held-Out RMSE ($\text{Ah}$) | Held-Out $R^2$ | Held-Out MAPE ($\%$) | $\text{RMSE}_{\%\text{mean}}$ ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0005` | 1–100 (100) | 101–168 (68) | `1.901404883` | `-0.003843534` | `0.00e+00` | `0.022708` | `0.025595` | `0.804869` | `1.6471%` | `1.8624%` |
| `B0006` | 1–100 (100) | 101–168 (68) | `2.033768795` | `-0.006409625` | `8.88e-16` | `0.135129` | `0.144032` | `-2.252947` | `10.5856%` | `11.0217%` |
| `B0007` | 1–100 (100) | 101–168 (68) | `1.935394085` | `-0.003567138` | `0.00e+00` | `0.027628` | `0.035518` | `0.467199` | `1.9061%` | `2.3975%` |
| `B0018` | 1–79 (79) | 80–132 (53) | `1.845466604` | `-0.004582946` | `4.44e-16` | `0.051148` | `0.066638` | `-2.956006` | `3.6685%` | `4.7530%` |

### 9.2 Cycle-Based and Timestamp-Cadence RUL Validation Summary
| Battery | Cutoff $k_{\text{cut}}$ & Capacity | Pred. Crossing $\hat{k}_{\text{cross}}$ | Pred. $\widehat{\text{RUL}}_{\text{cycles}}$ | First Observed Test Crossing ($k_{\text{obs}}$, $Q_{\text{obs}}$) | Actual $\text{RUL}_{\text{cycles}}$ | Cycle Error ($\hat{k} - k_{\text{obs}}$) | Training Interval Median / Mean / IQR ($\text{h}$) | Pred. Time RUL (Median / Mean, $\text{h}$) | Actual Elapsed Time to Crossing ($\text{h}$) | Median Time Error ($\text{h}$) | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `B0005` | `100` (`1.485868 Ah`) | `130.454` | `+30.454` | `125` (`1.396701 Ah`) | `25` | `+5.454` | `4.953` / `9.469` / `0.193` | `150.841 h` (`6.285 d`) / `288.362 h` | `144.428 h` (`6.018 d`) | `+6.413 h` (`+6h 25m`) | Observed in test & +RUL predicted |
| `B0006` | `100` (`1.431211 Ah`) | `98.878` | `-1.122` (no +RUL) | `109` (`1.395164 Ah`) | `9` | `-10.122` | `4.953` / `9.469` / `0.193` | — (crossing $\le k_{\text{cut}}$) | `50.239 h` (`2.093 d`) | — | Observed in test (`k=109`); model predicted crossing before cutoff (`98.88`) |
| `B0007` | `100` (`1.570257 Ah`) | `150.091` | `+50.091` | Not observed (min `1.400455 Ah`) | — | — | `4.953` / `9.469` / `0.193` | `248.102 h` (`10.338 d`) / `474.295 h` | — | — | Threshold crossing not observed in test horizon |
| `B0018` | `79` (`1.458092 Ah`) | `97.201` | `+18.201` | `97` (`1.396855 Ah`) | `18` | `+0.201` | `3.942` / `9.037` / `0.660` | `71.755 h` (`2.990 d`) / `164.475 h` | `91.572 h` (`3.816 d`) | `-19.818 h` (`-19h 49m`) | Observed in test & +RUL predicted |

### 9.3 Discussion of Cell-by-Cell Findings
1. **`B0005` (Well-behaved linear regime):** The linear fit achieves $R^2 = 0.805$ on the held-out 68 cycles and over-predicts the $1.4\text{ Ah}$ crossing by $+5.454$ cycles ($130.454$ vs. observed cycle $125$). Using the median training interval ($4.953\text{ h}$ from full-precision `.mat` timestamps, or $4.950\text{ h}$ from minute-truncated timestamps), the time forecast is within $+6.41\text{ h}$ ($+6.31\text{ h}$ / $6\text{h }19\text{m}$ under minute truncation) of the actual $144.43\text{ h}$ elapsed duration. By contrast, using the mean interval ($9.469\text{ h}$) inflates the forecast to $288.36\text{ h}$, demonstrating why median cadence is essential when rest pauses are present.
2. **`B0006` (Early rapid fade followed by curvature change):** `B0006` degrades steeply over its first 100 cycles ($\beta_1 = -0.006410\text{ Ah/cycle}$), so the linear trend line already falls below $1.4\text{ Ah}$ at $\hat{k}_{\text{cross}} = 98.878$ even though measured capacity at $k=100$ is still $1.4312\text{ Ah}$ (due to a capacity regeneration jump near cycle 90). Consequently, the linear model yields $\widehat{\text{RUL}}_{\text{cycles}} = -1.122$ at cutoff $k=100$ and under-predicts the actual held-out crossing ($k=109$) by $-10.122$ cycles, with negative test $R^2 = -2.253$.
3. **`B0007` (Threshold never reached in test horizon):** `B0007` degrades more gradually ($\beta_1 = -0.003567\text{ Ah/cycle}$). Although the linear model predicts crossing $1.4\text{ Ah}$ at $\hat{k}_{\text{cross}} = 150.091$ ($\widehat{\text{RUL}} = 50.091$ cycles), the actual cell's minimum measured capacity across all 168 cycles is $1.400455\text{ Ah}$ (at cycle 160). Thus no actual $1.4\text{ Ah}$ crossing occurs within the test horizon.
4. **`B0018` (Accurate crossing cycle, pause-affected time error):** With $n_{\text{train}} = 79$ ($60\%$ of 132 cycles), the linear model predicts crossing at $\hat{k}_{\text{cross}} = 97.201$, matching the observed crossing at $k = 97$ to within $+0.201$ cycles. However, because an unplanned multi-hour laboratory pause occurred between cycles 79 and 97, actual elapsed time ($91.57\text{ h}$) exceeds the median-cadence estimate ($71.75\text{ h}$) by $19.82\text{ h}$, despite negative overall test $R^2 = -2.956$ caused by large capacity-regeneration spikes later in the test window.

---

## 10. Space-Power Relevance and Potential Future Application
In Low-Earth-Orbit (LEO) and Geostationary-Earth-Orbit (GEO) spacecraft, lithium-ion batteries support payload and bus loads during solar eclipse passages. While the orbital visualizer in this dashboard illustrates how eclipse-driven cycling rates depend on orbital altitude and inclination, **all spacecraft battery figures in the orbital view are hypothetical scenarios**. Real spacecraft batteries operate under tightly managed shallow depths of discharge (typically $10\%$–$40\%$ DOD), controlled thermal environments, radiation exposure, and seasonal eclipse profiles. The transparent matrix regression developed here serves as an educational and methodological baseline for terrestrial laboratory data, with potential future extension toward aerospace telemetry once mission-representative data are available.

---

## 11. Limitations and Further Work Required
1. **Small Sample Size:** Only four cells from a single room-temperature ($24\,^\circ\text{C}$) NASA PCoE batch are analyzed; findings cannot be generalized across chemistries or thermal regimes.
2. **Linearity Assumption and Capacity Regeneration:** Lithium-ion aging exhibits nonlinear knee points and temporary capacity self-recharge (regeneration) during rest periods, which an affine model $\hat{Q}(k) = \beta_0 + \beta_1 k$ cannot capture (evidenced by negative held-out $R^2$ on `B0006` and `B0018`).
3. **Irregular Laboratory Timing:** Calendar-time RUL depends heavily on the experimental test schedule; multi-day pauses between discharge cycles decouple cycle count from wall-clock time.
4. **Threshold Sensitivity:** The $1.4\text{ Ah}$ threshold ($30\%$ fade) is the termination criterion of this specific NASA experiment (`B0007` stops at $1.4005\text{ Ah}$), not a universal operational limit.
5. **Lack of Flight Telemetry Validation:** No actual spacecraft battery telemetry is used. Future work before any aerospace deployment requires piecewise/nonlinear state-space filtering (e.g., Kalman or particle filters), impedance (EIS) features, temperature-dependent degradation models, and validation on authentic in-orbit or mission-profile life-test datasets.

---

## 12. References
1. B. Saha and K. Goebel (2007), *"Battery Data Set"*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. Official NASA dataset portal: [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets). Direct archive: [https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip).
2. B. Saha, K. Goebel, S. Poll, and J. Christophersen (2009), *"Prognostics Methods for Battery Health Monitoring Using a Bayesian Framework"*, IEEE Transactions on Instrumentation and Measurement, vol. 58, no. 2, pp. 291–296.
3. Secondary cleaned CSV representation (referenced for third-party column/timestamp comparison): [https://github.com/Wondermindimran/Battery_Energy_Storage_System_Performance_Analysis](https://github.com/Wondermindimran/Battery_Energy_Storage_System_Performance_Analysis).
