import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM
 
from src.config import DEFAULT_CONFIG, all_training_windows
from src.pairs_discovery.kalman_hedge import kalman_spread
 
# The OU process in continuous time is: dX_t = κ(θ - X_t)dt + σ dW_t
# Discretised over dt=1 day it becomes an AR(1): X_t = a + b·X_{t-1} + ε_t, ε_t ~ N(0, σ_ε²) where:
#   b = exp(-κ·dt) (AR coefficient)
#   a = θ·(1 - b) (intercept)
#   σ_ε² = σ²·(1 - exp(-2κ·dt)) / (2κ) (innovation variance)
# We recover the OU parameters from OLS estimates of (a, b, σ_ε).
 
@njit(cache=True)
def _fit_ar1(x: np.ndarray) -> tuple:
    """OLS fit of AR(1): x_t = a + b·x_{t-1} + ε_t."""
    x_prev, x_curr = x[:-1], x[1:]
    n = len(x_prev)
    mx, my = np.mean(x_prev), np.mean(x_curr)
    num = np.sum((x_prev - mx) * (x_curr - my))
    den = np.sum((x_prev - mx)**2)
    if den == 0: return np.nan, np.nan, np.nan
    b = num / den
    a = my - b * mx
    resid_std = np.sqrt(np.sum((x_curr - (a + b * x_prev))**2) / (n - 2))
    return b, a, resid_std
 
@njit(cache=True)
def _garch_nll(params: np.ndarray, residuals: np.ndarray, h_init: float) -> float:
    """
    h_t = omega + alpha·ε²_{t-1} + beta·h_{t-1}
    L   = -0.5 · Σ [ ln(h_t) + ε_t² / h_t ]
    """
    omega, alpha, beta = params[0], params[1], params[2]
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10
    h_prev = h_init
    nll = 0.0
    for t in range(1, len(residuals)):
        h_t = omega + alpha * residuals[t-1]**2 + beta * h_prev
        if h_t <= 0:
            return 1e10
        nll += np.log(h_t) + residuals[t]**2 / h_t
        h_prev = h_t
    return 0.5 * nll
 
@njit(cache=True)
def _garch_variance(residuals: np.ndarray, omega: float, alpha: float, beta: float, h_init: float) -> np.ndarray:
    n = len(residuals)
    h = np.empty(n)
    h[0] = h_init
    for t in range(1, n):
        h[t] = omega + alpha * residuals[t-1]**2 + beta * h[t-1]
    return h
 
@njit(cache=True)
def _logaddexp2(a: float, b: float) -> float:
    if a > b:
        return a + np.log(1.0 + np.exp(b - a))
    return b + np.log(1.0 + np.exp(a - b))
 
@njit(cache=True)
def _forward_backward(log_obs: np.ndarray, log_P: np.ndarray, log_pi: np.ndarray) -> tuple:
    n = log_obs.shape[0]
    K = log_obs.shape[1]   # always 2 for our HMM
 
    la = np.full((n, K), -np.inf)   # log forward probabilities
    lb = np.full((n, K), -np.inf)   # log backward probabilities
 
    # Initialise
    for k in range(K):
        la[0, k] = log_pi[k] + log_obs[0, k]
        lb[n-1, k] = 0.0   # log(1)
 
    # Forward pass: α_t(j) = p(x_t|state=j) · Σ_i α_{t-1}(i) · P(i→j)
    for t in range(1, n):
        for j in range(K):
            tmp = -np.inf
            for i in range(K):
                tmp = _logaddexp2(tmp, la[t-1, i] + log_P[i, j])
            la[t, j] = log_obs[t, j] + tmp
 
    # Backward pass: β_t(i) = Σ_j P(i→j) · p(x_{t+1}|state=j) · β_{t+1}(j)
    for t in range(n-2, -1, -1):
        for i in range(K):
            tmp = -np.inf
            for j in range(K):
                tmp = _logaddexp2(tmp, log_P[i, j] + log_obs[t+1, j] + lb[t+1, j])
            lb[t, i] = tmp
 
    # Posterior: γ_t(k) = α_t(k) · β_t(k) / Σ_k α_t(k) · β_t(k)
    log_gamma = la + lb
    for t in range(n):
        norm = _logaddexp2(log_gamma[t, 0], log_gamma[t, 1])
        log_gamma[t, 0] -= norm
        log_gamma[t, 1] -= norm
 
    return la, lb, log_gamma

class OUModel: # Baseline
    # Parameter recovery from AR(1) estimates:
    #  κ = -ln(b) / dt
    #  θ = a / (1 - b)
    #  σ = σ_ε · √(2κ / (1 - b²))
    #  σ_eq = σ / √(2κ) = σ_ε / √(1 - b²)
    # The z-score at time t is: z_t = (X_t - θ) / σ_eq
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.kappa = np.nan
        self.theta = np.nan
        self.sigma = np.nan
        self.eq_std = np.nan
 
    def fit(self, spread_window: np.ndarray) -> bool:
        b, a, resid_std = _fit_ar1(spread_window)
        if np.isnan(b) or not (0.0 < b < 1.0):
            return False
        kappa = -np.log(b) / self.dt
        if kappa <= 1e-12:
            return False
        theta  = a / (1.0 - b)
        sigma  = resid_std * np.sqrt(2.0 * kappa / (1.0 - b**2))
        eq_var = sigma**2 / (2.0 * kappa)
        if eq_var <= 0.0:
            return False
        eq_std = np.sqrt(eq_var)
        if not np.isfinite(eq_std) or eq_std < 1e-8:
            return False
        self.kappa  = kappa
        self.theta  = theta
        self.sigma  = sigma
        self.eq_std = eq_std
        return True
 
    def get_z_score(self, current_value: float) -> float:
        if np.isnan(self.theta) or self.eq_std == 0:
            return np.nan
        return (current_value - self.theta) / self.eq_std
 
    @property
    def half_life(self) -> float:
        if np.isnan(self.kappa) or self.kappa <= 0.0:
            return np.inf
        return np.log(2.0) / self.kappa
 
# Model 2 — OU + GARCH(1,1)
# Problem: constant eq_std means z=2 on a calm day ≠ z=2 on a volatile day.
# To fix, fit GARCH(1,1) on AR(1) residuals → time-varying h_t.
# z_t = (spread_t - θ) / √h_t
# GARCH variance equation: h_t = ω + α·ε²_{t-1} + β·h_{t-1}
# Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity)
# Long-run variance: h_∞ = ω / (1 - α - β)
 
class OUGARCHModel:
    """
    Fit steps:
      1. Fit OU/AR(1) → θ and residuals ε_t = spread_t - (a + b·spread_{t-1})
      2. Fit GARCH(1,1) on ε_t via MLE (numba NLL loop, scipy optimiser)
      3. At time t: one-step-ahead forecast h_t = ω + α·ε²_{t-1} + β·h_{t-1}
      4. z_t = (spread_t - θ) / √h_t
    """
    def __init__(self, dt: float = 1.0):
        self.dt      = dt
        self.ou      = OUModel(dt=dt)
        self.omega   = np.nan
        self.alpha_g = np.nan
        self.beta_g  = np.nan
        self._last_eps = np.nan
        self._last_h   = np.nan
 
    def fit(self, spread_window: np.ndarray) -> bool:
        if not self.ou.fit(spread_window):
            return False
 
        b, a, _ = _fit_ar1(spread_window)
        residuals = spread_window[1:] - (a + b * spread_window[:-1])
        if len(residuals) < 30:
            return False
 
        h_inf = float(np.var(residuals))
        if h_inf <= 0:
            return False
 
        def objective(p):
            return _garch_nll(np.asarray(p, dtype=np.float64), residuals, h_inf)
 
        result = minimize(objective, [h_inf * 0.05, 0.10, 0.85], method="L-BFGS-B", bounds=[(1e-9, None), (1e-9, 0.5), (1e-9, 0.999)])
        if not result.success:
            return False
 
        omega, alpha, beta = result.x
        if alpha + beta >= 1.0:
            return False
 
        self.omega, self.alpha_g, self.beta_g = float(omega), float(alpha), float(beta)
        # Store last h and residual for one-step-ahead forecast at score time
        h = _garch_variance(residuals, self.omega, self.alpha_g, self.beta_g, h_inf)
        self._last_h   = float(h[-1])
        self._last_eps = float(residuals[-1])
        return True
 
    def get_z_score(self, current_value: float) -> float:
        if np.isnan(self.ou.theta) or np.isnan(self.omega):
            return np.nan
        h_next = self.omega + self.alpha_g * self._last_eps**2 + self.beta_g * self._last_h
        if h_next <= 0 or not np.isfinite(h_next):
            return np.nan
        return (current_value - self.ou.theta) / np.sqrt(h_next)
 
    @property
    def half_life(self) -> float:
        return self.ou.half_life
 
# Regime-switching OU (2-state HMM via Baum-Welch EM)
# Problem: one (κ, θ, σ) fitted regardless of whether spread is in
# normal mean-reversion or breakdown mode (earnings, sector rotation).
# To fix, use 2-state HMM, each state has its own OU parameters.
#   State 0: normal regime — fast κ
#   State 1: breakdown regime — slow κ, spread drifting
# z_t = P(state=0|data) · (X_t - θ_0) / σ_eq_0
# Signal shrinks toward 0 automatically when breakdown prob is high.
 
class RegimeSwitchingOU:
    N_STATES = 2
    MAX_ITER = 30
    TOL      = 1e-4
 
    def __init__(self, dt: float = 1.0):
        self.dt      = dt
        self.fitted  = False
        self.kappa   = np.full(self.N_STATES, np.nan)
        self.theta   = np.full(self.N_STATES, np.nan)
        self.eq_std  = np.full(self.N_STATES, np.nan)
        self.P_trans = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.pi      = np.array([0.8, 0.2])
 
    def _log_obs_prob(self, x: np.ndarray) -> np.ndarray: # log p(x_t | state=k) for all t and k. Shape: (n, 2).
        n   = len(x)
        out = np.full((n, self.N_STATES), -np.inf)
        for k in range(self.N_STATES):
            if np.isnan(self.kappa[k]) or self.kappa[k] <= 0:
                continue
            b_k     = np.exp(-self.kappa[k] * self.dt)
            a_k     = self.theta[k] * (1.0 - b_k)
            sigma_k = self.eq_std[k] * np.sqrt(1.0 - b_k**2)
            if sigma_k <= 0:
                continue
            predicted = a_k + b_k * x[:-1]
            residuals = x[1:] - predicted
            out[1:, k] = (-0.5 * np.log(2 * np.pi * sigma_k**2)- 0.5 * (residuals / sigma_k)**2)
            out[0, k] = out[1, k]
        return out
 
    def _m_step(self, x: np.ndarray, gamma: np.ndarray): # EM M-step: re-estimate OU params for each state using weighted OLS.
        x_prev, x_curr = x[:-1], x[1:]
        for k in range(self.N_STATES):
            w = gamma[1:, k] + 1e-10
            W = w.sum()
            if W < 5:
                continue
            mx = np.sum(w * x_prev) / W
            my = np.sum(w * x_curr) / W
            num = np.sum(w * (x_prev - mx) * (x_curr - my))
            den = np.sum(w * (x_prev - mx)**2)
            if den < 1e-12:
                continue
            b = num / den
            a = my - b * mx
            if not (0.0 < b < 1.0):
                continue
            kappa = -np.log(b) / self.dt
            if kappa <= 1e-12:
                continue
            theta   = a / (1.0 - b)
            resid   = x_curr - (a + b * x_prev)
            res_var = np.sum(w * resid**2) / W
            sigma_e = np.sqrt(max(res_var, 1e-16))
            sigma   = sigma_e * np.sqrt(2.0 * kappa / (1.0 - b**2))
            eq_std  = sigma / np.sqrt(2.0 * kappa)
            if not np.isfinite(eq_std) or eq_std < 1e-8:
                continue
            self.kappa[k]  = kappa
            self.theta[k]  = theta
            self.eq_std[k] = eq_std
 
        # Update transition matrix from marginal posteriors
        for i in range(self.N_STATES):
            denom = gamma[:-1, i].sum() + 1e-10
            for j in range(self.N_STATES):
                self.P_trans[i, j] = gamma[:-1, i].sum() / denom
            self.P_trans[i] /= self.P_trans[i].sum()
 
    def fit(self, spread_window: np.ndarray) -> bool:
        if len(spread_window) < 40:
            return False
 
        # Initialise by fitting OU on each half of the window
        mid = len(spread_window) // 2
        ou0, ou1 = OUModel(self.dt), OUModel(self.dt)
        if not ou0.fit(spread_window[:mid]) or not ou1.fit(spread_window[mid:]):
            return False
 
        self.kappa  = np.array([ou0.kappa,  ou1.kappa])
        self.theta  = np.array([ou0.theta,  ou1.theta])
        self.eq_std = np.array([ou0.eq_std, ou1.eq_std])
 
        # State 0 = faster mean-reversion (larger kappa)
        if self.kappa[1] > self.kappa[0]:
            self.kappa  = self.kappa[[1, 0]]
            self.theta  = self.theta[[1, 0]]
            self.eq_std = self.eq_std[[1, 0]]
 
        log_P  = np.log(self.P_trans + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        prev_ll = -np.inf
 
        for _ in range(self.MAX_ITER):
            log_obs = self._log_obs_prob(spread_window)
 
            # E-step via compiled forward-backward (78x faster than pure Python)
            _, _, log_gamma = _forward_backward(log_obs, log_P, log_pi)
            gamma = np.exp(log_gamma)
 
            ll = _logaddexp2(log_gamma[-1, 0], log_gamma[-1, 1])
            if abs(ll - prev_ll) < self.TOL:
                break
            prev_ll = ll
 
            self._m_step(spread_window, gamma)
            log_P  = np.log(self.P_trans + 1e-300)
 
        self._last_gamma = gamma[-1]
        self.fitted = True
        return True
 
    def get_z_score(self, current_value: float) -> tuple:
        """
        p_normal = P(state=0 | data) as a confidence filter.
        z_score is already weighted: z = p_normal · (X - θ_0) / σ_eq_0
        """
        if not self.fitted or np.isnan(self.kappa[0]):
            return np.nan, np.nan
        p_normal   = float(self._last_gamma[0])
        z_raw      = (current_value - self.theta[0]) / self.eq_std[0]
        z_weighted = p_normal * z_raw
        return float(z_weighted), p_normal
 
    @property
    def half_life(self) -> float:
        if np.isnan(self.kappa[0]) or self.kappa[0] <= 0:
            return np.inf
        return np.log(2.0) / self.kappa[0]
 
# VECM (Vector Error Correction Model)
# Problem: OU models the spread in isolation. VECM models both stocks
# simultaneously and shows which one drives the correction.
# VECM: Δy_t = α · β'·y_{t-1} + Γ·Δy_{t-1} + ε_t
#   β = cointegrating vector [1, -β_lr]
#   α = adjustment speeds [α_1, α_2]
#     α_1 < 0: stock A corrects toward equilibrium
#     α_2 > 0: stock B corrects toward equilibrium
# z_t = (ec_t - ec_mean) / ec_std  where ec_t = p1_t - β_lr · p2_t
 
class VECMModel:
    def __init__(self, k_ar_diff: int = 1):
        self.k_ar_diff = k_ar_diff
        self.fitted    = False
        self.alpha     = np.full(2, np.nan)
        self.beta_lr   = np.nan
        self.ec_mean   = np.nan
        self.ec_std    = np.nan
 
    def fit(self, p1_window: np.ndarray, p2_window: np.ndarray) -> bool:
        if len(p1_window) < 30:
            return False
        data = pd.DataFrame({"p1": p1_window, "p2": p2_window})
        try:
            result = VECM(data, k_ar_diff=self.k_ar_diff, coint_rank=1, deterministic="ci").fit()
        except Exception:
            return False
 
        # statsmodels normalises β so β[0]=1, giving β = [1, -β_lr]
        beta_vec = result.beta.flatten()
        if len(beta_vec) < 2 or abs(beta_vec[0]) < 1e-8:
            return False
 
        self.beta_lr = float(-beta_vec[1] / beta_vec[0])
        self.alpha   = result.alpha.flatten()[:2].astype(float)
 
        ec = p1_window - self.beta_lr * p2_window
        self.ec_mean = float(ec.mean())
        self.ec_std  = float(ec.std())
        if self.ec_std < 1e-8:
            return False
 
        self.fitted = True
        return True
 
    def get_z_score(self, p1_current: float, p2_current: float) -> float:
        if not self.fitted:
            return np.nan
        ec = p1_current - self.beta_lr * p2_current
        return (ec - self.ec_mean) / self.ec_std
 
    @property
    def half_life(self) -> float:
        # α_1 < 0 drives reversion in p1: half-life = -ln(2) / α_1
        alpha1 = self.alpha[0]
        if np.isnan(alpha1) or alpha1 >= 0:
            return np.inf
        return -np.log(2.0) / alpha1
 
class OUOrchestrator:
    """
    At each timestep t:
      1. window = spread[t-lookback : t]  (in-sample)
      2. Fit all four models on window
      3. Score spread[t] (out-of-sample) with each model
    No look-ahead: parameters fitted on [t-lookback, t-1], applied at t.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
 
    def _resolve_tickers(self, pair_str: str, available: set) -> tuple | None:
        for i, ch in enumerate(pair_str):
            if ch == "-":
                s1, s2 = pair_str[:i], pair_str[i+1:]
                if s1 in available and s2 in available:
                    return s1, s2
        return None
 
    def run_walk_forward(self, log_price_df: pd.DataFrame, pairs_metadata: pd.DataFrame) -> pd.DataFrame:
        all_signals = []
        available   = set(log_price_df.columns)
        eligible    = pairs_metadata[pairs_metadata["is_eligible"] == True]
 
        for _, meta in eligible.iterrows():
            tickers = self._resolve_tickers(str(meta["pair"]), available)
            if tickers is None:
                continue
            s1, s2 = tickers
            dates  = log_price_df.index
 
            spread, _ = kalman_spread(log_price_df[s1].values, log_price_df[s2].values,delta=0.01)
            p1_arr = log_price_df[s1].values
            p2_arr = log_price_df[s2].values
 
            # Instantiate one model object per pair — reused across timesteps
            ou_model   = OUModel()
            ou_garch   = OUGARCHModel()
            rs_ou      = RegimeSwitchingOU()
            vecm_model = VECMModel()
 
            pair_results = []
 
            for t in range(self.lookback, len(spread)):
                window    = spread[t - self.lookback : t]
                p1_window = p1_arr[t - self.lookback : t]
                p2_window = p2_arr[t - self.lookback : t]
 
                row = {"date": dates[t], "pair": meta["pair"]}
 
                # Baseline OU-AR(1)
                if ou_model.fit(window):
                    z = ou_model.get_z_score(spread[t])
                    if np.isfinite(z):
                        row["ou_z_score"]   = z
                        row["ou_kappa"]     = ou_model.kappa
                        row["ou_theta"]     = ou_model.theta
                        row["ou_half_life"] = ou_model.half_life
 
                # Baseline OU + GARCH(1,1)
                if ou_garch.fit(window):
                    z = ou_garch.get_z_score(spread[t])
                    if np.isfinite(z):
                        row["garch_z_score"]   = z
                        row["garch_half_life"] = ou_garch.half_life
 
                # Regime-switching OU
                if rs_ou.fit(window):
                    z, p_normal = rs_ou.get_z_score(spread[t])
                    if np.isfinite(z):
                        row["rs_z_score"]      = z
                        row["rs_p_normal"]     = p_normal
                        row["rs_kappa_normal"] = rs_ou.kappa[0]
                        row["rs_half_life"]    = rs_ou.half_life
 
                # VECM
                if vecm_model.fit(p1_window, p2_window):
                    z = vecm_model.get_z_score(p1_arr[t], p2_arr[t])
                    if np.isfinite(z):
                        row["vecm_z_score"]   = z
                        row["vecm_alpha1"]    = vecm_model.alpha[0]
                        row["vecm_alpha2"]    = vecm_model.alpha[1]
                        row["vecm_beta_lr"]   = vecm_model.beta_lr
                        row["vecm_half_life"] = vecm_model.half_life
 
                # Only emit if baseline OU produced a signal
                if "ou_z_score" in row:
                    pair_results.append(row)
 
            all_signals.extend(pair_results)
 
        return pd.DataFrame(all_signals)
 
 
def run_ou_model():
    raw_df   = pd.read_csv(DEFAULT_CONFIG.engineered_features_path, parse_dates=["Date"])
    pairs_df = pd.read_csv(DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv")
 
    prices_pivot  = raw_df.pivot(index="Date", columns="Ticker", values="Close")
    log_prices_df = np.log(prices_pivot)
    log_prices_df.index = pd.to_datetime(log_prices_df.index)
 
    print(f"\n{'='*50}")
    print(f"Starting OU Baseline Orchestration")
    print(f"Targeting: {pairs_df['is_eligible'].sum()} eligible pairs")
    print(f"Models: OU | OU+GARCH | Regime-switching OU | VECM")
    print(f"{'='*50}\n")
 
    orchestrator       = OUOrchestrator(lookback=60)
    all_window_results = []
 
    for start_date, end_date, label in all_training_windows(DEFAULT_CONFIG):
        print(f"Processing window: {label}")
 
        window_pairs = pairs_df[pairs_df["training_window"] == label]
        if window_pairs.empty:
            print(f"  No pairs for window {label}, skipping.")
            continue
 
        val_start  = pd.to_datetime(end_date)
        val_end    = val_start + pd.DateOffset(years=1)
        hist_start = val_start - pd.DateOffset(days=120)
 
        val_prices = log_prices_df.loc[hist_start : val_end]
        if val_prices.empty:
            print(f"  No price data for window {label}, skipping.")
            continue
 
        print(f"  Generating OOS signals ({val_start.date()} → {val_end.date()})...")
        signals     = orchestrator.run_walk_forward(val_prices, window_pairs)
        oos_signals = signals[signals["date"] >= val_start]
 
        if not oos_signals.empty:
            oos_signals = oos_signals.copy()
            oos_signals["training_window"] = label
            all_window_results.append(oos_signals)
            print(f"  {len(oos_signals)} signals across {oos_signals['pair'].nunique()} pairs.")
        else:
            print(f"  No OOS signals for window {label}.")
 
    if not all_window_results:
        print("\n[FAILED] No valid signals found for any training window.")
        return
 
    final_df = (pd.concat(all_window_results, ignore_index=True)
                .sort_values("date")
                .drop_duplicates(subset=["date", "pair"], keep="last"))
 
    output_path = DEFAULT_CONFIG.processed_dir / "ou_baseline_signals_extended.csv"
    final_df.to_csv(output_path, index=False)
 
    print(f"\n{'='*50}")
    print(f"Total OOS signals: {len(final_df)}")
    print(f"Pairs covered:     {final_df['pair'].nunique()}")
    print(f"Saved to:          {output_path}")
    print(f"{'='*50}")
 
 
if __name__ == "__main__":
    run_ou_model()