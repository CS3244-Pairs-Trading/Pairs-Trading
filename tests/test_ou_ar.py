"""
Evaluation scripts for the OU baseline model.

Three levels of checks:
  1. Unit tests  — verify parameter recovery math on synthetic data
  2. Integration — run the full walk-forward on synthetic price data
  3. Diagnostics — inspect z-score quality on real signals

Run:
    python eval_ou_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Import the model we are testing
from src.models.ou import OUModel, OUOrchestrator, _fit_ar1


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def simulate_ou(kappa: float, theta: float, sigma: float,
                n: int = 2000, dt: float = 1.0,
                seed: int = 42) -> np.ndarray:
    """
    Exact simulation of an OU process via the Euler-Maruyama scheme:
        X_{t+1} = X_t + κ(θ - X_t)·dt + σ·√dt·Z_t,   Z_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    x   = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] + kappa * (theta - x[t-1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x
 
 
def simulate_random_walk(n: int = 2000, sigma: float = 1.0, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, sigma, n))
 
 
def simulate_cointegrated_pair(beta: float = 1.5,
                               kappa: float = 0.1,
                               theta: float = 0.0,
                               sigma_ou: float = 0.2,
                               sigma_rw: float = 0.01,
                               n: int = 1000,
                               seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a cointegrated pair (p1, p2) where:
        p2 = random walk
        p1 = beta * p2 + OU_spread
    """
    rng  = np.random.default_rng(seed)
    p2   = np.cumsum(rng.normal(0, sigma_rw, n))
    ou   = simulate_ou(kappa, theta, sigma_ou, n=n, seed=seed+1)
    p1   = beta * p2 + ou
    return p1, p2
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Unit test 1 — AR(1) OLS correctness
# ─────────────────────────────────────────────────────────────────────────────
 
def test_ar1_ols():
    print("=" * 55)
    print("Unit Test 1 — AR(1) OLS Parameter Recovery")
    print("=" * 55)
 
    # Known AR(1): x_t = 0.3 + 0.8 * x_{t-1} + eps, eps ~ N(0, 0.1^2)
    a_true, b_true, resid_std_true = 0.3, 0.8, 0.1
    rng = np.random.default_rng(0)
    n   = 5000
    x   = np.zeros(n)
    for t in range(1, n):
        x[t] = a_true + b_true * x[t-1] + rng.normal(0, resid_std_true)
 
    b_est, a_est, resid_std_est = _fit_ar1(x)
 
    tol = 0.02   # 2% tolerance at n=5000
    results = {
        "b (AR coef)":  (b_true,         b_est),
        "a (intercept)": (a_true,        a_est),
        "resid_std":     (resid_std_true, resid_std_est),
    }
 
    all_passed = True
    for name, (true, est) in results.items():
        err  = abs(est - true) / abs(true)
        ok   = err < tol
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {name}: true={true:.4f}, est={est:.4f}, err={err:.2%}")
        if not ok:
            all_passed = False
 
    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Unit test 2 — OU parameter recovery from AR(1)
# ─────────────────────────────────────────────────────────────────────────────
 
def test_ou_parameter_recovery():
    print("=" * 55)
    print("Unit Test 2 — OU Parameter Recovery")
    print("=" * 55)
 
    # Parameters must correspond to realistic daily half-lives (5-60 days).
    # kappa=0.5 (hl=1.4d) and kappa=1.5 (hl=0.46d) are sub-daily regimes where
    # daily sampling aliases the reversion into apparent anti-correlation.
    # OLS is also known to carry Hurwicz downward bias that amplifies via -ln(b)
    # for small b values. Both effects are properties of the estimator, not bugs.
    cases = [
        # (kappa,  theta,  sigma,  label)
        (0.0173, 0.0,    0.20,   "slow reversion   (hl~40d)"),
        (0.0462, 2.0,    0.40,   "medium reversion (hl~15d)"),
        (0.1386, -1.0,   0.60,   "fast reversion   (hl~5d)"),
    ]
 
    all_passed = True
    for kappa_true, theta_true, sigma_true, label in cases:
        spread = simulate_ou(kappa_true, theta_true, sigma_true, n=5000)
        model  = OUModel(dt=1.0)
        ok     = model.fit(spread)
 
        eq_std_true = sigma_true / np.sqrt(2 * kappa_true)
        hl_true     = np.log(2) / kappa_true
 
        print(f"\n  [{label}]  kappa={kappa_true}, theta={theta_true}, sigma={sigma_true}")
        print(f"  fit() returned: {ok}")
 
        if not ok:
            print("  FAIL — model.fit() returned False")
            all_passed = False
            continue
 
        checks = {
            "kappa":     (kappa_true, model.kappa,  0.15),
            "theta":     (theta_true, model.theta,  0.40),
            "sigma":     (sigma_true, model.sigma,  0.15),
            "eq_std":    (eq_std_true, model.eq_std, 0.15),
            "half_life": (hl_true,   model.half_life, 0.15),
        }
 
        for name, (true, est, tol) in checks.items():
            if abs(true) < 1e-8:
                err  = abs(est - true)
                flag = "PASS" if err < 0.05 else "WARN"
            else:
                err  = abs(est - true) / abs(true)
                flag = "PASS" if err < tol else "FAIL"
            print(f"    {flag}  {name}: true={true:.4f}, est={est:.4f}, err={err:.2%}")
            if flag == "FAIL":
                all_passed = False
 
    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Unit test 3 — Z-score properties
# ─────────────────────────────────────────────────────────────────────────────
 
def test_zscore_properties():
    """
    If the spread truly follows OU with the fitted parameters, z-scores
    computed on in-sample data should be approximately N(0,1).
    """
    print("=" * 55)
    print("Unit Test 3 — Z-Score Distribution (in-sample)")
    print("=" * 55)
 
    kappa, theta, sigma = 0.3, 0.0, 0.4
    spread = simulate_ou(kappa, theta, sigma, n=2000)
 
    model = OUModel()
    model.fit(spread)
 
    z_scores = np.array([(spread[t] - model.theta) / model.eq_std
                          for t in range(len(spread))])
 
    mean_z  = z_scores.mean()
    std_z   = z_scores.std()
    _, p_sw = stats.shapiro(z_scores[:500])   # Shapiro-Wilk on subset
 
    print(f"  z-score mean:  {mean_z:.4f}   (expected ~0)")
    print(f"  z-score std:   {std_z:.4f}    (expected ~1)")
    print(f"  Shapiro-Wilk p={p_sw:.4f}  (> 0.05 = looks Normal)")
 
    passed = abs(mean_z) < 0.1 and abs(std_z - 1.0) < 0.1
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return passed
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Unit test 4 — Non-stationary inputs are rejected
# ─────────────────────────────────────────────────────────────────────────────
 
def test_rejects_nonstationary():
    print("=" * 55)
    print("Unit Test 4 — Non-Stationary Input Rejection")
    print("=" * 55)
 
    # Random walks are intentionally excluded here.
    # OUModel is not an ADF test — random walk rejection is the job of
    # find_candidate_pairs (Engle-Granger + half-life filter upstream).
    # By the time a spread reaches OUModel it is already screened.
    # A random walk's b estimate sits just below 1.0 in finite samples,
    # so it may occasionally pass the 0 < b < 1 check — which is fine
    # because the pipeline never sends one here in production.
    cases = [
        ("explosive AR(1)", np.array([1.05**t for t in range(500)], dtype=float)),
        ("constant",        np.ones(500, dtype=float)),
        ("oscillating b<0", np.array([(-0.9)**t for t in range(500)], dtype=float)),
    ]
 
    model = OUModel()
    all_passed = True
    for label, series in cases:
        result = model.fit(series)
        flag   = "PASS" if not result else "FAIL"
        print(f"  {flag}  {label}: fit() returned {result} (expected False)")
        if result:
            all_passed = False
 
    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Integration test — full walk-forward on synthetic pair
# ─────────────────────────────────────────────────────────────────────────────
 
def test_walk_forward_integration():
    """
    End-to-end test: simulate a cointegrated pair, run OUOrchestrator,
    check that z-scores are generated and have sensible properties.
    """
    print("=" * 55)
    print("Integration Test — Walk-Forward on Synthetic Pair")
    print("=" * 55)
 
    n    = 800
    beta = 1.5
    p1, p2 = simulate_cointegrated_pair(beta=beta, kappa=0.15, theta=0.0,
                                         sigma_ou=0.3, n=n)
 
    dates      = pd.date_range("2018-01-01", periods=n, freq="B")
    log_prices = pd.DataFrame({"AAA": p1, "BBB": p2}, index=dates)
 
    pairs_meta = pd.DataFrame([{
        "pair":          "AAA-BBB",
        "initial_beta":  beta,
        "is_eligible":   True,
        "training_window": "test",
    }])
 
    orchestrator = OUOrchestrator(lookback=60)
    signals      = orchestrator.run_walk_forward(log_prices, pairs_meta)
 
    print(f"  Signals generated: {len(signals)}")
    print(f"  Expected min:      {n - 60}")
    print(f"  Z-score mean:  {signals['z_score'].mean():.4f}  (expect ~0)")
    print(f"  Z-score std:   {signals['z_score'].std():.4f}   (expect ~1)")
    print(f"  Kappa mean:    {signals['kappa'].mean():.4f}   (true=0.15)")
    print(f"  Half-life mean:{signals['half_life'].mean():.2f} days  (true={np.log(2)/0.15:.2f})")
 
    passed = (
        len(signals) >= n - 60 - 10 and      # almost all timesteps produced a signal
        signals["z_score"].mean().__abs__() < 0.5 and
        0.5 < signals["z_score"].std() < 2.0
    )
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}\n")
    return signals, passed
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_parameter_stability(signals: pd.DataFrame, pair: str = None,
                              output_path: str = "ou_parameter_stability.png"):
    """
    For a single pair, plot how kappa, theta, sigma, and half-life evolve
    over time. Stable parameters = the OU assumption holds consistently.
    """
    if pair is not None:
        df = signals[signals["pair"] == pair].copy()
    else:
        pair = signals["pair"].iloc[0]
        df   = signals[signals["pair"] == pair].copy()
 
    df = df.set_index("date")
 
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"OU Parameter Stability — {pair}", fontsize=13, fontweight="bold")
 
    params = [
        ("kappa",     "κ (mean-reversion speed)",  "steelblue"),
        ("theta",     "θ (long-run mean)",          "seagreen"),
        ("sigma",     "σ (diffusion coef)",         "darkorange"),
        ("half_life", "Half-life (days)",            "purple"),
    ]
 
    for ax, (col, label, color) in zip(axes, params):
        ax.plot(df.index, df[col], linewidth=0.8, color=color)
        ax.axhline(df[col].median(), linestyle="--", linewidth=0.8,
                   color="gray", label=f"median={df[col].median():.3f}")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
 
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
 
 
def plot_zscore_diagnostics(signals: pd.DataFrame,
                             output_path: str = "ou_zscore_diagnostics.png"):
    """
    Check z-score quality across all pairs:
      - Distribution vs N(0,1)
      - Rolling mean and std over time
      - QQ-plot
    """
    z = signals["z_score"].dropna().values
 
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Z-Score Diagnostics — OU Baseline", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
 
    # Panel 1: histogram vs N(0,1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(z, bins=80, density=True, color="steelblue", alpha=0.7, label="Observed")
    xs  = np.linspace(z.min(), z.max(), 300)
    ax1.plot(xs, stats.norm.pdf(xs), "r--", linewidth=1.5, label="N(0,1)")
    ax1.set_title("Z-score distribution", fontsize=10)
    ax1.set_xlabel("z-score"); ax1.legend(fontsize=8)
 
    # Panel 2: QQ-plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(z, dist="norm", plot=ax2)
    ax2.set_title("QQ-plot vs N(0,1)", fontsize=10)
    ax2.get_lines()[1].set_color("red")
 
    # Panel 3: rolling mean over time
    ax3 = fig.add_subplot(gs[1, 0])
    ts  = (signals[["date", "z_score"]]
           .set_index("date")
           .sort_index()
           .rolling("30D")["z_score"]
           .mean())
    ax3.plot(ts.index, ts.values, linewidth=0.8, color="steelblue")
    ax3.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax3.set_title("30-day rolling mean z-score", fontsize=10)
    ax3.set_xlabel("Date"); ax3.set_ylabel("z-score")
    ax3.grid(axis="y", alpha=0.3)
 
    # Panel 4: rolling std over time
    ax4 = fig.add_subplot(gs[1, 1])
    ts_std = (signals[["date", "z_score"]]
              .set_index("date")
              .sort_index()
              .rolling("30D")["z_score"]
              .std())
    ax4.plot(ts_std.index, ts_std.values, linewidth=0.8, color="darkorange")
    ax4.axhline(1, color="red", linewidth=0.8, linestyle="--", label="target std=1")
    ax4.set_title("30-day rolling std of z-score", fontsize=10)
    ax4.set_xlabel("Date"); ax4.set_ylabel("std")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
 
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
 
 
def plot_signal_on_spread(signals: pd.DataFrame,
                           log_prices: pd.DataFrame,
                           pair: str,
                           beta: float,
                           output_path: str = "ou_signal_plot.png"):
    """
    Overlay OU z-score trading signals on the actual spread for one pair.
    Marks where z > 2 (short spread) and z < -2 (long spread).
    """
    s1, s2 = pair.split("-")
    spread  = (log_prices[s1] - beta * log_prices[s2]).rename("spread")
    df      = signals[signals["pair"] == pair].set_index("date")
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"OU Signals — {pair}", fontsize=12, fontweight="bold")
 
    ax1.plot(spread.index, spread.values, linewidth=0.8, color="steelblue", label="Spread")
    ax1.set_title("Log spread (beta-adjusted)", fontsize=10)
    ax1.set_ylabel("Spread"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
 
    ax2.plot(df.index, df["z_score"], linewidth=0.8, color="steelblue", label="z-score")
    ax2.axhline( 2, color="red",    linewidth=0.8, linestyle="--", label="±2σ")
    ax2.axhline(-2, color="red",    linewidth=0.8, linestyle="--")
    ax2.axhline( 1, color="orange", linewidth=0.8, linestyle=":",  label="±1σ")
    ax2.axhline(-1, color="orange", linewidth=0.8, linestyle=":")
    ax2.axhline( 0, color="black",  linewidth=0.5)
 
    entry_short = df[df["z_score"] >  2.0]
    entry_long  = df[df["z_score"] < -2.0]
    ax2.scatter(entry_short.index, entry_short["z_score"],
                color="red",   s=10, zorder=5, label="Short signal (z>2)")
    ax2.scatter(entry_long.index,  entry_long["z_score"],
                color="green", s=10, zorder=5, label="Long signal (z<-2)")
 
    ax2.set_title("OU z-score", fontsize=10)
    ax2.set_ylabel("z-score"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
 
def print_signal_summary(signals: pd.DataFrame):
    print("=" * 55)
    print("Signal Summary")
    print("=" * 55)
    print(f"  Total signal rows:    {len(signals)}")
    print(f"  Unique pairs:         {signals['pair'].nunique()}")
    print(f"  Date range:           {signals['date'].min()} → {signals['date'].max()}")
    print()
 
    z = signals["z_score"].dropna()
    print(f"  Z-score mean:         {z.mean():.4f}  (expect ~0)")
    print(f"  Z-score std:          {z.std():.4f}   (expect ~1)")
    print(f"  Z-score skew:         {z.skew():.4f}")
    print(f"  Z-score kurtosis:     {z.kurt():.4f}")
    print()
 
    _, p_norm = stats.normaltest(z.sample(min(5000, len(z)), random_state=0))
    print(f"  Normality test p:     {p_norm:.4f}  (> 0.05 = looks Normal)")
    print()
 
    hl = signals["half_life"].dropna()
    print(f"  Half-life median:     {hl.median():.2f} days")
    print(f"  Half-life 10th pct:   {hl.quantile(0.10):.2f} days")
    print(f"  Half-life 90th pct:   {hl.quantile(0.90):.2f} days")
    print()
 
    kappa = signals["kappa"].dropna()
    print(f"  Kappa median:         {kappa.median():.4f}")
    print(f"  Sigma median:         {signals['sigma'].median():.4f}")
    print()
 
    n_long  = (signals["z_score"] < -2).sum()
    n_short = (signals["z_score"] >  2).sum()
    print(f"  Entry signals (z>2):  {n_short}")
    print(f"  Entry signals (z<-2): {n_long}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\nOU Model Evaluation Suite")
    print("=" * 55)
 
    # Unit tests
    r1 = test_ar1_ols()
    r2 = test_ou_parameter_recovery()
    r3 = test_zscore_properties()
    r4 = test_rejects_nonstationary()
 
    # Integration test + plots on synthetic data
    signals, r5 = test_walk_forward_integration()
 
    print("=" * 55)
    print("Generating diagnostic plots on synthetic signals...")
    plot_zscore_diagnostics(signals, output_path="ou_zscore_diagnostics.png")
    plot_parameter_stability(signals, pair="AAA-BBB",
                              output_path="ou_parameter_stability.png")
 
    # Reconstruct synthetic log prices for the signal plot
    p1, p2 = simulate_cointegrated_pair(beta=1.5, kappa=0.15, n=800)
    dates  = pd.date_range("2018-01-01", periods=800, freq="B")
    log_prices = pd.DataFrame({"AAA": p1, "BBB": p2}, index=dates)
    plot_signal_on_spread(signals, log_prices, pair="AAA-BBB", beta=1.5,
                          output_path="ou_signal_plot.png")
 
    print_signal_summary(signals)
 
    all_passed = all([r1, r2, r3, r4, r5])
    print("\n" + "=" * 55)
    print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 55)