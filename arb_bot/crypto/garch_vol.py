"""GARCH(1,1) volatility forecaster for crypto returns.

Fits GARCH(1,1) parameters via maximum likelihood estimation on
1-minute log-returns, produces multi-step-ahead volatility forecasts
for binary option pricing.

The conditional variance equation:
    σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

where ω > 0, α ≥ 0, β ≥ 0, and α + β < 1 (stationarity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _scipy_minimize = None  # type: ignore[assignment]


# Crypto trades 24/7.
_MINUTES_PER_YEAR_CRYPTO = 365.25 * 24.0 * 60.0


@dataclass(frozen=True)
class GarchParams:
    """Fitted GARCH(1,1) parameters."""

    omega: float       # Long-run variance intercept
    alpha: float       # Shock coefficient (weight on ε²(t-1))
    beta: float        # Persistence coefficient (weight on σ²(t-1))
    long_run_var: float  # ω / (1 - α - β)
    log_likelihood: float
    n_obs: int
    converged: bool


@dataclass(frozen=True)
class GarchForecast:
    """GARCH volatility forecast at a specific horizon."""

    sigma_annualized: float    # Annualized vol forecast
    sigma_horizon: float       # Vol over the forecast horizon (not annualized)
    sigma_1step: float         # Current 1-step-ahead conditional vol
    forecast_steps: int        # Number of steps ahead
    params: GarchParams        # Underlying parameters


class GarchForecaster:
    """GARCH(1,1) volatility forecaster.

    Parameters
    ----------
    min_obs:
        Minimum number of observations to fit (default 120 = 2 hours
        of 1-minute data).
    refit_interval:
        Number of new observations before re-estimating parameters.
    lookback_obs:
        Maximum number of observations to use for fitting
        (default 1440 = 24 hours of 1-minute data).
    interval_seconds:
        Interval between observations in seconds (default 60 = 1 min).
    """

    def __init__(
        self,
        min_obs: int = 120,
        refit_interval: int = 60,
        lookback_obs: int = 1440,
        interval_seconds: int = 60,
    ) -> None:
        self._min_obs = min_obs
        self._refit_interval = refit_interval
        self._lookback_obs = lookback_obs
        self._interval_seconds = interval_seconds

        # Annualization factor for per-minute returns.
        self._obs_per_year = _MINUTES_PER_YEAR_CRYPTO * (60.0 / interval_seconds)

        # Cache last fit.
        self._last_params: Optional[GarchParams] = None
        self._last_sigma2: Optional[float] = None  # Last conditional variance
        self._obs_since_fit: int = 0

    # ── Public API ────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> Optional[GarchParams]:
        """Estimate GARCH(1,1) parameters via MLE.

        Parameters
        ----------
        returns:
            Array of log-returns (not annualized).

        Returns
        -------
        GarchParams or None if fitting fails or insufficient data.
        """
        if len(returns) < self._min_obs:
            return None

        # Trim to lookback window.
        r = np.asarray(returns[-self._lookback_obs:], dtype=np.float64)
        n = len(r)

        sample_var = float(np.var(r, ddof=1))
        if sample_var < 1e-20:
            return None  # Constant series

        # Try MLE first, then grid search fallback.
        params = self._fit_mle(r, sample_var)
        if params is None:
            params = self._fit_grid(r, sample_var)

        if params is not None:
            self._last_params = params
            # Initialize conditional variance from the full series.
            sigma2_series = self._compute_sigma2_series(params, r)
            self._last_sigma2 = float(sigma2_series[-1])
            self._obs_since_fit = 0

        return params

    def forecast(
        self,
        params: GarchParams,
        returns: np.ndarray,
        horizon_steps: int,
    ) -> GarchForecast:
        """Multi-step-ahead GARCH volatility forecast.

        Uses the closed-form multi-step GARCH(1,1) forecast formula:

            σ²(t+h) = σ²_∞ + (α + β)^(h-1) · (σ²(t+1) - σ²_∞)

        and total integrated variance over the horizon:

            V_h = h·σ²_∞ + (σ²_1 - σ²_∞) · (1 - (α+β)^h) / (1 - (α+β))

        Parameters
        ----------
        params:
            Fitted GARCH parameters.
        returns:
            Array of log-returns to initialize the conditional variance.
        horizon_steps:
            Number of periods to forecast ahead.

        Returns
        -------
        GarchForecast with annualized and horizon-specific vol estimates.
        """
        r = np.asarray(returns, dtype=np.float64)

        # Get current conditional variance.
        sigma2_series = self._compute_sigma2_series(params, r)
        sigma2_current = float(sigma2_series[-1])

        # 1-step-ahead conditional variance.
        last_return = float(r[-1])
        sigma2_1 = params.omega + params.alpha * last_return ** 2 + params.beta * sigma2_current

        persistence = params.alpha + params.beta
        long_run_var = params.long_run_var

        # Total integrated variance over the horizon.
        if abs(1.0 - persistence) < 1e-10:
            # Near unit root: approximate as constant.
            total_var = horizon_steps * sigma2_1
        else:
            # Closed-form sum of multi-step variances.
            total_var = (
                horizon_steps * long_run_var
                + (sigma2_1 - long_run_var)
                * (1.0 - persistence ** horizon_steps)
                / (1.0 - persistence)
            )

        # Ensure non-negative.
        total_var = max(total_var, 1e-20)

        sigma_horizon = math.sqrt(total_var)
        sigma_1step = math.sqrt(max(sigma2_1, 1e-20))

        # Annualize: total_var is in per-minute^2 units over horizon_steps minutes.
        sigma_annualized = sigma_horizon * math.sqrt(self._obs_per_year / horizon_steps)

        return GarchForecast(
            sigma_annualized=sigma_annualized,
            sigma_horizon=sigma_horizon,
            sigma_1step=sigma_1step,
            forecast_steps=horizon_steps,
            params=params,
        )

    def update_online(
        self,
        new_return: float,
        current_sigma2: float,
    ) -> float:
        """Online GARCH variance update without re-estimation.

        σ²(t+1) = ω + α·r²(t) + β·σ²(t)

        Used between full re-estimations for low-latency updates.
        """
        if self._last_params is None:
            return current_sigma2

        p = self._last_params
        new_sigma2 = p.omega + p.alpha * new_return ** 2 + p.beta * current_sigma2
        self._last_sigma2 = new_sigma2
        self._obs_since_fit += 1
        return new_sigma2

    @property
    def needs_refit(self) -> bool:
        """Whether the model should be re-estimated."""
        return self._obs_since_fit >= self._refit_interval

    @property
    def last_params(self) -> Optional[GarchParams]:
        return self._last_params

    @property
    def last_sigma2(self) -> Optional[float]:
        return self._last_sigma2

    # ── Internal: MLE fitting ─────────────────────────────────────

    def _fit_mle(
        self,
        returns: np.ndarray,
        sample_var: float,
    ) -> Optional[GarchParams]:
        """Fit via scipy.optimize.minimize (L-BFGS-B)."""
        if _scipy_minimize is None:
            return None  # pragma: no cover

        n = len(returns)
        r2 = returns ** 2

        # Initial guess via variance targeting.
        alpha_0, beta_0 = 0.05, 0.90
        omega_0 = sample_var * (1.0 - alpha_0 - beta_0)

        def neg_log_likelihood(params_vec: np.ndarray) -> float:
            omega, alpha, beta = params_vec

            # Stationarity penalty.
            if alpha + beta >= 0.999:
                return 1e10

            sigma2 = np.empty(n)
            sigma2[0] = omega / max(1.0 - alpha - beta, 0.001)

            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
                if sigma2[t] < 1e-20:
                    sigma2[t] = 1e-20

            # Log-likelihood (dropping constant).
            ll = -0.5 * np.sum(np.log(sigma2) + r2 / sigma2)

            if not np.isfinite(ll):
                return 1e10

            return -ll  # Minimize negative LL.

        try:
            result = _scipy_minimize(
                neg_log_likelihood,
                x0=np.array([omega_0, alpha_0, beta_0]),
                method="L-BFGS-B",
                bounds=[
                    (1e-12, 10.0 * sample_var),  # omega
                    (0.01, 0.50),                  # alpha
                    (0.50, 0.98),                  # beta
                ],
                options={"maxiter": 200, "ftol": 1e-10},
            )
        except Exception:
            return None

        if not result.success and result.fun > 1e9:
            return None

        omega, alpha, beta = result.x
        persistence = alpha + beta
        if persistence >= 0.999:
            return None

        long_run_var = omega / (1.0 - persistence)

        return GarchParams(
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            long_run_var=float(long_run_var),
            log_likelihood=float(-result.fun),
            n_obs=n,
            converged=result.success,
        )

    def _fit_grid(
        self,
        returns: np.ndarray,
        sample_var: float,
    ) -> Optional[GarchParams]:
        """Fallback grid search over alpha/beta when MLE fails."""
        n = len(returns)
        r2 = returns ** 2

        alphas = [0.03, 0.05, 0.08, 0.10, 0.15]
        betas = [0.80, 0.85, 0.88, 0.90, 0.93, 0.95]

        best_ll = -np.inf
        best_params: Optional[GarchParams] = None

        for alpha in alphas:
            for beta in betas:
                persistence = alpha + beta
                if persistence >= 0.999:
                    continue

                omega = sample_var * (1.0 - persistence)
                if omega <= 0:
                    continue

                sigma2 = np.empty(n)
                sigma2[0] = sample_var

                for t in range(1, n):
                    sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
                    if sigma2[t] < 1e-20:
                        sigma2[t] = 1e-20

                ll = -0.5 * float(np.sum(np.log(sigma2) + r2 / sigma2))

                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
                    best_params = GarchParams(
                        omega=omega,
                        alpha=alpha,
                        beta=beta,
                        long_run_var=omega / (1.0 - persistence),
                        log_likelihood=ll,
                        n_obs=n,
                        converged=False,
                    )

        return best_params

    # ── Internal: conditional variance series ─────────────────────

    def _compute_sigma2_series(
        self,
        params: GarchParams,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute full conditional variance series."""
        n = len(returns)
        r2 = returns ** 2

        sigma2 = np.empty(n)
        sigma2[0] = params.long_run_var

        for t in range(1, n):
            sigma2[t] = params.omega + params.alpha * r2[t - 1] + params.beta * sigma2[t - 1]
            if sigma2[t] < 1e-20:
                sigma2[t] = 1e-20

        return sigma2
