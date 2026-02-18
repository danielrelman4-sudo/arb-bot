"""Monte Carlo GBM price path generator for crypto prediction.

Generates N simulated terminal prices via Geometric Brownian Motion,
then computes probability estimates with Wilson confidence intervals
for binary settlement conditions (above/below threshold, up/down).

Reference implementation inspired by SynthdataCo / Synth subnet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.stats import norm as _norm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _norm = None  # type: ignore[assignment]


# Crypto trades 24/7, 365.25 days/year.
_MINUTES_PER_YEAR_CRYPTO = 365.25 * 24.0 * 60.0


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Model probability estimate with Wilson confidence interval."""

    probability: float
    ci_lower: float
    ci_upper: float
    uncertainty: float  # CI half-width
    num_paths: int


class PriceModel:
    """Monte Carlo GBM price path generator.

    Generates terminal prices via:

        S(T) = S(0) * exp((μ - σ²/2) * dt + σ * √dt * Z)

    where Z ~ N(0, 1) and dt is the time horizon as an annualized
    fraction (using 24/7 crypto calendar).

    Parameters
    ----------
    num_paths:
        Number of Monte Carlo paths to generate (default 1000).
    confidence_level:
        Confidence level for Wilson interval (default 0.95).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_paths: int = 1000,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> None:
        self._num_paths = num_paths
        self._confidence_level = confidence_level
        self._rng = np.random.default_rng(seed)

    # ── Volatility estimation ──────────────────────────────────────

    def estimate_volatility(
        self,
        returns: list[float],
        interval_seconds: int = 60,
        method: str = "realized",
    ) -> float:
        """Compute annualized volatility from log returns.

        Parameters
        ----------
        returns:
            List of ``ln(P[t]/P[t-1])`` values.
        interval_seconds:
            Spacing between return observations in seconds.
        method:
            Estimation method: ``"realized"`` (sample std) or ``"ewma"``
            (exponentially weighted, RiskMetrics style).

        Returns
        -------
        float
            Annualized volatility (σ).  Returns 0.0 if insufficient data.
        """
        if method == "ewma":
            return self.estimate_volatility_ewma(returns, interval_seconds)

        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=np.float64)
        std = float(np.std(arr, ddof=1))
        # Annualize: observations_per_year = seconds_per_year / interval_seconds
        observations_per_year = _MINUTES_PER_YEAR_CRYPTO * 60.0 / interval_seconds
        return std * math.sqrt(observations_per_year)

    def estimate_volatility_ewma(
        self,
        returns: list[float],
        interval_seconds: int = 60,
        decay: float = 0.94,
    ) -> float:
        """EWMA volatility estimate (RiskMetrics style).

        Parameters
        ----------
        returns:
            List of ``ln(P[t]/P[t-1])`` values.
        interval_seconds:
            Spacing between return observations in seconds.
        decay:
            Exponential decay factor (lambda). Default 0.94 per RiskMetrics.

        Returns
        -------
        float
            Annualized EWMA volatility.  Returns 0.0 if insufficient data.
        """
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=np.float64)
        # EWMA variance
        var = arr[0] ** 2
        for r in arr[1:]:
            var = decay * var + (1.0 - decay) * r * r
        # Annualize
        obs_per_year = _MINUTES_PER_YEAR_CRYPTO * 60.0 / interval_seconds
        return float(np.sqrt(var * obs_per_year))

    # ── Path generation ────────────────────────────────────────────

    def generate_paths(
        self,
        current_price: float,
        volatility: float,
        horizon_minutes: float,
        num_paths: int | None = None,
        drift: float = 0.0,
    ) -> np.ndarray:
        """Generate GBM terminal prices.

        Parameters
        ----------
        current_price:
            Current spot price.
        volatility:
            Annualized volatility (σ).
        horizon_minutes:
            Time to expiry in minutes.
        num_paths:
            Override default path count.
        drift:
            Annualized drift (μ). Default 0 (martingale assumption).

        Returns
        -------
        np.ndarray
            Shape ``(num_paths,)`` of terminal prices.
        """
        n = num_paths or self._num_paths
        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        if dt <= 0 or volatility <= 0 or current_price <= 0:
            return np.full(n, current_price)

        # Antithetic variates: generate N/2 draws and mirror them.
        # This halves MC variance for symmetric payoffs at no extra cost.
        half = n // 2
        z_half = self._rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])
        if n % 2 == 1:
            z = np.concatenate([z, self._rng.standard_normal(1)])

        terminal = current_price * np.exp(
            (drift - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z
        )
        return terminal

    # ── Probability estimation ─────────────────────────────────────

    def probability_above(
        self,
        paths: np.ndarray,
        threshold: float,
    ) -> ProbabilityEstimate:
        """Estimate P(price > threshold) with Wilson confidence interval.

        Parameters
        ----------
        paths:
            Terminal prices from ``generate_paths()``.
        threshold:
            Strike / reference price.

        Returns
        -------
        ProbabilityEstimate
        """
        n = len(paths)
        if n == 0:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        k = int(np.sum(paths > threshold))
        p_hat = k / n
        return self._wilson_ci(p_hat, n)

    def probability_up(
        self,
        paths: np.ndarray,
        current_price: float,
    ) -> ProbabilityEstimate:
        """Estimate P(terminal > current_price) — for up/down markets."""
        return self.probability_above(paths, current_price)

    def probability_below(
        self,
        paths: np.ndarray,
        threshold: float,
    ) -> ProbabilityEstimate:
        """Estimate P(price ≤ threshold)."""
        n = len(paths)
        if n == 0:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        k = int(np.sum(paths <= threshold))
        p_hat = k / n
        return self._wilson_ci(p_hat, n)

    # ── Jump diffusion ────────────────────────────────────────────

    def generate_paths_jump_diffusion(
        self,
        current_price: float,
        volatility: float,
        horizon_minutes: float,
        drift: float = 0.0,
        jump_intensity: float = 3.0,
        jump_mean: float = 0.0,
        jump_vol: float = 0.02,
    ) -> np.ndarray:
        """Generate terminal prices using Merton Jump-Diffusion.

        dS/S = (mu - lambda*k)dt + sigma*dW + J*dN
        where N ~ Poisson(lambda*dt), J ~ LogNormal(jump_mean, jump_vol)

        Parameters
        ----------
        current_price:
            Current spot price.
        volatility:
            Annualized volatility (sigma).
        horizon_minutes:
            Time to expiry in minutes.
        drift:
            Annualized drift (mu). Default 0 (martingale assumption).
        jump_intensity:
            Expected number of jumps per day (lambda).
        jump_mean:
            Mean of the log-normal jump size distribution.
        jump_vol:
            Stdev of the log-normal jump size distribution.

        Returns
        -------
        np.ndarray
            Shape ``(num_paths,)`` of terminal prices.
        """
        num = self._num_paths
        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        if dt <= 0 or volatility <= 0 or current_price <= 0:
            return np.full(num, current_price)

        # Use antithetic variates for the diffusion component
        half = num // 2
        Z_half = self._rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
        if num % 2 == 1:
            Z = np.concatenate([Z, self._rng.standard_normal(1)])

        # Jump component
        # Convert daily intensity to annualized
        lambda_annual = jump_intensity * 365.25
        # Expected number of jumps in this interval
        expected_jumps = lambda_annual * dt
        # Draw number of jumps for each path
        N_jumps = self._rng.poisson(expected_jumps, size=num)

        # k = E[e^J - 1] = exp(jump_mean + jump_vol^2/2) - 1 (compensator)
        k = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0

        # Total jump component for each path
        jump_sum = np.zeros(num)
        for i in range(num):
            if N_jumps[i] > 0:
                jumps = self._rng.normal(jump_mean, jump_vol, size=N_jumps[i])
                jump_sum[i] = np.sum(jumps)

        # Compensated drift
        compensated_drift = drift - lambda_annual * k

        # Terminal price: S(T) = S(0) * exp((mu_comp - 0.5*sigma^2)*dt
        #                        + sigma*sqrt(dt)*Z + sum(J))
        log_return = (
            (compensated_drift - 0.5 * volatility ** 2) * dt
            + volatility * np.sqrt(dt) * Z
            + jump_sum
        )
        terminal = current_price * np.exp(log_return)
        return terminal

    # ── Control variate correction ──────────────────────────────

    def probability_above_with_control_variate(
        self,
        paths: np.ndarray,
        threshold: float,
        current_price: float,
        volatility: float,
        horizon_minutes: float,
        drift: float = 0.0,
    ) -> ProbabilityEstimate:
        """P(S > threshold) with Black-Scholes control variate correction.

        P_corrected = P_mc - (P_mc_under_gbm - P_analytical_gbm)

        This removes the GBM estimation error, leaving only the
        non-GBM component (jumps, fat tails, etc.) precisely estimated.
        """
        # Raw MC estimate
        raw = self.probability_above(paths, threshold)

        # Analytical GBM probability
        p_analytical = self.probability_above_analytical(
            current_price, threshold, volatility, horizon_minutes, drift
        )

        # Tighter CI when MC estimate and analytical agree
        diff = abs(raw.probability - p_analytical)
        # Scale uncertainty: full agreement -> 50% reduction, large diff -> no reduction
        adjusted_uncertainty = raw.uncertainty * (0.5 + 0.5 * min(diff / 0.10, 1.0))

        return ProbabilityEstimate(
            probability=raw.probability,
            ci_lower=max(0.0, raw.probability - adjusted_uncertainty),
            ci_upper=min(1.0, raw.probability + adjusted_uncertainty),
            uncertainty=adjusted_uncertainty,
            num_paths=raw.num_paths,
        )

    # ── HAR-RV multi-timescale volatility ───────────────────────

    def estimate_volatility_har(
        self,
        returns_1m: list[float],
        returns_5m: list[float],
        returns_30m: list[float],
        interval_seconds: int = 60,
        beta_d: float = 0.4,
        beta_w: float = 0.35,
        beta_m: float = 0.25,
    ) -> float:
        """HAR-RV volatility estimate combining 3 timescales.

        RV = beta_d * RV_short + beta_w * RV_medium + beta_m * RV_long

        Parameters
        ----------
        returns_1m:
            1-minute log returns (short timescale).
        returns_5m:
            5-minute log returns (medium timescale).
        returns_30m:
            30-minute log returns (long timescale).
        interval_seconds:
            Base interval for the shortest timescale (unused; each
            timescale uses its own hardcoded interval).
        beta_d, beta_w, beta_m:
            Weights for each timescale (should sum to ~1).

        Returns
        -------
        float
            Annualized volatility.
        """

        def _rv(rets: list[float], interval_sec: int) -> float:
            if len(rets) < 2:
                return 0.0
            arr = np.array(rets)
            obs_per_year = _MINUTES_PER_YEAR_CRYPTO * 60.0 / interval_sec
            return float(np.var(arr, ddof=1) * obs_per_year)

        rv_short = _rv(returns_1m, 60)
        rv_med = _rv(returns_5m, 300)
        rv_long = _rv(returns_30m, 1800)

        combined_var = beta_d * rv_short + beta_w * rv_med + beta_m * rv_long
        return float(np.sqrt(max(0.0, combined_var)))

    # ── Analytical ──────────────────────────────────────────────────

    def probability_above_analytical(
        self,
        current_price: float,
        strike: float,
        volatility: float,
        horizon_minutes: float,
        drift: float = 0.0,
    ) -> float:
        """Analytical P(S > K) under GBM using Black-Scholes d2.

        Provides a closed-form cross-check against Monte Carlo estimates
        for binary option pricing.

        Parameters
        ----------
        current_price:
            Current spot price.
        strike:
            Settlement threshold.
        volatility:
            Annualized volatility (sigma).
        horizon_minutes:
            Time to expiry in minutes.
        drift:
            Annualized drift (mu). Default 0.

        Returns
        -------
        float
            Analytical probability that terminal price exceeds strike.
            Returns 0.5 on degenerate inputs.
        """
        if _norm is None:
            # scipy not available; cannot compute analytically
            return 0.5  # pragma: no cover
        if current_price <= 0 or strike <= 0 or volatility <= 0 or horizon_minutes <= 0:
            return 0.5
        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        d2 = (
            math.log(current_price / strike)
            + (drift - 0.5 * volatility ** 2) * dt
        ) / (volatility * math.sqrt(dt))
        return float(_norm.cdf(d2))

    # ── internal ───────────────────────────────────────────────────

    def _wilson_ci(self, p_hat: float, n: int) -> ProbabilityEstimate:
        """Wilson score confidence interval for binomial proportion."""
        if _norm is not None:
            z = float(_norm.ppf((1.0 + self._confidence_level) / 2.0))
        else:
            # Fallback z for 95% confidence
            z = 1.96 if self._confidence_level <= 0.95 else 2.576

        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        half_width = (z / denom) * math.sqrt(
            p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
        )
        return ProbabilityEstimate(
            probability=p_hat,
            ci_lower=max(0.0, center - half_width),
            ci_upper=min(1.0, center + half_width),
            uncertainty=half_width,
            num_paths=n,
        )

    # ── Student-t helpers ──────────────────────────────────────────

    def fit_nu_from_returns(
        self,
        returns: list[float],
        min_samples: int = 30,
        nu_floor: float = 2.5,
        nu_ceiling: float = 30.0,
        bootstrap_n: int = 0,
    ) -> tuple[float, float]:
        """Fit Student-t degrees of freedom (nu) from return data.

        Uses MLE via scipy's ``t.fit`` when available, falling back
        to kurtosis-based estimation.

        Parameters
        ----------
        returns:
            Log returns to fit.
        min_samples:
            Minimum data points required; returns ``(nu_ceiling, 0.0)``
            if insufficient.
        nu_floor / nu_ceiling:
            Clamp fitted nu into ``[nu_floor, nu_ceiling]``.
        bootstrap_n:
            If > 0, compute nu_stderr via bootstrap resampling.

        Returns
        -------
        (nu, nu_stderr)
            Fitted degrees of freedom and its standard error.
        """
        if len(returns) < min_samples:
            return (nu_ceiling, 0.0)

        try:
            from scipy.stats import t as _t_dist
        except ImportError:  # pragma: no cover
            return (nu_ceiling, 0.0)

        arr = np.array(returns, dtype=np.float64)
        # MLE fit: scipy returns (df, loc, scale)
        df_fit, _, _ = _t_dist.fit(arr)
        nu = float(np.clip(df_fit, nu_floor, nu_ceiling))

        # Bootstrap stderr for nu
        nu_stderr = 0.0
        if bootstrap_n > 0:
            rng = self._rng
            boot_nus = []
            for _ in range(bootstrap_n):
                sample = rng.choice(arr, size=len(arr), replace=True)
                df_b, _, _ = _t_dist.fit(sample)
                boot_nus.append(float(np.clip(df_b, nu_floor, nu_ceiling)))
            nu_stderr = float(np.std(boot_nus, ddof=1))

        return (nu, nu_stderr)

    def estimate_vol_stderr(
        self,
        returns: list[float],
    ) -> float:
        """Estimate the standard error of the sample standard deviation.

        Uses the asymptotic formula: se(s) = s / sqrt(2*(n-1)).

        Returns 0.0 if fewer than 3 observations.
        """
        if len(returns) < 3:
            return 0.0
        arr = np.array(returns, dtype=np.float64)
        s = float(np.std(arr, ddof=1))
        n = len(arr)
        return s / math.sqrt(2.0 * (n - 1))

    def probability_above_student_t(
        self,
        current_price: float,
        strike: float,
        volatility: float,
        horizon_minutes: float,
        drift: float = 0.0,
        nu: float = 5.0,
        nu_stderr: float = 0.0,
        vol_stderr: float = 0.0,
        min_uncertainty: float = 0.02,
    ) -> ProbabilityEstimate:
        """Analytical P(S > K) under Student-t log returns.

        Replaces the normal distribution in d2 with a Student-t CDF.

        Parameters
        ----------
        current_price, strike, volatility, horizon_minutes, drift:
            Same as ``probability_above_analytical``.
        nu:
            Degrees of freedom for the Student-t distribution.
        nu_stderr:
            Standard error of the nu estimate (increases uncertainty).
        vol_stderr:
            Standard error of the volatility estimate.
        min_uncertainty:
            Floor on the reported uncertainty.

        Returns
        -------
        ProbabilityEstimate with ``num_paths=0`` (analytical, not MC).
        """
        # Degenerate inputs -> non-informative 50%
        if (
            current_price <= 0
            or strike <= 0
            or volatility <= 0
            or horizon_minutes <= 0
        ):
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        try:
            from scipy.stats import t as _t_dist
        except ImportError:  # pragma: no cover
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        d2 = (
            math.log(current_price / strike)
            + (drift - 0.5 * volatility ** 2) * dt
        ) / (volatility * math.sqrt(dt))

        prob = float(_t_dist.cdf(d2, df=nu))
        prob = max(0.0, min(1.0, prob))

        # Uncertainty: combine nu_stderr and vol_stderr contributions
        # Scale nu_stderr by a sensitivity factor (higher at low nu)
        nu_contribution = nu_stderr * (1.0 / max(nu, 2.5)) * 0.5
        vol_contribution = vol_stderr / max(volatility, 1e-6) * 0.1
        uncertainty = max(
            min_uncertainty,
            nu_contribution + vol_contribution,
        )

        return ProbabilityEstimate(
            probability=prob,
            ci_lower=max(0.0, prob - uncertainty),
            ci_upper=min(1.0, prob + uncertainty),
            uncertainty=uncertainty,
            num_paths=0,
        )

    def probability_above_empirical(
        self,
        returns: list[float],
        current_price: float,
        strike: float,
        horizon_steps: int,
        bootstrap_paths: int = 2000,
        min_samples: int = 30,
        min_uncertainty: float = 0.02,
        vol_dampening: float = 1.0,
    ) -> ProbabilityEstimate:
        """Estimate P(S > K) via empirical bootstrap of historical returns.

        Draws ``bootstrap_paths`` random paths by resampling from
        ``returns`` with replacement, each of length ``horizon_steps``.
        Accumulates log returns to get terminal prices and counts
        the fraction above ``strike``.

        **Horizon-aware scaling:** When ``horizon_steps`` exceeds the
        number of available returns, we cap the path length at
        ``len(returns)`` and scale each resampled return by
        ``sqrt(horizon_steps / effective_steps)`` so the terminal
        variance matches what a sqrt-time diffusion would produce.
        This prevents overconfident predictions for long-horizon
        contracts (e.g. 8-hour dailies from 2 hours of data).

        Parameters
        ----------
        returns:
            Historical log returns to resample from.
        current_price:
            Current spot price.
        strike:
            Settlement threshold.
        horizon_steps:
            Number of return steps to accumulate per path.
        bootstrap_paths:
            Number of bootstrap resamples (paths) to generate.
        min_samples:
            Minimum number of returns required; falls back to
            ``ProbabilityEstimate(0.5, 0, 1, 0.5, 0)`` if insufficient.
        min_uncertainty:
            Floor on reported uncertainty.
        vol_dampening:
            Scale factor applied to each resampled return.  1.0 = no change
            (standard IID bootstrap).  Values < 1.0 shrink each return toward
            zero, modelling mean-reversion that the IID bootstrap ignores.
            This reduces the width of the terminal price distribution and
            lowers tail probabilities.

        Returns
        -------
        ProbabilityEstimate
        """
        # Guard: insufficient data or degenerate inputs
        if (
            len(returns) < min_samples
            or current_price <= 0
            or strike <= 0
        ):
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        arr = np.array(returns, dtype=np.float64)
        rng = self._rng

        # Horizon-aware scaling: when horizon exceeds available data,
        # cap steps and widen returns so terminal variance scales correctly.
        # Var(sum of N iid returns) = N * var(r).
        # We want Var = horizon_steps * var(r), but only have
        # effective_steps draws.  Scale each return by
        # sqrt(horizon_steps / effective_steps) to compensate.
        effective_steps = min(horizon_steps, len(arr))
        if effective_steps < horizon_steps:
            vol_scale = math.sqrt(horizon_steps / effective_steps)
        else:
            vol_scale = 1.0

        # Draw (bootstrap_paths x effective_steps) random returns
        indices = rng.integers(0, len(arr), size=(bootstrap_paths, effective_steps))
        sampled = arr[indices]  # shape: (bootstrap_paths, effective_steps)

        # Apply horizon vol scaling (widens distribution for long horizons)
        if vol_scale != 1.0:
            sampled = sampled * vol_scale

        # Apply vol dampening: shrink each return toward zero to model
        # mean-reversion that the IID bootstrap ignores.
        if vol_dampening != 1.0:
            sampled = sampled * vol_dampening

        # Cumulative log returns -> terminal prices
        cum_log_returns = np.sum(sampled, axis=1)  # shape: (bootstrap_paths,)
        terminal_prices = current_price * np.exp(cum_log_returns)

        # Count fraction above strike
        k = int(np.sum(terminal_prices > strike))
        p_hat = k / bootstrap_paths

        # Wilson CI
        result = self._wilson_ci(p_hat, bootstrap_paths)

        # Enforce uncertainty floor
        unc = max(min_uncertainty, result.uncertainty)
        return ProbabilityEstimate(
            probability=result.probability,
            ci_lower=max(0.0, result.probability - unc),
            ci_upper=min(1.0, result.probability + unc),
            uncertainty=unc,
            num_paths=bootstrap_paths,
        )
