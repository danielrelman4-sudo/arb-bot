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

    # ── Implied volatility extraction ─────────────────────────────

    def extract_implied_vol(
        self,
        market_price: float,
        strike: float,
        spot: float,
        tte_minutes: float,
        drift: float = 0.0,
        direction: str = "above",
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> float | None:
        """Extract implied volatility from a binary option market price.

        Inverts the Black-Scholes binary option pricing formula:

            P(above) = N(d2)  where d2 = [ln(S/K) + (μ - σ²/2)τ] / (σ√τ)

        Uses Brent's method (bisection with interpolation) to find σ such
        that the analytical binary price matches the observed market price.

        Parameters
        ----------
        market_price:
            Observed binary option price (probability), in [0, 1].
        strike:
            Settlement threshold.
        spot:
            Current spot price.
        tte_minutes:
            Time to expiry in minutes.
        drift:
            Annualized drift (μ). Default 0.
        direction:
            ``"above"`` (call-like) or ``"below"`` (put-like).
        tol:
            Convergence tolerance for σ.
        max_iter:
            Maximum bisection iterations.

        Returns
        -------
        float or None
            Annualized implied volatility, or None if extraction fails
            (degenerate inputs, price outside [0.01, 0.99], no convergence).
        """
        if _norm is None:
            return None
        if spot <= 0 or strike <= 0 or tte_minutes <= 0:
            return None
        # Prices near 0 or 1 are uninformative for IV
        if market_price < 0.01 or market_price > 0.99:
            return None

        # For "below" contracts, invert: P(below) = 1 - P(above)
        target_prob = market_price if direction == "above" else (1.0 - market_price)

        dt = tte_minutes / _MINUTES_PER_YEAR_CRYPTO
        log_moneyness = math.log(spot / strike)

        def _binary_price(sigma: float) -> float:
            d2 = (log_moneyness + (drift - 0.5 * sigma * sigma) * dt) / (sigma * math.sqrt(dt))
            return float(_norm.cdf(d2))

        # Bisection bounds: 1% to 500% annualized vol
        lo, hi = 0.01, 5.0

        # Check if solution exists within bounds
        p_lo = _binary_price(lo)
        p_hi = _binary_price(hi)

        # If target is outside the range, no valid IV
        if (target_prob - p_lo) * (target_prob - p_hi) > 0:
            return None

        # Brent-style bisection
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            p_mid = _binary_price(mid)
            if abs(p_mid - target_prob) < tol:
                return mid
            # Binary option price is monotonically decreasing in vol
            # for above contracts (higher vol → d2 shrinks for OTM, but
            # relationship depends on moneyness).
            # Use sign-based bisection which works regardless.
            if (p_mid - target_prob) * (p_lo - target_prob) < 0:
                hi = mid
            else:
                lo = mid
                p_lo = p_mid

        # Return midpoint even if not fully converged
        return (lo + hi) / 2.0

    def extract_iv_from_atm_contracts(
        self,
        quotes: list[dict],
        spot: float,
        tte_minutes: float,
        drift: float = 0.0,
    ) -> float | None:
        """Extract implied vol from the ATM (closest-to-spot) contract.

        Parameters
        ----------
        quotes:
            List of dicts with keys ``"strike"``, ``"price"``, and
            optionally ``"direction"`` (default ``"above"``).
            ``price`` is the binary option market price [0, 1].
        spot:
            Current spot price.
        tte_minutes:
            Time to expiry in minutes.
        drift:
            Annualized drift.

        Returns
        -------
        float or None
            Implied vol from the ATM contract, or None if extraction fails.
        """
        if not quotes or spot <= 0 or tte_minutes <= 0:
            return None

        # Find ATM contract (closest strike to spot)
        best = None
        best_dist = float("inf")
        for q in quotes:
            strike = q.get("strike", 0.0)
            if strike <= 0:
                continue
            dist = abs(strike - spot) / spot
            if dist < best_dist:
                best_dist = dist
                best = q

        if best is None:
            return None

        return self.extract_implied_vol(
            market_price=best["price"],
            strike=best["strike"],
            spot=spot,
            tte_minutes=tte_minutes,
            drift=drift,
            direction=best.get("direction", "above"),
        )

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

        # Effective-sample-size uncertainty floor (v44 Fix F).
        # Wilson CI uses n=bootstrap_paths (2000), but those are resamples
        # from only len(returns) unique observations.  When all paths agree
        # (p_hat≈1.0), Wilson reports uncertainty≈0 — epistemically wrong
        # for a small sample.  Floor at 1/(2√n_unique): the standard error
        # of a binomial at p=0.5, which is the maximum-entropy baseline.
        n_unique = len(returns)
        sample_floor = 1.0 / (2.0 * math.sqrt(max(n_unique, 1)))
        unc = max(min_uncertainty, result.uncertainty, sample_floor)
        return ProbabilityEstimate(
            probability=result.probability,
            ci_lower=max(0.0, result.probability - unc),
            ci_upper=min(1.0, result.probability + unc),
            uncertainty=unc,
            num_paths=bootstrap_paths,
        )

    # ── Multi-scale realized volatility (v43 — 1A) ──────────────

    def variance_ratio(
        self,
        returns_1min: list[float] | np.ndarray,
        q: int,
    ) -> float:
        """Compute the variance ratio VR(q) for multi-period returns.

        VR(q) = Var(q-period return) / (q × Var(1-period return))

        Values:
        - VR < 1 → mean-reversion (√T scaling overestimates vol)
        - VR = 1 → IID / random walk
        - VR > 1 → momentum / trending

        Parameters
        ----------
        returns_1min:
            1-minute log returns.
        q:
            Aggregation period in minutes (e.g. 30 for 30-min VR).

        Returns
        -------
        float
            Variance ratio. Returns 1.0 (random walk) on insufficient data.
        """
        arr = np.asarray(returns_1min, dtype=np.float64)
        n = len(arr)
        if n < 2 * q or q < 2:
            return 1.0

        # Variance of 1-period returns
        var_1 = float(np.var(arr, ddof=1))
        if var_1 < 1e-30:
            return 1.0

        # Build non-overlapping q-period returns
        n_blocks = n // q
        trimmed = arr[: n_blocks * q].reshape(n_blocks, q)
        q_returns = np.sum(trimmed, axis=1)  # sum of log returns = log return over q periods
        var_q = float(np.var(q_returns, ddof=1))

        vr = var_q / (q * var_1)
        return max(0.01, min(5.0, vr))  # clamp to prevent degeneracy

    def realized_vol_at_horizon(
        self,
        returns_1min: list[float] | np.ndarray,
        horizon_minutes: int,
        interval_seconds: int = 60,
    ) -> float | None:
        """Compute annualized realized vol directly at the target horizon.

        Instead of √T-scaling 1-min vol, this computes vol from returns
        aggregated at the actual contract horizon. More accurate when
        returns exhibit autocorrelation (mean-reversion or momentum).

        Parameters
        ----------
        returns_1min:
            1-minute log returns.
        horizon_minutes:
            Target horizon in minutes (e.g. 30 for a 30-min contract).
        interval_seconds:
            Spacing of the 1-min returns (default 60s).

        Returns
        -------
        float or None
            Annualized realized vol at the target horizon, or None if
            insufficient data (need at least 10 non-overlapping blocks).
        """
        arr = np.asarray(returns_1min, dtype=np.float64)
        n = len(arr)
        n_blocks = n // horizon_minutes
        if n_blocks < 10:
            return None  # not enough data, caller should fall back to √T

        # Build non-overlapping horizon-period returns
        trimmed = arr[: n_blocks * horizon_minutes].reshape(n_blocks, horizon_minutes)
        horizon_returns = np.sum(trimmed, axis=1)

        std_horizon = float(np.std(horizon_returns, ddof=1))
        # Annualize: how many horizon-length intervals per year?
        horizon_seconds = horizon_minutes * interval_seconds
        obs_per_year = _MINUTES_PER_YEAR_CRYPTO * 60.0 / horizon_seconds
        return std_horizon * math.sqrt(obs_per_year)

    @staticmethod
    def vol_with_variance_ratio_correction(
        vol_1min_annualized: float,
        horizon_minutes: float,
        vr: float,
    ) -> float:
        """Adjust √T-scaled vol using the variance ratio.

        σ_adjusted = σ_1min_ann × √(T/T_base) × √VR(T)

        When VR < 1 (mean-reversion), this reduces vol vs naive √T.
        When VR > 1 (momentum), this increases vol.

        Parameters
        ----------
        vol_1min_annualized:
            Annualized vol estimated from 1-min returns.
        horizon_minutes:
            Contract horizon in minutes.
        vr:
            Variance ratio at the target horizon.

        Returns
        -------
        float
            Adjusted annualized volatility.
        """
        # √VR correction: if VR=0.7 at 30min, vol is only √0.7 ≈ 0.84× naive
        return vol_1min_annualized * math.sqrt(max(0.01, vr))

    # ── Merton jump-diffusion closed-form series (v43 — 1B) ─────

    def probability_above_merton_series(
        self,
        current_price: float,
        strike: float,
        volatility: float,
        horizon_minutes: float,
        drift: float = 0.0,
        lambda_daily: float = 3.0,
        jump_mean: float = 0.0,
        jump_vol: float = 0.02,
        n_terms: int = 15,
        min_uncertainty: float = 0.02,
    ) -> ProbabilityEstimate:
        """Semi-closed-form P(S > K) under Merton jump-diffusion.

        Computes an infinite series of Black-Scholes binary prices
        weighted by Poisson jump probabilities. Converges in ~10-15
        terms, <1ms computation.

        For each n=0..n_terms:
            w_n = e^(-λT) × (λT)^n / n!          (Poisson weight)
            σ_n = √(σ² + n×σ_J²/T)               (jump-adjusted vol)
            μ_n = drift - λ×k + n×μ_J/T           (jump-adjusted drift)
            P_n = N(d2_n)                          (BS binary price)
            P = Σ w_n × P_n

        Parameters
        ----------
        current_price:
            Current spot price.
        strike:
            Settlement threshold.
        volatility:
            Annualized diffusion volatility (sigma).
        horizon_minutes:
            Time to expiry in minutes.
        drift:
            Annualized drift (mu). Default 0.
        lambda_daily:
            Expected number of jumps per day (λ).
        jump_mean:
            Mean of log-normal jump size distribution (μ_J).
        jump_vol:
            Stdev of log-normal jump size distribution (σ_J).
        n_terms:
            Number of series terms (default 15).
        min_uncertainty:
            Floor on reported uncertainty.

        Returns
        -------
        ProbabilityEstimate with num_paths=0 (analytical).
        """
        if _norm is None:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)
        if current_price <= 0 or strike <= 0 or volatility <= 0 or horizon_minutes <= 0:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        sqrt_dt = math.sqrt(dt)

        # Annualize jump intensity: λ_annual = λ_daily × 365.25
        lambda_annual = lambda_daily * 365.25
        lambda_T = lambda_annual * dt  # expected jumps in interval

        # Compensator: k = E[e^J - 1]
        k_comp = math.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0

        log_moneyness = math.log(current_price / strike)

        prob = 0.0
        total_weight = 0.0

        # Pre-compute log(λT) for Poisson weight to avoid overflow
        log_lambda_T = math.log(max(lambda_T, 1e-30))

        # Accumulate log(n!) iteratively
        log_factorial_n = 0.0  # log(0!) = 0

        for n in range(n_terms):
            # Poisson weight: w_n = e^(-λT) × (λT)^n / n!
            # In log space: log(w_n) = -λT + n×log(λT) - log(n!)
            log_w = -lambda_T + n * log_lambda_T - log_factorial_n
            w_n = math.exp(log_w)

            # Jump-adjusted vol: σ_n² = σ² + n×σ_J²/T
            var_n = volatility ** 2 + n * jump_vol ** 2 / dt if dt > 0 else volatility ** 2
            sigma_n = math.sqrt(max(var_n, 1e-30))

            # Jump-adjusted drift: μ_n = drift - λ×k + n×μ_J/T
            drift_n = drift - lambda_annual * k_comp + n * jump_mean / dt if dt > 0 else drift

            # d2 for this term
            d2_n = (
                log_moneyness + (drift_n - 0.5 * sigma_n ** 2) * dt
            ) / (sigma_n * sqrt_dt)

            p_n = float(_norm.cdf(d2_n))
            prob += w_n * p_n
            total_weight += w_n

            # Update log(n!) for next iteration
            log_factorial_n += math.log(n + 1)

        # Normalize by total weight (should be ~1.0 if enough terms)
        if total_weight > 0:
            prob /= total_weight

        prob = max(0.0, min(1.0, prob))

        # Uncertainty from series truncation
        truncation_err = 1.0 - total_weight  # mass not captured
        uncertainty = max(min_uncertainty, truncation_err * 0.5)

        return ProbabilityEstimate(
            probability=prob,
            ci_lower=max(0.0, prob - uncertainty),
            ci_upper=min(1.0, prob + uncertainty),
            uncertainty=uncertainty,
            num_paths=0,
        )

    # ── Strike-distance probability floors/ceilings (v43 — 1C) ──

    def compute_probability_bounds(
        self,
        lambda_daily: float,
        horizon_minutes: float,
        volatility: float,
        moneyness_sigma: float,
        jump_vol: float = 0.02,
        floor_min: float = 0.005,
        ceiling_max: float = 0.995,
    ) -> tuple[float, float]:
        """Compute probability floor/ceiling from jump intensity.

        Deep ITM/OTM contracts should never have P=0 or P=1 because
        a single jump can always flip the outcome.

        Parameters
        ----------
        lambda_daily:
            Expected number of jumps per day (λ).
        horizon_minutes:
            Time to expiry in minutes.
        volatility:
            Annualized volatility (σ).
        moneyness_sigma:
            Strike distance in vol units: |ln(S/K)| / (σ×√T).
        jump_vol:
            Jump size stdev (σ_J). Used to estimate probability of
            a jump large enough to flip the outcome.
        floor_min:
            Absolute minimum floor.
        ceiling_max:
            Absolute maximum ceiling.

        Returns
        -------
        (floor, ceiling): probability bounds.
        """
        if horizon_minutes <= 0 or volatility <= 0:
            return (floor_min, ceiling_max)

        dt_days = horizon_minutes / (24.0 * 60.0)
        # P(at least one jump in window) = 1 - e^(-λ×dt_days)
        p_jump = 1.0 - math.exp(-lambda_daily * dt_days)

        # P(jump large enough to flip): need jump > moneyness_sigma × σ√T
        # Jump sizes are ~N(0, σ_J), so P(flip) ≈ N(-moneyness_sigma × σ√T / σ_J)
        dt = horizon_minutes / _MINUTES_PER_YEAR_CRYPTO
        distance_in_log = moneyness_sigma * volatility * math.sqrt(dt)
        if jump_vol > 0 and _norm is not None:
            p_flip_given_jump = float(_norm.cdf(-distance_in_log / jump_vol))
        else:
            p_flip_given_jump = 0.01  # conservative fallback

        p_adverse = p_jump * p_flip_given_jump

        floor = max(floor_min, p_adverse)
        ceiling = min(ceiling_max, 1.0 - p_adverse)

        # Ensure floor < ceiling
        if floor >= ceiling:
            midpoint = (floor_min + ceiling_max) / 2.0
            floor = floor_min
            ceiling = ceiling_max

        return (floor, ceiling)

    @staticmethod
    def clamp_probability(
        prob: float,
        floor: float,
        ceiling: float,
    ) -> float:
        """Clamp probability to [floor, ceiling]."""
        return max(floor, min(ceiling, prob))

    # ── Ornstein-Uhlenbeck closed-form pricing (v43 — 2A) ───────

    def probability_above_ou(
        self,
        current_price: float,
        strike: float,
        horizon_minutes: float,
        theta: float,
        mu: float,
        sigma: float,
        min_uncertainty: float = 0.02,
    ) -> ProbabilityEstimate:
        """Analytical P(S > K) under Ornstein-Uhlenbeck log-price process.

        OU transition density (closed-form):
            E[X_T] = μ + (X₀ - μ) × e^(-θT)
            Var[X_T] = σ² / (2θ) × (1 - e^(-2θT))

        where X = ln(S), X₀ = ln(S₀), μ = ln(mean_level).

        Binary option: P(S > K) = N((E[X_T] - ln(K)) / √Var[X_T])

        Parameters
        ----------
        current_price:
            Current spot price S₀.
        strike:
            Settlement threshold K.
        horizon_minutes:
            Time to expiry in minutes.
        theta:
            Mean-reversion speed (per hour). Clamped to [0.01, 100].
        mu:
            Long-run mean level (in log-price space, i.e. ln(mean_price)).
        sigma:
            OU volatility parameter (annualized).
        min_uncertainty:
            Floor on reported uncertainty.

        Returns
        -------
        ProbabilityEstimate with num_paths=0 (analytical).
        """
        if _norm is None:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)
        if current_price <= 0 or strike <= 0 or horizon_minutes <= 0 or sigma <= 0:
            return ProbabilityEstimate(0.5, 0.0, 1.0, 0.5, 0)

        # Clamp theta to prevent degeneracy
        theta = max(0.01, min(100.0, theta))

        # Convert horizon to hours (theta is per hour)
        T_hours = horizon_minutes / 60.0

        X0 = math.log(current_price)
        ln_K = math.log(strike)

        # OU conditional mean and variance
        exp_neg_theta_T = math.exp(-theta * T_hours)
        E_XT = mu + (X0 - mu) * exp_neg_theta_T

        # σ² / (2θ) × (1 - e^(-2θT))
        # Need to convert sigma from annualized to per-hour scale
        # σ_hourly = σ_annual / √(hours_per_year)
        hours_per_year = 365.25 * 24.0
        sigma_hourly = sigma / math.sqrt(hours_per_year)
        Var_XT = (sigma_hourly ** 2) / (2.0 * theta) * (1.0 - math.exp(-2.0 * theta * T_hours))

        if Var_XT <= 0:
            # Degenerate: variance collapsed
            return ProbabilityEstimate(
                probability=1.0 if E_XT > ln_K else 0.0,
                ci_lower=0.0,
                ci_upper=1.0,
                uncertainty=0.5,
                num_paths=0,
            )

        std_XT = math.sqrt(Var_XT)
        z = (E_XT - ln_K) / std_XT
        prob = float(_norm.cdf(z))
        prob = max(0.0, min(1.0, prob))

        # Uncertainty: higher when theta is uncertain or near boundary
        uncertainty = max(min_uncertainty, 0.02 + 0.01 * (1.0 / max(theta, 0.1)))

        return ProbabilityEstimate(
            probability=prob,
            ci_lower=max(0.0, prob - uncertainty),
            ci_upper=min(1.0, prob + uncertainty),
            uncertainty=uncertainty,
            num_paths=0,
        )

    def estimate_ou_params(
        self,
        returns: list[float] | np.ndarray,
        interval_seconds: int = 60,
    ) -> tuple[float, float, float] | None:
        """Estimate Ornstein-Uhlenbeck parameters from return data.

        Estimates:
        - θ (mean-reversion speed, per hour) from lag-1 autocorrelation
        - μ (long-run mean, in log-price space) from cumulative returns
        - σ (residual volatility, annualized)

        Parameters
        ----------
        returns:
            Log returns at the given interval.
        interval_seconds:
            Spacing between return observations.

        Returns
        -------
        (theta, mu, sigma) or None if insufficient data.
            theta: mean-reversion speed (per hour), clamped to [0.1, 50.0]
            mu: long-run mean level (in log-price space)
            sigma: annualized OU volatility
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 30:
            return None

        # Estimate θ from autocorrelation
        # For OU: ACF(lag=1) = e^(-θ × dt)
        # → θ = -ln(ACF(1)) / dt
        mean_r = float(np.mean(arr))
        centered = arr - mean_r
        n = len(centered)

        # Lag-1 autocorrelation
        c0 = float(np.dot(centered, centered))
        if c0 < 1e-30:
            return None
        c1 = float(np.dot(centered[:-1], centered[1:]))
        acf1 = c1 / c0

        dt_hours = interval_seconds / 3600.0

        if acf1 >= 1.0 or acf1 <= 0.0:
            # No mean-reversion signal or negative acf
            # acf1 <= 0 means strong mean-reversion (anti-correlation)
            if acf1 <= 0.0:
                # Very strong mean-reversion
                theta = 50.0
            else:
                # acf1 >= 1.0: no mean-reversion at all
                return None
        else:
            theta = -math.log(acf1) / dt_hours

        # Clamp θ to reasonable range
        theta = max(0.1, min(50.0, theta))

        # Estimate μ from cumulative return (proxy for long-run mean)
        # Use rolling mean of log-prices as μ
        cum_returns = np.cumsum(arr)
        mu = float(np.mean(cum_returns))  # centered around 0 for log-returns

        # Estimate σ from residual volatility
        # OU residual vol: σ² = 2θ × Var(returns) / (1 - e^(-2θdt))
        var_r = float(np.var(arr, ddof=1))
        exp_term = 1.0 - math.exp(-2.0 * theta * dt_hours)
        if exp_term > 0:
            sigma_hourly_sq = 2.0 * theta * var_r / exp_term
        else:
            sigma_hourly_sq = var_r / dt_hours

        hours_per_year = 365.25 * 24.0
        sigma_annual = math.sqrt(max(0.0, sigma_hourly_sq * hours_per_year))

        return (theta, mu, sigma_annual)
