"""Vol-spread signal generator using GARCH(1,1) vs implied volatility.

Computes a trading signal from the divergence between GARCH-forecasted
volatility and market-implied volatility extracted from Kalshi binary
option prices.

Signal logic:
    GARCH vol > implied vol  →  market underprices tail moves  →  buy OTM
    GARCH vol < implied vol  →  market overprices tail moves   →  sell OTM

The z-scored vol spread normalises for vol regime changes and centers
around any systematic GARCH bias.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.stats import norm as _norm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _norm = None  # type: ignore[assignment]

from arb_bot.crypto.garch_vol import GarchForecaster, GarchParams
from arb_bot.crypto.price_model import ProbabilityEstimate

# Crypto trades 24/7.
_MINUTES_PER_YEAR_CRYPTO = 365.25 * 24.0 * 60.0


@dataclass(frozen=True)
class VolSpreadSignal:
    """Vol-spread trading signal."""

    garch_vol: float          # Annualized GARCH vol forecast
    implied_vol: float        # Annualized implied vol from market price
    vol_spread: float         # garch_vol - implied_vol (raw)
    vol_spread_zscore: float  # Z-scored spread against rolling history
    signal_direction: str     # "buy" | "sell" | "neutral"
    garch_probability: float  # P(above) using GARCH vol
    confidence: float         # Signal confidence [0, 1]


class VolSpreadModel:
    """Vol-spread signal generator.

    Wraps a GarchForecaster and provides the full signal pipeline:
    1. Fit / update GARCH on 1-minute returns
    2. Forecast GARCH vol at the contract horizon
    3. Extract implied vol from the market price
    4. Compute z-scored vol spread
    5. Compute GARCH-based probability estimate

    Parameters
    ----------
    garch_forecaster:
        Pre-configured GarchForecaster instance. If None, creates one
        with defaults.
    spread_history_size:
        Rolling window size for z-scoring the vol spread.
    entry_z_threshold:
        Minimum |z-score| to generate a buy/sell signal.
    min_moneyness_sigma:
        Minimum moneyness in sigma units for signal generation.
    max_moneyness_sigma:
        Maximum moneyness in sigma units for signal generation.
    uncertainty_base:
        Base uncertainty added to GARCH probability estimates.
    """

    def __init__(
        self,
        garch_forecaster: GarchForecaster | None = None,
        spread_history_size: int = 500,
        entry_z_threshold: float = 1.5,
        min_moneyness_sigma: float = 0.3,
        max_moneyness_sigma: float = 2.5,
        uncertainty_base: float = 0.03,
    ) -> None:
        self._garch = garch_forecaster or GarchForecaster()
        self._spread_history: deque[float] = deque(maxlen=spread_history_size)
        self._entry_z = entry_z_threshold
        self._min_moneyness_sigma = min_moneyness_sigma
        self._max_moneyness_sigma = max_moneyness_sigma
        self._uncertainty_base = uncertainty_base

    # ── Public API ────────────────────────────────────────────────

    def compute_signal(
        self,
        returns_1m: np.ndarray,
        spot: float,
        strike: float,
        tte_minutes: float,
        market_price: float,
        direction: str = "above",
        drift: float = 0.0,
    ) -> Optional[VolSpreadSignal]:
        """Compute full vol-spread signal.

        Parameters
        ----------
        returns_1m:
            Array of 1-minute log-returns (most recent last).
        spot:
            Current spot price.
        strike:
            Binary option strike (settlement threshold).
        tte_minutes:
            Time to expiry in minutes.
        market_price:
            Observed binary option price [0, 1].
        direction:
            ``"above"`` or ``"below"``.
        drift:
            Annualized drift (mu). Default 0.

        Returns
        -------
        VolSpreadSignal or None
            None if GARCH fitting fails, IV extraction fails,
            or moneyness is out of range.
        """
        if spot <= 0 or strike <= 0 or tte_minutes <= 0:
            return None

        # ── 1. Fit or re-use GARCH parameters ───────────────────
        params = self._garch.last_params
        if params is None or self._garch.needs_refit:
            params = self._garch.fit(returns_1m)
        if params is None:
            return None

        # ── 2. Forecast GARCH vol at contract horizon ────────────
        horizon_steps = max(1, int(round(tte_minutes)))
        forecast = self._garch.forecast(params, returns_1m, horizon_steps)
        garch_vol = forecast.sigma_annualized

        # ── 3. Moneyness filter ──────────────────────────────────
        if not self._moneyness_in_range(spot, strike, garch_vol, tte_minutes):
            return None

        # ── 4. Extract implied vol from market price ─────────────
        implied_vol = self._extract_iv(
            market_price, strike, spot, tte_minutes, drift, direction,
        )
        if implied_vol is None:
            return None

        # ── 5. Vol spread + z-score ──────────────────────────────
        spread = garch_vol - implied_vol
        self._spread_history.append(spread)
        zscore = self._zscore(spread)

        # ── 6. Signal direction ──────────────────────────────────
        if zscore >= self._entry_z:
            signal_dir = "buy"
        elif zscore <= -self._entry_z:
            signal_dir = "sell"
        else:
            signal_dir = "neutral"

        # ── 7. GARCH-based probability ───────────────────────────
        garch_prob = self._garch_probability(
            spot, strike, garch_vol, tte_minutes, drift, direction,
        )

        # ── 8. Signal confidence ─────────────────────────────────
        confidence = self._signal_confidence(zscore, params, len(returns_1m))

        return VolSpreadSignal(
            garch_vol=garch_vol,
            implied_vol=implied_vol,
            vol_spread=spread,
            vol_spread_zscore=zscore,
            signal_direction=signal_dir,
            garch_probability=garch_prob,
            confidence=confidence,
        )

    def compute_garch_probability(
        self,
        spot: float,
        strike: float,
        garch_vol: float,
        tte_minutes: float,
        drift: float = 0.0,
        direction: str = "above",
    ) -> ProbabilityEstimate:
        """Compute binary probability estimate using GARCH vol.

        Uses Black-Scholes d2 formula:
            P(above) = N(d2)
            d2 = [ln(S/K) + (μ - σ²/2)·T] / (σ·√T)

        Parameters
        ----------
        spot, strike, garch_vol, tte_minutes, drift, direction:
            As in compute_signal.

        Returns
        -------
        ProbabilityEstimate
            With probability, confidence interval, and uncertainty.
        """
        prob = self._garch_probability(
            spot, strike, garch_vol, tte_minutes, drift, direction,
        )

        # Uncertainty: base + inverse-sqrt scaling with history size.
        history_n = max(len(self._spread_history), 1)
        uncertainty = self._uncertainty_base + 0.05 / math.sqrt(history_n)
        uncertainty = min(uncertainty, 0.15)  # Cap uncertainty

        ci_lower = max(0.0, prob - uncertainty)
        ci_upper = min(1.0, prob + uncertainty)

        return ProbabilityEstimate(
            probability=prob,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            uncertainty=uncertainty,
            num_paths=0,  # No Monte Carlo paths used
        )

    @property
    def garch_forecaster(self) -> GarchForecaster:
        """Access the underlying GARCH forecaster."""
        return self._garch

    @property
    def spread_history(self) -> list[float]:
        """Copy of the rolling spread history."""
        return list(self._spread_history)

    @property
    def spread_history_size(self) -> int:
        """Current number of spread observations."""
        return len(self._spread_history)

    # ── Internal helpers ──────────────────────────────────────────

    def _extract_iv(
        self,
        market_price: float,
        strike: float,
        spot: float,
        tte_minutes: float,
        drift: float,
        direction: str,
    ) -> Optional[float]:
        """Extract implied volatility using bisection on BS binary formula."""
        if _norm is None:
            return None  # pragma: no cover
        if market_price < 0.01 or market_price > 0.99:
            return None
        if spot <= 0 or strike <= 0 or tte_minutes <= 0:
            return None

        # For "below" contracts, invert: P(below) = 1 - P(above)
        target_prob = market_price if direction == "above" else (1.0 - market_price)

        dt = tte_minutes / _MINUTES_PER_YEAR_CRYPTO
        sqrt_dt = math.sqrt(dt)
        log_moneyness = math.log(spot / strike)

        def _binary_price(sigma: float) -> float:
            d2 = (log_moneyness + (drift - 0.5 * sigma * sigma) * dt) / (
                sigma * sqrt_dt
            )
            return float(_norm.cdf(d2))

        # Bisection bounds: 1% to 500% annualized vol.
        lo, hi = 0.01, 5.0

        p_lo = _binary_price(lo)
        p_hi = _binary_price(hi)

        # Check if solution exists within bounds.
        if (target_prob - p_lo) * (target_prob - p_hi) > 0:
            return None

        # Bisection.
        for _ in range(60):
            mid = (lo + hi) / 2.0
            p_mid = _binary_price(mid)
            if abs(p_mid - target_prob) < 1e-6:
                return mid
            if (p_mid - target_prob) * (p_lo - target_prob) < 0:
                hi = mid
            else:
                lo = mid
                p_lo = p_mid

        return (lo + hi) / 2.0

    def _garch_probability(
        self,
        spot: float,
        strike: float,
        garch_vol: float,
        tte_minutes: float,
        drift: float,
        direction: str,
    ) -> float:
        """P(above) or P(below) using GARCH vol and BS d2."""
        if _norm is None:
            return 0.5  # pragma: no cover
        if spot <= 0 or strike <= 0 or garch_vol <= 0 or tte_minutes <= 0:
            return 0.5

        dt = tte_minutes / _MINUTES_PER_YEAR_CRYPTO
        d2 = (
            math.log(spot / strike)
            + (drift - 0.5 * garch_vol ** 2) * dt
        ) / (garch_vol * math.sqrt(dt))

        p_above = float(_norm.cdf(d2))

        if direction == "below":
            return 1.0 - p_above
        return p_above

    def _moneyness_in_range(
        self,
        spot: float,
        strike: float,
        garch_vol: float,
        tte_minutes: float,
    ) -> bool:
        """Check if moneyness is within acceptable sigma range."""
        if garch_vol <= 0 or tte_minutes <= 0:
            return False

        dt = tte_minutes / _MINUTES_PER_YEAR_CRYPTO
        sigma_t = garch_vol * math.sqrt(dt)
        if sigma_t < 1e-10:
            return False

        log_moneyness = abs(math.log(spot / strike))
        moneyness_sigma = log_moneyness / sigma_t

        return self._min_moneyness_sigma <= moneyness_sigma <= self._max_moneyness_sigma

    def _zscore(self, spread: float) -> float:
        """Z-score a spread observation against rolling history.

        Returns 0.0 if insufficient history (< 20 observations) or
        if the standard deviation is negligible.
        """
        if len(self._spread_history) < 20:
            return 0.0

        arr = np.array(self._spread_history)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std < 1e-12:
            return 0.0

        return (spread - mean) / std

    def _signal_confidence(
        self,
        zscore: float,
        params: GarchParams,
        n_returns: int,
    ) -> float:
        """Compute signal confidence from z-score strength and data quality.

        Confidence is a product of:
        1. Z-score strength: sigmoid mapping |z| to [0, 1]
        2. GARCH convergence: penalise grid-search fallback
        3. Data sufficiency: penalise short histories
        """
        # z-score strength: sigmoid-ish mapping
        z_strength = min(abs(zscore) / 3.0, 1.0)

        # GARCH quality: converged MLE is more trustworthy
        garch_quality = 1.0 if params.converged else 0.7

        # Data sufficiency: ramp from 0.5 at min_obs to 1.0 at 1000+
        data_factor = min(1.0, 0.5 + 0.5 * n_returns / 1000.0)

        # Spread history depth
        hist_factor = min(1.0, len(self._spread_history) / 100.0)

        return z_strength * garch_quality * data_factor * hist_factor
