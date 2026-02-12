//! Rust port of `arb_bot/binary_math.py`.
//!
//! Pure arithmetic on binary-outcome (YES/NO) prediction market quotes.
//! All prices are in the range [0.0, 1.0]. Functions here are called on
//! every quote received, so performance is critical.
//!
//! ## Numerical Precision
//!
//! Python `float` and Rust `f64` are both IEEE 754 double-precision.
//! All operations (add, subtract, multiply, min, max) produce bit-identical
//! results. The `_EPSILON` constant and clamping logic are replicated exactly.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::types::{EffectiveBuy, QuoteDecomposition, QuoteDiagnostics};

/// Matches Python `_EPSILON = 1e-9`.
const EPSILON: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Internal helpers (not exported via PyO3)
// ---------------------------------------------------------------------------

/// Mirrors `_validate_price(value) -> float`.
///
/// Clamps to [0.0, 1.0] if within epsilon tolerance, raises ValueError
/// if outside.
fn validate_price(value: f64) -> Result<f64, String> {
    if value < -EPSILON || value > 1.0 + EPSILON {
        return Err(format!("price out of range [0, 1]: {}", value));
    }
    Ok(value.clamp(0.0, 1.0))
}

/// Mirrors `_normalize_price_or_none(value) -> float | None`.
///
/// Returns None for None input or out-of-range values.
fn normalize_price_or_none(value: Option<f64>) -> Option<f64> {
    let v = value?;
    if v < -EPSILON || v > 1.0 + EPSILON {
        return None;
    }
    Some(v.clamp(0.0, 1.0))
}

/// Mirrors `_clamp_price(value) -> float`.
fn clamp_price(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Mirrors `_normalize_size(value) -> float`.
fn normalize_size(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    value.max(0.0)
}

// ---------------------------------------------------------------------------
// Core functions (pure Rust, no PyO3 dependency)
// ---------------------------------------------------------------------------

/// Mirrors `decompose_binary_quote(yes_price, no_price)`.
pub fn decompose(yes_price: f64, no_price: f64) -> Result<QuoteDecomposition, String> {
    let yes = validate_price(yes_price)?;
    let no = validate_price(no_price)?;
    let implied_probability = 0.5 * (yes + (1.0 - no));
    let edge_per_side = 0.5 * (yes + no - 1.0);
    Ok(QuoteDecomposition {
        implied_probability,
        edge_per_side,
    })
}

/// Mirrors `reconstruct_binary_quote(implied_probability, edge_per_side)`.
pub fn reconstruct(implied_probability: f64, edge_per_side: f64) -> Result<(f64, f64), String> {
    let p = validate_price(implied_probability)?;
    let edge = edge_per_side;
    let yes = clamp_price(p + edge);
    let no = clamp_price((1.0 - p) + edge);
    Ok((yes, no))
}

/// Mirrors `zero_spread_yes_buy_from_no_bid(no_bid_price)`.
fn zero_spread_yes_buy_from_no_bid(no_bid_price: Option<f64>) -> Option<f64> {
    no_bid_price.map(|p| clamp_price(1.0 - p))
}

/// Mirrors `zero_spread_no_buy_from_yes_bid(yes_bid_price)`.
fn zero_spread_no_buy_from_yes_bid(yes_bid_price: Option<f64>) -> Option<f64> {
    yes_bid_price.map(|p| clamp_price(1.0 - p))
}

/// Mirrors `choose_effective_buy_price(side, ask, ask_size, opp_bid, opp_bid_size)`.
pub fn choose_effective_buy(
    side: &str,
    direct_ask_price: Option<f64>,
    direct_ask_size: f64,
    opposite_bid_price: Option<f64>,
    opposite_bid_size: f64,
) -> Option<EffectiveBuy> {
    let mut candidates: Vec<EffectiveBuy> = Vec::with_capacity(2);

    let normalized_ask = normalize_price_or_none(direct_ask_price);
    let normalized_ask_size = normalize_size(direct_ask_size);
    if let Some(ask) = normalized_ask {
        if normalized_ask_size > 0.0 {
            candidates.push(EffectiveBuy {
                price: ask,
                size: normalized_ask_size,
                source: "direct_ask".to_string(),
            });
        }
    }

    let transformed_price = if side == "yes" {
        zero_spread_yes_buy_from_no_bid(opposite_bid_price)
    } else {
        zero_spread_no_buy_from_yes_bid(opposite_bid_price)
    };
    let normalized_transformed_size = normalize_size(opposite_bid_size);
    if let Some(tp) = transformed_price {
        if normalized_transformed_size > 0.0 {
            candidates.push(EffectiveBuy {
                price: tp,
                size: normalized_transformed_size,
                source: "opposite_bid_transform".to_string(),
            });
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // Favor lowest price; on ties, favor deeper size (hence -size).
    candidates.sort_by(|a, b| {
        a.price
            .partial_cmp(&b.price)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                // Negate size comparison: larger size = lower sort key.
                b.size
                    .partial_cmp(&a.size)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    Some(candidates.into_iter().next().unwrap())
}

/// Mirrors `build_quote_diagnostics(yes_buy, no_buy, yes_bid, no_bid)`.
pub fn diagnostics(
    yes_buy_price: f64,
    no_buy_price: f64,
    yes_bid_price: Option<f64>,
    no_bid_price: Option<f64>,
) -> Result<QuoteDiagnostics, String> {
    let ask = decompose(yes_buy_price, no_buy_price)?;

    let yes_bid = normalize_price_or_none(yes_bid_price);
    let no_bid = normalize_price_or_none(no_bid_price);

    if yes_bid.is_none() || no_bid.is_none() {
        return Ok(QuoteDiagnostics {
            ask_implied_probability: ask.implied_probability,
            ask_edge_per_side: ask.edge_per_side,
            bid_implied_probability: None,
            bid_edge_per_side: None,
            midpoint_consistency_gap: None,
            yes_spread: None,
            no_spread: None,
            spread_asymmetry: None,
        });
    }

    let yb = yes_bid.unwrap();
    let nb = no_bid.unwrap();

    let bid_implied_probability = 0.5 * (yb + (1.0 - nb));
    let bid_edge_per_side = 0.5 * (1.0 - yb - nb);
    let yes_spread = yes_buy_price - yb;
    let no_spread = no_buy_price - nb;

    Ok(QuoteDiagnostics {
        ask_implied_probability: ask.implied_probability,
        ask_edge_per_side: ask.edge_per_side,
        bid_implied_probability: Some(bid_implied_probability),
        bid_edge_per_side: Some(bid_edge_per_side),
        midpoint_consistency_gap: Some((ask.implied_probability - bid_implied_probability).abs()),
        yes_spread: Some(yes_spread),
        no_spread: Some(no_spread),
        spread_asymmetry: Some(yes_spread - no_spread),
    })
}

// ---------------------------------------------------------------------------
// PyO3 registration
// ---------------------------------------------------------------------------

/// Register all binary_math PyO3 functions on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decompose_binary_quote_py, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_binary_quote_py, m)?)?;
    m.add_function(wrap_pyfunction!(choose_effective_buy_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_quote_diagnostics_py, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 boundary functions
// ---------------------------------------------------------------------------

/// PyO3 wrapper for `decompose_binary_quote`.
///
/// Returns a dict with keys: implied_probability, edge_per_side.
#[pyfunction]
#[pyo3(name = "decompose_binary_quote")]
pub fn decompose_binary_quote_py(
    yes_price: f64,
    no_price: f64,
) -> PyResult<(f64, f64)> {
    let result = decompose(yes_price, no_price)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok((result.implied_probability, result.edge_per_side))
}

/// PyO3 wrapper for `reconstruct_binary_quote`.
///
/// Returns a tuple (yes_price, no_price).
#[pyfunction]
#[pyo3(name = "reconstruct_binary_quote")]
pub fn reconstruct_binary_quote_py(
    implied_probability: f64,
    edge_per_side: f64,
) -> PyResult<(f64, f64)> {
    reconstruct(implied_probability, edge_per_side)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// PyO3 wrapper for `choose_effective_buy_price`.
///
/// Returns a tuple (price, size, source) or None.
#[pyfunction]
#[pyo3(name = "choose_effective_buy_price", signature = (side, direct_ask_price, direct_ask_size, opposite_bid_price, opposite_bid_size))]
pub fn choose_effective_buy_price_py(
    side: &str,
    direct_ask_price: Option<f64>,
    direct_ask_size: f64,
    opposite_bid_price: Option<f64>,
    opposite_bid_size: f64,
) -> Option<(f64, f64, String)> {
    choose_effective_buy(
        side,
        direct_ask_price,
        direct_ask_size,
        opposite_bid_price,
        opposite_bid_size,
    )
    .map(|eb| (eb.price, eb.size, eb.source))
}

/// PyO3 wrapper for `build_quote_diagnostics`.
///
/// Returns a dict with all QuoteDiagnostics fields.
#[pyfunction]
#[pyo3(name = "build_quote_diagnostics", signature = (yes_buy_price, no_buy_price, yes_bid_price=None, no_bid_price=None))]
pub fn build_quote_diagnostics_py<'py>(
    py: Python<'py>,
    yes_buy_price: f64,
    no_buy_price: f64,
    yes_bid_price: Option<f64>,
    no_bid_price: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = diagnostics(yes_buy_price, no_buy_price, yes_bid_price, no_bid_price)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("ask_implied_probability", result.ask_implied_probability)?;
    dict.set_item("ask_edge_per_side", result.ask_edge_per_side)?;
    dict.set_item("bid_implied_probability", result.bid_implied_probability)?;
    dict.set_item("bid_edge_per_side", result.bid_edge_per_side)?;
    dict.set_item("midpoint_consistency_gap", result.midpoint_consistency_gap)?;
    dict.set_item("yes_spread", result.yes_spread)?;
    dict.set_item("no_spread", result.no_spread)?;
    dict.set_item("spread_asymmetry", result.spread_asymmetry)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Rust unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_price_normal() {
        assert_eq!(validate_price(0.5).unwrap(), 0.5);
        assert_eq!(validate_price(0.0).unwrap(), 0.0);
        assert_eq!(validate_price(1.0).unwrap(), 1.0);
    }

    #[test]
    fn test_validate_price_clamps_epsilon() {
        // Slightly negative within epsilon → clamps to 0.0.
        assert_eq!(validate_price(-1e-10).unwrap(), 0.0);
        // Slightly above 1.0 within epsilon → clamps to 1.0.
        assert_eq!(validate_price(1.0 + 1e-10).unwrap(), 1.0);
    }

    #[test]
    fn test_validate_price_rejects_out_of_range() {
        assert!(validate_price(-0.1).is_err());
        assert!(validate_price(1.1).is_err());
    }

    #[test]
    fn test_decompose_standard() {
        let r = decompose(0.57, 0.48).unwrap();
        // implied_prob = 0.5 * (0.57 + (1.0 - 0.48)) = 0.5 * 1.09 = 0.545
        assert!((r.implied_probability - 0.545).abs() < 1e-12);
        // edge = 0.5 * (0.57 + 0.48 - 1.0) = 0.5 * 0.05 = 0.025
        assert!((r.edge_per_side - 0.025).abs() < 1e-12);
    }

    #[test]
    fn test_decompose_fair() {
        let r = decompose(0.50, 0.50).unwrap();
        assert!((r.implied_probability - 0.50).abs() < 1e-12);
        assert!((r.edge_per_side - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_reconstruct_roundtrip() {
        let original = decompose(0.57, 0.48).unwrap();
        let (yes, no) = reconstruct(original.implied_probability, original.edge_per_side).unwrap();
        assert!((yes - 0.57).abs() < 1e-12);
        assert!((no - 0.48).abs() < 1e-12);
    }

    #[test]
    fn test_choose_effective_buy_direct_only() {
        let result = choose_effective_buy("yes", Some(0.55), 100.0, None, 0.0);
        let eb = result.unwrap();
        assert!((eb.price - 0.55).abs() < 1e-12);
        assert!((eb.size - 100.0).abs() < 1e-12);
        assert_eq!(eb.source, "direct_ask");
    }

    #[test]
    fn test_choose_effective_buy_transform_wins() {
        // Direct ask at 0.60, opposite bid at 0.55 → transform = 1.0 - 0.55 = 0.45.
        let result = choose_effective_buy("yes", Some(0.60), 50.0, Some(0.55), 75.0);
        let eb = result.unwrap();
        assert!((eb.price - 0.45).abs() < 1e-12);
        assert_eq!(eb.source, "opposite_bid_transform");
    }

    #[test]
    fn test_choose_effective_buy_none() {
        let result = choose_effective_buy("yes", None, 0.0, None, 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_diagnostics_with_bids() {
        let r = diagnostics(0.57, 0.48, Some(0.53), Some(0.44)).unwrap();
        assert!((r.ask_implied_probability - 0.545).abs() < 1e-12);
        assert!(r.bid_implied_probability.is_some());
        assert!(r.yes_spread.is_some());
        assert!(r.spread_asymmetry.is_some());
    }

    #[test]
    fn test_diagnostics_without_bids() {
        let r = diagnostics(0.57, 0.48, None, None).unwrap();
        assert!(r.bid_implied_probability.is_none());
        assert!(r.yes_spread.is_none());
        assert!(r.spread_asymmetry.is_none());
    }

    #[test]
    fn test_normalize_size_nan() {
        assert_eq!(normalize_size(f64::NAN), 0.0);
    }

    #[test]
    fn test_normalize_size_negative() {
        assert_eq!(normalize_size(-5.0), 0.0);
    }

    #[test]
    fn test_clamp_price_bounds() {
        assert_eq!(clamp_price(-0.5), 0.0);
        assert_eq!(clamp_price(1.5), 1.0);
        assert_eq!(clamp_price(0.5), 0.5);
    }
}
