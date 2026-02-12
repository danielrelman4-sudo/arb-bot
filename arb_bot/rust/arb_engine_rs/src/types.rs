//! Shared Rust types mirroring arb_bot Python dataclasses.
//!
//! These types are used internally by the Rust hot-path modules.
//! The PyO3 boundary converts between these and Python dicts/tuples.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// binary_math types
// ---------------------------------------------------------------------------

/// Mirrors `binary_math.QuoteDecomposition`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuoteDecomposition {
    pub implied_probability: f64,
    pub edge_per_side: f64,
}

/// Mirrors `binary_math.EffectiveBuy`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveBuy {
    pub price: f64,
    pub size: f64,
    /// "direct_ask" or "opposite_bid_transform"
    pub source: String,
}

/// Mirrors `binary_math.QuoteDiagnostics`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuoteDiagnostics {
    pub ask_implied_probability: f64,
    pub ask_edge_per_side: f64,
    pub bid_implied_probability: Option<f64>,
    pub bid_edge_per_side: Option<f64>,
    pub midpoint_consistency_gap: Option<f64>,
    pub yes_spread: Option<f64>,
    pub no_spread: Option<f64>,
    pub spread_asymmetry: Option<f64>,
}
