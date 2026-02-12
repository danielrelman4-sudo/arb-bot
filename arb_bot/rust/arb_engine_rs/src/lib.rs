//! Rust hot-path engine for arb_bot (Phase 7).
//!
//! This PyO3 extension module provides high-performance implementations
//! of the arb bot's compute-intensive functions: quote normalization,
//! opportunity detection, fill/execution modeling, and Kelly sizing.
//!
//! The Python orchestration layer (engine.py) dispatches to these Rust
//! functions when the module is available and enabled via env vars.

use pyo3::prelude::*;

pub mod types;
pub mod binary_math;
pub mod strategy;
pub mod eval_pipeline;

/// Returns the version of the Rust engine module.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns the list of available Rust-accelerated function names.
#[pyfunction]
fn available_functions() -> Vec<&'static str> {
    vec![
        "decompose_binary_quote",
        "reconstruct_binary_quote",
        "choose_effective_buy_price",
        "build_quote_diagnostics",
        "find_opportunities",
        "estimate_fee",
        "estimate_fill",
        "simulate_execution",
        "compute_kelly",
        "compute_sizing",
        "execution_aware_kelly_fraction",
    ]
}

/// The main PyO3 module definition.
#[pymodule]
fn arb_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available_functions, m)?)?;

    // binary_math functions (7B-1)
    binary_math::register(m)?;

    // strategy functions (7B-2)
    strategy::register(m)?;

    // eval_pipeline functions (7B-3)
    eval_pipeline::register(m)?;

    Ok(())
}
