//! Rust port of the sizing/eval pipeline (Phase 7B-3).
//!
//! Combines fee_model, fill_model, execution_model, kelly_sizing, and sizing
//! into a single evaluation pipeline. Each module is a standalone function
//! accessible via PyO3, plus a combined `evaluate_opportunity` that chains
//! them all.
//!
//! All operations are pure arithmetic — no I/O, no state mutation.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Fee Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct VenueFeeScheduleInput {
    pub venue: String,
    pub taker_fee_per_contract: f64,
    pub maker_fee_per_contract: f64,
    pub taker_fee_rate: f64,
    pub maker_fee_rate: f64,
    pub taker_curve_coefficient: f64,
    pub maker_curve_coefficient: f64,
    pub curve_round_up: bool,
    pub settlement_fee_per_contract: f64,
    pub min_fee_per_order: f64,
    pub max_fee_per_order: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeeEstimateOutput {
    pub venue: String,
    pub order_type: String,
    pub contracts: i64,
    pub per_contract_fee: f64,
    pub total_fee: f64,
    pub settlement_fee: f64,
}

/// Mirrors `FeeModel.estimate()`.
pub fn estimate_fee(
    venue: &str,
    order_type: &str,       // "taker" or "maker"
    contracts: i64,
    price: f64,
    schedule: &VenueFeeScheduleInput,
) -> FeeEstimateOutput {
    let is_taker = order_type == "taker";

    // Flat component.
    let flat_per = if is_taker {
        schedule.taker_fee_per_contract
    } else {
        schedule.maker_fee_per_contract
    };
    let flat_total = flat_per * contracts as f64;

    // Proportional component.
    let rate = if is_taker {
        schedule.taker_fee_rate
    } else {
        schedule.maker_fee_rate
    };
    let prop_total = rate * price * contracts as f64;

    // Curve component (Kalshi P*(1-P)).
    let coeff = if is_taker {
        schedule.taker_curve_coefficient
    } else {
        schedule.maker_curve_coefficient
    };
    let curve_total = if coeff > 0.0 && price > 0.0 && price < 1.0 {
        let raw_curve = coeff * contracts as f64 * price * (1.0 - price);
        if schedule.curve_round_up {
            (raw_curve * 100.0).ceil() / 100.0
        } else {
            raw_curve
        }
    } else {
        0.0
    };

    // Raw total before caps.
    let mut raw_total = flat_total + prop_total + curve_total;

    // Min/max caps.
    if schedule.min_fee_per_order > 0.0 && raw_total < schedule.min_fee_per_order {
        raw_total = schedule.min_fee_per_order;
    }
    if schedule.max_fee_per_order > 0.0 && raw_total > schedule.max_fee_per_order {
        raw_total = schedule.max_fee_per_order;
    }

    // Settlement fee (separate).
    let settlement = schedule.settlement_fee_per_contract * contracts as f64;

    // Per-contract average.
    let per_contract = raw_total / (contracts.max(1) as f64);

    FeeEstimateOutput {
        venue: venue.to_string(),
        order_type: order_type.to_string(),
        contracts,
        per_contract_fee: per_contract,
        total_fee: raw_total,
        settlement_fee: settlement,
    }
}

// ---------------------------------------------------------------------------
// Fill Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct FillModelConfig {
    pub stale_quote_half_life_seconds: f64,
    pub queue_depth_factor: f64,
    pub spread_penalty_weight: f64,
    pub transform_source_penalty: f64,
    pub partial_fill_penalty_per_contract: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FillLegInput {
    pub side: String,
    pub buy_price: f64,
    pub buy_size: f64,
    pub spread: f64,
    pub source: String, // "direct_ask" or "opposite_bid_transform"
}

#[derive(Debug, Clone, Serialize)]
pub struct FillEstimateOutput {
    pub all_fill_probability: f64,
    pub partial_fill_probability: f64,
    pub expected_slippage_per_contract: f64,
    pub fill_quality_score: f64,
    pub adverse_selection_flag: bool,
    pub expected_realized_edge_per_contract: f64,
    pub expected_realized_profit: f64,
}

/// Mirrors `FillModel.estimate()`.
pub fn estimate_fill(
    legs: &[FillLegInput],
    contracts: i64,
    edge_per_contract: f64,
    staleness_seconds: f64,
    config: &FillModelConfig,
) -> FillEstimateOutput {
    if contracts <= 0 || legs.is_empty() {
        return FillEstimateOutput {
            all_fill_probability: 0.0,
            partial_fill_probability: 1.0,
            expected_slippage_per_contract: 0.0,
            fill_quality_score: 0.0,
            adverse_selection_flag: false,
            expected_realized_edge_per_contract: -config.partial_fill_penalty_per_contract,
            expected_realized_profit: 0.0,
        };
    }

    let half_life = config.stale_quote_half_life_seconds.max(0.1);
    let staleness_factor = (-staleness_seconds.max(0.0) / half_life).exp();

    let mut all_fill_prob = 1.0f64;
    let mut total_slippage = 0.0f64;
    let mut total_quality = 0.0f64;

    for leg in legs {
        // Depth ratio.
        let available = leg.buy_size.max(0.0);
        let denom = (contracts as f64) * config.queue_depth_factor.max(1.0);
        let depth_ratio = (available / denom.max(1.0)).min(1.0);

        // Spread penalty.
        let spread = leg.spread.max(0.0);
        let spread_penalty = (spread * config.spread_penalty_weight).clamp(0.0, 0.9);

        // Source penalty.
        let source_penalty = if leg.source == "opposite_bid_transform" {
            config.transform_source_penalty
        } else {
            0.0
        };

        // Leg probability.
        let prob = (depth_ratio * staleness_factor * (1.0 - spread_penalty) * (1.0 - source_penalty))
            .clamp(0.0, 1.0);
        all_fill_prob *= prob;

        // Leg slippage.
        total_slippage += spread * (1.0 - prob);

        // Fill quality (simplified).
        total_quality += if spread > 0.0 { 0.0 } else { 0.0 };
    }

    let n = legs.len() as f64;
    let partial_fill_prob = 1.0 - all_fill_prob;
    let avg_slippage = total_slippage / n;
    let fill_quality = total_quality / n;

    let expected_realized_edge = edge_per_contract * all_fill_prob
        - config.partial_fill_penalty_per_contract * partial_fill_prob
        - avg_slippage;
    let expected_realized_profit = expected_realized_edge * contracts as f64;

    FillEstimateOutput {
        all_fill_probability: all_fill_prob,
        partial_fill_probability: partial_fill_prob,
        expected_slippage_per_contract: avg_slippage,
        fill_quality_score: fill_quality,
        adverse_selection_flag: fill_quality < 0.0,
        expected_realized_edge_per_contract: expected_realized_edge,
        expected_realized_profit,
    }
}

// ---------------------------------------------------------------------------
// Execution Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionModelConfig {
    pub queue_decay_half_life_seconds: f64,
    pub latency_seconds: f64,
    pub market_impact_factor: f64,
    pub max_market_impact: f64,
    pub min_fill_fraction: f64,
    pub fill_fraction_steps: i64,
    pub sequential_leg_delay_seconds: f64,
    pub enable_queue_decay: bool,
    pub enable_market_impact: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExecLegInput {
    pub venue: String,
    pub market_id: String,
    pub side: String,
    pub buy_price: f64,
    pub available_size: f64,
    pub spread: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LegFillEstimateOutput {
    pub venue: String,
    pub market_id: String,
    pub side: String,
    pub fill_probability: f64,
    pub expected_fill_fraction: f64,
    pub queue_position_score: f64,
    pub market_impact: f64,
    pub expected_slippage: f64,
    pub time_offset_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionEstimateOutput {
    pub legs: Vec<LegFillEstimateOutput>,
    pub all_fill_probability: f64,
    pub expected_fill_fraction: f64,
    pub expected_slippage_per_contract: f64,
    pub expected_market_impact_per_contract: f64,
    pub graduated_fill_distribution: Vec<(f64, f64)>,
}

/// Mirrors `ExecutionModel.simulate()`.
pub fn simulate_execution(
    legs: &[ExecLegInput],
    contracts: i64,
    staleness_seconds: f64,
    sequential: bool,
    config: &ExecutionModelConfig,
) -> ExecutionEstimateOutput {
    // Early return for zero or negative contracts.
    if contracts <= 0 {
        let empty_legs: Vec<LegFillEstimateOutput> = legs.iter().map(|leg| {
            LegFillEstimateOutput {
                venue: leg.venue.clone(),
                market_id: leg.market_id.clone(),
                side: leg.side.clone(),
                fill_probability: 0.0,
                expected_fill_fraction: 0.0,
                queue_position_score: 0.0,
                market_impact: 0.0,
                expected_slippage: 0.0,
                time_offset_seconds: staleness_seconds + config.latency_seconds,
            }
        }).collect();
        return ExecutionEstimateOutput {
            legs: empty_legs,
            all_fill_probability: 0.0,
            expected_fill_fraction: 0.0,
            expected_slippage_per_contract: 0.0,
            expected_market_impact_per_contract: 0.0,
            graduated_fill_distribution: vec![(0.0, 1.0)],
        };
    }

    // Early return for empty legs.
    if legs.is_empty() {
        return ExecutionEstimateOutput {
            legs: Vec::new(),
            all_fill_probability: 1.0,
            expected_fill_fraction: 0.0,
            expected_slippage_per_contract: 0.0,
            expected_market_impact_per_contract: 0.0,
            graduated_fill_distribution: vec![(0.0, 1.0)],
        };
    }

    let mut leg_estimates = Vec::with_capacity(legs.len());
    let mut all_fill_prob = 1.0f64;
    let mut total_slippage = 0.0f64;
    let mut total_impact = 0.0f64;
    let mut min_fill_fraction = f64::MAX;

    for (i, leg) in legs.iter().enumerate() {
        // Time offset.
        let mut time_offset = staleness_seconds + config.latency_seconds;
        if sequential && i > 0 {
            time_offset += config.sequential_leg_delay_seconds * i as f64;
        }

        // Queue position score.
        let base = (leg.available_size / (contracts.max(1) as f64)).min(1.0);
        let queue_score = if config.enable_queue_decay && config.queue_decay_half_life_seconds > 0.0 {
            let decay = (-time_offset * 2.0f64.ln() / config.queue_decay_half_life_seconds.max(0.01)).exp();
            base * decay
        } else {
            base
        };

        // Market impact.
        let impact = if config.enable_market_impact && leg.spread > 0.0 {
            let size_ratio = contracts as f64 / leg.available_size.max(1.0);
            let raw = leg.spread * size_ratio * config.market_impact_factor;
            let max_imp = leg.spread * config.max_market_impact;
            raw.min(max_imp)
        } else {
            0.0
        };

        // Slippage.
        let slippage = leg.spread * (1.0 - queue_score) + impact;

        // Fill probability.
        let depth_coverage = (leg.available_size / (contracts.max(1) as f64)).min(1.0);
        let fill_prob = (queue_score * depth_coverage).clamp(0.0, 1.0);

        // Fill fraction.
        let size_cap = (leg.available_size / contracts.max(1) as f64).min(1.0);
        let raw_frac = fill_prob * size_cap;
        let fill_frac = if raw_frac < config.min_fill_fraction {
            0.0
        } else {
            raw_frac
        };

        all_fill_prob *= fill_prob;
        total_slippage += slippage;
        total_impact += impact;
        if fill_frac < min_fill_fraction {
            min_fill_fraction = fill_frac;
        }

        leg_estimates.push(LegFillEstimateOutput {
            venue: leg.venue.clone(),
            market_id: leg.market_id.clone(),
            side: leg.side.clone(),
            fill_probability: fill_prob,
            expected_fill_fraction: fill_frac,
            queue_position_score: queue_score,
            market_impact: impact,
            expected_slippage: slippage,
            time_offset_seconds: time_offset,
        });
    }

    let n = legs.len().max(1) as f64;
    let expected_fill = if min_fill_fraction == f64::MAX {
        0.0
    } else {
        min_fill_fraction
    };

    // Graduated fill distribution.
    let steps = (config.fill_fraction_steps.max(2) + 1) as usize;
    let mut distribution = Vec::with_capacity(steps);
    let min_leg_fill = leg_estimates
        .iter()
        .map(|l| l.fill_probability)
        .fold(f64::MAX, f64::min);
    let min_leg_fill = if min_leg_fill == f64::MAX { 0.0 } else { min_leg_fill };
    let num_legs = legs.len() as f64;

    let mut remaining = 1.0f64;
    for s in 0..steps {
        let frac = s as f64 / (steps - 1).max(1) as f64;
        let prob = if s == steps - 1 {
            remaining.max(0.0)
        } else if s == 0 {
            (1.0 - min_leg_fill).powf(num_legs).max(0.0)
        } else {
            let intermediate_count = (steps - 2).max(1) as f64;
            let partial_total = remaining - min_leg_fill.powf(num_legs);
            if partial_total > 0.0 {
                partial_total / intermediate_count
            } else {
                0.0
            }
        };
        distribution.push((frac, prob));
        remaining -= prob;
    }

    ExecutionEstimateOutput {
        legs: leg_estimates,
        all_fill_probability: all_fill_prob,
        expected_fill_fraction: expected_fill,
        expected_slippage_per_contract: total_slippage / n,
        expected_market_impact_per_contract: total_impact / n,
        graduated_fill_distribution: distribution,
    }
}

// ---------------------------------------------------------------------------
// Kelly Sizing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct KellyResult {
    pub raw_kelly: f64,
    pub adjusted_fraction: f64,
    pub uncertainty_haircut: f64,
    pub variance_haircut: f64,
    pub confidence: f64,
    pub blocked: bool,
    pub block_reason: String,
}

/// Mirrors `_raw_kelly()` from kelly_sizing.py.
fn raw_kelly(edge: f64, cost: f64, fill_prob: f64, failure_loss: Option<f64>) -> f64 {
    if edge <= 0.0 || cost <= 0.0 {
        return 0.0;
    }

    let p = fill_prob.clamp(0.0, 1.0);
    let q = 1.0 - p;
    let b = edge / cost;
    if b <= 0.0 {
        return 0.0;
    }

    let fl = failure_loss.unwrap_or(cost).clamp(0.0, cost);
    let a = fl / cost;

    let raw = if a <= 0.0 {
        1.0
    } else {
        ((b * p - a * q) / (a * b)).max(0.0)
    };

    // Damping by fill probability.
    let adjusted = raw * p.sqrt();
    adjusted.clamp(0.0, 1.0)
}

/// Mirrors `TailRiskKelly.compute()`.
pub fn compute_kelly(
    edge: f64,
    cost: f64,
    fill_prob: f64,
    model_uncertainty: f64,
    lane_variance: f64,
    failure_loss: Option<f64>,
    base_kelly_fraction: f64,
    uncertainty_haircut_factor: f64,
    variance_haircut_factor: f64,
    min_confidence: f64,
    max_model_uncertainty: f64,
) -> KellyResult {
    // Blocking checks.
    if model_uncertainty > max_model_uncertainty {
        return KellyResult {
            raw_kelly: 0.0,
            adjusted_fraction: 0.0,
            uncertainty_haircut: 1.0,
            variance_haircut: 0.0,
            confidence: (1.0 - model_uncertainty).max(0.0),
            blocked: true,
            block_reason: "model_uncertainty_too_high".to_string(),
        };
    }

    let confidence = (1.0 - model_uncertainty).max(0.0);
    if confidence < min_confidence {
        return KellyResult {
            raw_kelly: 0.0,
            adjusted_fraction: 0.0,
            uncertainty_haircut: 1.0,
            variance_haircut: 0.0,
            confidence,
            blocked: true,
            block_reason: "confidence_below_minimum".to_string(),
        };
    }

    let rk = raw_kelly(edge, cost, fill_prob, failure_loss);
    if rk <= 0.0 {
        return KellyResult {
            raw_kelly: rk,
            adjusted_fraction: 0.0,
            uncertainty_haircut: 0.0,
            variance_haircut: 0.0,
            confidence,
            blocked: true,
            block_reason: "negative_edge".to_string(),
        };
    }

    // Fractional cap.
    let mut fraction = rk.min(base_kelly_fraction);

    // Uncertainty haircut.
    let u_haircut = (model_uncertainty * uncertainty_haircut_factor).min(1.0);
    fraction *= 1.0 - u_haircut;

    // Variance haircut.
    let v_haircut = (lane_variance * variance_haircut_factor).min(1.0);
    fraction *= 1.0 - v_haircut;

    // Final clamp.
    fraction = fraction.clamp(0.0, 1.0);

    KellyResult {
        raw_kelly: rk,
        adjusted_fraction: fraction,
        uncertainty_haircut: u_haircut,
        variance_haircut: v_haircut,
        confidence,
        blocked: false,
        block_reason: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Sizing (build_trade_plan core logic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SizingInput {
    pub total_cost_per_contract: f64,
    pub net_edge_per_contract: f64,
    pub legs: Vec<SizingLegInput>,
    pub capital_per_contract_by_venue: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SizingLegInput {
    pub venue: String,
    pub buy_price: f64,
    pub buy_size: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SizingConfig {
    pub max_dollars_per_trade: f64,
    pub max_contracts_per_trade: i64,
    pub max_liquidity_fraction_per_trade: f64,
    pub max_bankroll_fraction_per_trade: f64,
    pub min_expected_profit_usd: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizingResult {
    pub contracts: i64,
    pub capital_required: f64,
    pub expected_profit: f64,
    pub capital_required_by_venue: std::collections::HashMap<String, f64>,
    pub capped_by: String,
}

/// Mirrors `PositionSizer.build_trade_plan()` sizing logic.
pub fn compute_sizing(
    opportunity: &SizingInput,
    available_cash_by_venue: &std::collections::HashMap<String, f64>,
    config: &SizingConfig,
) -> Option<SizingResult> {
    if opportunity.total_cost_per_contract <= 0.0 {
        return None;
    }

    // Per-venue cash caps.
    let mut max_from_cash = i64::MAX;
    for leg in &opportunity.legs {
        let venue_cash = available_cash_by_venue.get(&leg.venue).copied().unwrap_or(0.0);
        let unit_cost = leg.buy_price;
        if unit_cost > 0.0 {
            let cap = ((venue_cash * config.max_bankroll_fraction_per_trade) / unit_cost).floor() as i64;
            max_from_cash = max_from_cash.min(cap);
        }
    }

    // Trade dollar cap.
    let max_from_trade = (config.max_dollars_per_trade / opportunity.total_cost_per_contract).floor() as i64;

    // Liquidity cap.
    let liq_frac = config.max_liquidity_fraction_per_trade.clamp(0.0, 1.0);
    let max_from_liq = opportunity
        .legs
        .iter()
        .map(|l| (l.buy_size * liq_frac).floor() as i64)
        .min()
        .unwrap_or(0);

    // Combined cap.
    let caps = [
        (max_from_cash, "cash"),
        (max_from_trade, "trade_dollars"),
        (max_from_liq, "liquidity"),
        (config.max_contracts_per_trade, "max_contracts"),
    ];

    let (contracts, capped_by) = caps
        .iter()
        .min_by_key(|(v, _)| *v)
        .map(|(v, name)| (*v, name.to_string()))
        .unwrap_or((0, "none".to_string()));

    if contracts <= 0 {
        return None;
    }

    // Profit check.
    let expected_profit = contracts as f64 * opportunity.net_edge_per_contract;
    if expected_profit < config.min_expected_profit_usd {
        return None;
    }

    // Capital required by venue.
    let mut cap_by_venue = std::collections::HashMap::new();
    for (venue, unit_cost) in &opportunity.capital_per_contract_by_venue {
        cap_by_venue.insert(venue.clone(), unit_cost * contracts as f64);
    }

    Some(SizingResult {
        contracts,
        capital_required: contracts as f64 * opportunity.total_cost_per_contract,
        expected_profit,
        capital_required_by_venue: cap_by_venue,
        capped_by,
    })
}

// ---------------------------------------------------------------------------
// Execution-aware Kelly (standalone, mirrors sizing.py's static method)
// ---------------------------------------------------------------------------

/// Mirrors `PositionSizer.execution_aware_kelly_fraction()`.
pub fn execution_aware_kelly_fraction(
    edge: f64,
    cost: f64,
    fill_prob: f64,
    failure_loss: Option<f64>,
) -> f64 {
    raw_kelly(edge, cost, fill_prob, failure_loss)
}

// ---------------------------------------------------------------------------
// PyO3 registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_fee_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_fill_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_execution_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_kelly_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sizing_py, m)?)?;
    m.add_function(wrap_pyfunction!(execution_aware_kelly_fraction_py, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 boundary functions — JSON-in/JSON-out
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "estimate_fee")]
pub fn estimate_fee_py(
    venue: &str,
    order_type: &str,
    contracts: i64,
    price: f64,
    schedule_json: &str,
) -> PyResult<String> {
    let schedule: VenueFeeScheduleInput = serde_json::from_str(schedule_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("schedule JSON: {}", e)))?;
    let result = estimate_fee(venue, order_type, contracts, price, &schedule);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {}", e)))
}

#[pyfunction]
#[pyo3(name = "estimate_fill")]
pub fn estimate_fill_py(
    legs_json: &str,
    contracts: i64,
    edge_per_contract: f64,
    staleness_seconds: f64,
    config_json: &str,
) -> PyResult<String> {
    let legs: Vec<FillLegInput> = serde_json::from_str(legs_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("legs JSON: {}", e)))?;
    let config: FillModelConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config JSON: {}", e)))?;
    let result = estimate_fill(&legs, contracts, edge_per_contract, staleness_seconds, &config);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {}", e)))
}

#[pyfunction]
#[pyo3(name = "simulate_execution")]
pub fn simulate_execution_py(
    legs_json: &str,
    contracts: i64,
    staleness_seconds: f64,
    sequential: bool,
    config_json: &str,
) -> PyResult<String> {
    let legs: Vec<ExecLegInput> = serde_json::from_str(legs_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("legs JSON: {}", e)))?;
    let config: ExecutionModelConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config JSON: {}", e)))?;
    let result = simulate_execution(&legs, contracts, staleness_seconds, sequential, &config);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {}", e)))
}

#[pyfunction]
#[pyo3(name = "compute_kelly", signature = (edge, cost, fill_prob, model_uncertainty, lane_variance, failure_loss, base_kelly_fraction, uncertainty_haircut_factor, variance_haircut_factor, min_confidence, max_model_uncertainty))]
pub fn compute_kelly_py(
    edge: f64,
    cost: f64,
    fill_prob: f64,
    model_uncertainty: f64,
    lane_variance: f64,
    failure_loss: Option<f64>,
    base_kelly_fraction: f64,
    uncertainty_haircut_factor: f64,
    variance_haircut_factor: f64,
    min_confidence: f64,
    max_model_uncertainty: f64,
) -> PyResult<String> {
    let result = compute_kelly(
        edge,
        cost,
        fill_prob,
        model_uncertainty,
        lane_variance,
        failure_loss,
        base_kelly_fraction,
        uncertainty_haircut_factor,
        variance_haircut_factor,
        min_confidence,
        max_model_uncertainty,
    );
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {}", e)))
}

#[pyfunction]
#[pyo3(name = "compute_sizing")]
pub fn compute_sizing_py(
    opportunity_json: &str,
    cash_by_venue_json: &str,
    config_json: &str,
) -> PyResult<String> {
    let opp: SizingInput = serde_json::from_str(opportunity_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("opportunity JSON: {}", e)))?;
    let cash: std::collections::HashMap<String, f64> = serde_json::from_str(cash_by_venue_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("cash JSON: {}", e)))?;
    let config: SizingConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config JSON: {}", e)))?;
    let result = compute_sizing(&opp, &cash, &config);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialize: {}", e)))
}

#[pyfunction]
#[pyo3(name = "execution_aware_kelly_fraction", signature = (edge, cost, fill_prob, failure_loss=None))]
pub fn execution_aware_kelly_fraction_py(
    edge: f64,
    cost: f64,
    fill_prob: f64,
    failure_loss: Option<f64>,
) -> f64 {
    execution_aware_kelly_fraction(edge, cost, fill_prob, failure_loss)
}

// ---------------------------------------------------------------------------
// Rust unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_schedule() -> VenueFeeScheduleInput {
        VenueFeeScheduleInput {
            venue: "test".to_string(),
            taker_fee_per_contract: 0.01,
            maker_fee_per_contract: 0.005,
            taker_fee_rate: 0.0,
            maker_fee_rate: 0.0,
            taker_curve_coefficient: 0.0,
            maker_curve_coefficient: 0.0,
            curve_round_up: true,
            settlement_fee_per_contract: 0.0,
            min_fee_per_order: 0.0,
            max_fee_per_order: 0.0,
        }
    }

    #[test]
    fn test_fee_flat_taker() {
        let s = default_schedule();
        let r = estimate_fee("test", "taker", 10, 0.50, &s);
        assert!((r.total_fee - 0.10).abs() < 1e-12);
        assert!((r.per_contract_fee - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_fee_flat_maker() {
        let s = default_schedule();
        let r = estimate_fee("test", "maker", 10, 0.50, &s);
        assert!((r.total_fee - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_fee_curve_kalshi() {
        let s = VenueFeeScheduleInput {
            taker_curve_coefficient: 1.0,
            curve_round_up: true,
            ..default_schedule()
        };
        let r = estimate_fee("test", "taker", 10, 0.50, &s);
        // flat = 0.10, curve = ceil(1.0 * 10 * 0.5 * 0.5 * 100) / 100 = ceil(250) / 100 = 2.50
        assert!((r.total_fee - (0.10 + 2.50)).abs() < 1e-12);
    }

    #[test]
    fn test_raw_kelly_positive() {
        // Need edge/cost ratio (b) high enough: b*p > a*q => (edge/cost)*p > 1.0*(1-p)
        // edge=0.50, cost=0.50, fill_prob=0.8: b=1.0, b*p=0.8, a*q=0.2 => positive
        let k = raw_kelly(0.50, 0.50, 0.8, None);
        assert!(k > 0.0, "kelly should be positive, got {}", k);
        assert!(k <= 1.0);
    }

    #[test]
    fn test_raw_kelly_edge_too_small() {
        // edge=0.10, cost=0.50, fill_prob=0.8: b=0.2, b*p=0.16, a*q=0.20 => negative → 0
        let k = raw_kelly(0.10, 0.50, 0.8, None);
        assert!((k - 0.0).abs() < 1e-12, "kelly should be 0 for insufficient edge, got {}", k);
    }

    #[test]
    fn test_raw_kelly_negative_edge() {
        let k = raw_kelly(-0.10, 0.50, 0.8, None);
        assert!((k - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_raw_kelly_zero_cost() {
        let k = raw_kelly(0.10, 0.0, 0.8, None);
        assert!((k - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_kelly_blocked_uncertainty() {
        let r = compute_kelly(0.10, 0.50, 0.8, 0.9, 0.0, None, 0.25, 1.0, 0.5, 0.1, 0.8);
        assert!(r.blocked);
        assert_eq!(r.block_reason, "model_uncertainty_too_high");
    }

    #[test]
    fn test_compute_kelly_normal() {
        // edge=0.50, cost=0.50 → b=1.0, fill_prob=0.8 → b*p=0.8 > a*q=0.2 → positive Kelly
        let r = compute_kelly(0.50, 0.50, 0.8, 0.1, 0.0, None, 0.25, 1.0, 0.5, 0.1, 0.8);
        assert!(!r.blocked, "should not be blocked: {}", r.block_reason);
        assert!(r.adjusted_fraction > 0.0, "adjusted_fraction should be positive, got {}", r.adjusted_fraction);
        assert!(r.adjusted_fraction <= 0.25);
    }

    #[test]
    fn test_compute_kelly_blocked_insufficient_edge() {
        // edge=0.10, cost=0.50 → raw kelly = 0 → blocked with negative_edge
        let r = compute_kelly(0.10, 0.50, 0.8, 0.1, 0.0, None, 0.25, 1.0, 0.5, 0.1, 0.8);
        assert!(r.blocked);
        assert_eq!(r.block_reason, "negative_edge");
    }

    #[test]
    fn test_execution_aware_kelly() {
        // Same logic as raw_kelly — need sufficient edge
        let f = execution_aware_kelly_fraction(0.50, 0.50, 0.8, None);
        assert!(f > 0.0, "kelly fraction should be positive, got {}", f);
        assert!(f <= 1.0);
    }

    #[test]
    fn test_execution_aware_kelly_insufficient() {
        let f = execution_aware_kelly_fraction(0.10, 0.50, 0.8, None);
        assert!((f - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_sizing_basic() {
        let opp = SizingInput {
            total_cost_per_contract: 0.90,
            net_edge_per_contract: 0.10,
            legs: vec![
                SizingLegInput { venue: "kalshi".to_string(), buy_price: 0.30, buy_size: 100.0 },
                SizingLegInput { venue: "kalshi".to_string(), buy_price: 0.30, buy_size: 100.0 },
                SizingLegInput { venue: "kalshi".to_string(), buy_price: 0.30, buy_size: 100.0 },
            ],
            capital_per_contract_by_venue: [("kalshi".to_string(), 0.90)].into(),
        };
        let cash = [("kalshi".to_string(), 100.0)].iter().cloned().collect();
        let config = SizingConfig {
            max_dollars_per_trade: 50.0,
            max_contracts_per_trade: 100,
            max_liquidity_fraction_per_trade: 0.5,
            max_bankroll_fraction_per_trade: 0.25,
            min_expected_profit_usd: 0.01,
        };
        let r = compute_sizing(&opp, &cash, &config).unwrap();
        assert!(r.contracts > 0);
        assert!(r.expected_profit > 0.0);
    }

    #[test]
    fn test_sizing_zero_cost() {
        let opp = SizingInput {
            total_cost_per_contract: 0.0,
            net_edge_per_contract: 0.10,
            legs: vec![],
            capital_per_contract_by_venue: std::collections::HashMap::new(),
        };
        let cash = std::collections::HashMap::new();
        let config = SizingConfig {
            max_dollars_per_trade: 50.0,
            max_contracts_per_trade: 100,
            max_liquidity_fraction_per_trade: 0.5,
            max_bankroll_fraction_per_trade: 0.25,
            min_expected_profit_usd: 0.01,
        };
        assert!(compute_sizing(&opp, &cash, &config).is_none());
    }

    #[test]
    fn test_simulate_execution_basic() {
        let legs = vec![ExecLegInput {
            venue: "kalshi".to_string(),
            market_id: "m1".to_string(),
            side: "yes".to_string(),
            buy_price: 0.50,
            available_size: 100.0,
            spread: 0.02,
        }];
        let config = ExecutionModelConfig {
            queue_decay_half_life_seconds: 5.0,
            latency_seconds: 0.2,
            market_impact_factor: 0.01,
            max_market_impact: 0.5,
            min_fill_fraction: 0.1,
            fill_fraction_steps: 5,
            sequential_leg_delay_seconds: 1.0,
            enable_queue_decay: true,
            enable_market_impact: true,
        };
        let r = simulate_execution(&legs, 10, 0.0, false, &config);
        assert!(r.all_fill_probability > 0.0);
        assert!(!r.legs.is_empty());
    }
}
