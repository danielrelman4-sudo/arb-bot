//! Rust port of `arb_bot/strategy.py` — 5-lane ArbitrageFinder.
//!
//! ## Design
//!
//! Python owns the configuration, mappings, structural rules, and
//! bucket-quality state. On each call it serialises them as JSON
//! (serde_json) and passes them over the PyO3 boundary as `&str`.
//! The Rust side deserialises, runs all 5 detection lanes, and
//! returns the opportunities as a JSON string.
//!
//! ## Performance
//!
//! The O(n²) fuzzy cross-venue loop (tokenise + Jaccard) is the
//! dominant cost. Rust HashSet operations and tight iteration give
//! a large speedup here.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct FinderConfig {
    pub min_net_edge_per_contract: f64,
    pub enable_cross_venue: bool,
    pub cross_venue_min_match_score: f64,
    pub cross_venue_mapping_required: bool,
    pub enable_fuzzy_cross_venue_fallback: bool,
    pub enable_maker_estimates: bool,
    pub enable_structural_arb: bool,
    /// Which opportunity kinds to detect. Null/empty → all.
    pub selected_kinds: Option<Vec<String>>,
    /// Override threshold (from find_by_kind). None → use default.
    pub min_net_edge_override: Option<f64>,
}

// ---------------------------------------------------------------------------
// Input types (deserialized from Python JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct QuoteInput {
    pub venue: String,
    pub market_id: String,
    pub yes_buy_price: f64,
    pub no_buy_price: f64,
    pub yes_buy_size: f64,
    pub no_buy_size: f64,
    pub yes_bid_price: Option<f64>,
    pub no_bid_price: Option<f64>,
    pub yes_bid_size: f64,
    pub no_bid_size: f64,
    pub yes_maker_buy_price: Option<f64>,
    pub no_maker_buy_price: Option<f64>,
    pub yes_maker_buy_size: f64,
    pub no_maker_buy_size: f64,
    pub fee_per_contract: f64,
    /// ISO-formatted timestamp string.
    pub observed_at: String,
    pub market_text: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VenueRefInput {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MappingInput {
    pub group_id: String,
    pub kalshi: VenueRefInput,
    pub polymarket: VenueRefInput,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarketLegRefInput {
    pub venue: String,
    pub market_id: String,
    pub side: String, // "yes" or "no"
}

#[derive(Debug, Clone, Deserialize)]
pub struct BucketRuleInput {
    pub group_id: String,
    pub legs: Vec<MarketLegRefInput>,
    pub payout_per_contract: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventTreeRuleInput {
    pub group_id: String,
    pub parent: MarketLegRefInput,
    pub children: Vec<MarketLegRefInput>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParityRuleInput {
    pub group_id: String,
    pub left: MarketLegRefInput,
    pub right: MarketLegRefInput,
    pub relationship: String, // "equivalent" or "complement"
}

#[derive(Debug, Clone, Deserialize)]
pub struct RulesInput {
    #[serde(default)]
    pub buckets: Vec<BucketRuleInput>,
    #[serde(default)]
    pub event_trees: Vec<EventTreeRuleInput>,
    #[serde(default)]
    pub parity_checks: Vec<ParityRuleInput>,
}

// ---------------------------------------------------------------------------
// Output types (serialized to Python JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct LegOutput {
    pub venue: String,
    pub market_id: String,
    pub side: String,
    pub buy_price: f64,
    pub buy_size: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpportunityOutput {
    pub kind: String,
    pub execution_style: String,
    pub legs: Vec<LegOutput>,
    pub gross_edge_per_contract: f64,
    pub net_edge_per_contract: f64,
    pub fee_per_contract: f64,
    pub observed_at: String,
    pub match_key: String,
    pub match_score: f64,
    pub payout_per_contract: f64,
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Stopwords (matching Python _STOPWORDS)
// ---------------------------------------------------------------------------

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "will", "with",
];

// ---------------------------------------------------------------------------
// Text processing
// ---------------------------------------------------------------------------

fn normalize_text(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut result = String::with_capacity(lower.len());
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
        } else {
            result.push(' ');
        }
    }
    // Collapse whitespace.
    result.split_whitespace().collect::<Vec<&str>>().join(" ")
}

fn tokenize(text: &str) -> HashSet<String> {
    let normalized = normalize_text(text);
    let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();
    normalized
        .split_whitespace()
        .filter(|t| !t.is_empty() && !stopwords.contains(t))
        .filter(|t| t.len() > 1 || t.chars().next().map_or(false, |c| c.is_ascii_digit()))
        .map(|t| t.to_string())
        .collect()
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

// ---------------------------------------------------------------------------
// Internal leg/opportunity building
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Yes,
    No,
}

impl Side {
    fn from_str(s: &str) -> Option<Side> {
        match s {
            "yes" => Some(Side::Yes),
            "no" => Some(Side::No),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Side::Yes => "yes",
            Side::No => "no",
        }
    }

    fn flip(&self) -> Side {
        match self {
            Side::Yes => Side::No,
            Side::No => Side::Yes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionStyle {
    Taker,
    MakerEstimate,
}

impl ExecutionStyle {
    fn as_str(&self) -> &'static str {
        match self {
            ExecutionStyle::Taker => "taker",
            ExecutionStyle::MakerEstimate => "maker_estimate",
        }
    }
}

struct Leg {
    venue: String,
    market_id: String,
    side: Side,
    buy_price: f64,
    buy_size: f64,
    metadata: HashMap<String, String>,
}

fn leg_from_quote(quote: &QuoteInput, side: Side, style: ExecutionStyle) -> Option<Leg> {
    let (price_opt, size) = match (style, side) {
        (ExecutionStyle::Taker, Side::Yes) => (Some(quote.yes_buy_price), quote.yes_buy_size),
        (ExecutionStyle::Taker, Side::No) => (Some(quote.no_buy_price), quote.no_buy_size),
        (ExecutionStyle::MakerEstimate, Side::Yes) => (quote.yes_maker_buy_price, quote.yes_maker_buy_size),
        (ExecutionStyle::MakerEstimate, Side::No) => (quote.no_maker_buy_price, quote.no_maker_buy_size),
    };

    let price = price_opt?;
    if price < 0.0 || price > 1.0 {
        return None;
    }
    if size <= 0.0 {
        return None;
    }

    Some(Leg {
        venue: quote.venue.clone(),
        market_id: quote.market_id.clone(),
        side,
        buy_price: price,
        buy_size: size,
        metadata: quote.metadata.clone(),
    })
}

fn build_opportunity_from_legs(
    kind: &str,
    style: ExecutionStyle,
    legs: &[Leg],
    fee_per_contract: f64,
    payout_per_contract: f64,
    observed_at: &str,
    match_key: &str,
    match_score: f64,
    metadata: HashMap<String, String>,
    min_threshold: f64,
) -> Option<OpportunityOutput> {
    if legs.len() < 2 {
        return None;
    }

    let total_cost: f64 = legs.iter().map(|l| l.buy_price).sum();
    let gross_edge = payout_per_contract - total_cost;
    let net_edge = gross_edge - fee_per_contract;

    if net_edge < min_threshold {
        return None;
    }

    let leg_outputs: Vec<LegOutput> = legs
        .iter()
        .map(|l| LegOutput {
            venue: l.venue.clone(),
            market_id: l.market_id.clone(),
            side: l.side.as_str().to_string(),
            buy_price: l.buy_price,
            buy_size: l.buy_size,
            metadata: l.metadata.clone(),
        })
        .collect();

    Some(OpportunityOutput {
        kind: kind.to_string(),
        execution_style: style.as_str().to_string(),
        legs: leg_outputs,
        gross_edge_per_contract: gross_edge,
        net_edge_per_contract: net_edge,
        fee_per_contract,
        observed_at: observed_at.to_string(),
        match_key: match_key.to_string(),
        match_score,
        payout_per_contract,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Lane 1: Intra-venue
// ---------------------------------------------------------------------------

fn find_intra_venue(
    quotes: &[QuoteInput],
    min_threshold: f64,
    styles: &[ExecutionStyle],
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    for quote in quotes {
        for &style in styles {
            let yes_leg = match leg_from_quote(quote, Side::Yes, style) {
                Some(l) => l,
                None => continue,
            };
            let no_leg = match leg_from_quote(quote, Side::No, style) {
                Some(l) => l,
                None => continue,
            };

            let match_key = format!("{}:{}", quote.venue, quote.market_id);
            let mut metadata = HashMap::new();
            metadata.insert("market_text".to_string(), quote.market_text.clone());

            if let Some(opp) = build_opportunity_from_legs(
                "intra_venue",
                style,
                &[yes_leg, no_leg],
                quote.fee_per_contract,
                1.0,
                &quote.observed_at,
                &match_key,
                1.0,
                metadata,
                min_threshold,
            ) {
                results.push(opp);
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Lane 2: Cross-venue (mappings + fuzzy)
// ---------------------------------------------------------------------------

fn quote_matches_ref(quote: &QuoteInput, mapping_ref: &VenueRefInput, target: &str) -> bool {
    let key = mapping_ref.key.trim().to_lowercase();

    if key == "kalshi_market_id"
        || key == "kalshi_ticker"
        || key == "polymarket_market_id"
        || key == "polymarket_condition_id"
    {
        return quote.market_id.trim().to_lowercase() == target;
    }

    if key == "kalshi_event_ticker" {
        let val = quote
            .metadata
            .get("event_ticker")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        return val == target;
    }

    if key == "polymarket_slug" {
        let val = quote
            .metadata
            .get("slug")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        return val == target;
    }

    false
}

fn lookup_mapped_quote<'a>(
    venue: &str,
    mapping_ref: &VenueRefInput,
    venue_quotes: &[&'a QuoteInput],
) -> Option<&'a QuoteInput> {
    let target = mapping_ref.value.trim().to_lowercase();
    if target.is_empty() {
        return None;
    }

    venue_quotes
        .iter()
        .find(|q| q.venue == venue && quote_matches_ref(q, mapping_ref, &target))
        .copied()
}

fn cross_pair_opportunities(
    left_quote: &QuoteInput,
    right_quote: &QuoteInput,
    match_key: &str,
    match_score: f64,
    min_threshold: f64,
    styles: &[ExecutionStyle],
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    let observed_at = if left_quote.observed_at > right_quote.observed_at {
        &left_quote.observed_at
    } else {
        &right_quote.observed_at
    };
    let pair_fee = left_quote.fee_per_contract + right_quote.fee_per_contract;

    for &style in styles {
        let yes_left = match leg_from_quote(left_quote, Side::Yes, style) {
            Some(l) => l,
            None => continue,
        };
        let no_left = match leg_from_quote(left_quote, Side::No, style) {
            Some(l) => l,
            None => continue,
        };
        let yes_right = match leg_from_quote(right_quote, Side::Yes, style) {
            Some(l) => l,
            None => continue,
        };
        let no_right = match leg_from_quote(right_quote, Side::No, style) {
            Some(l) => l,
            None => continue,
        };

        let mut metadata = HashMap::new();
        metadata.insert("left_text".to_string(), left_quote.market_text.clone());
        metadata.insert("right_text".to_string(), right_quote.market_text.clone());

        // First: left YES + right NO
        if let Some(opp) = build_opportunity_from_legs(
            "cross_venue",
            style,
            &[yes_left, no_right],
            pair_fee,
            1.0,
            observed_at,
            match_key,
            match_score,
            metadata.clone(),
            min_threshold,
        ) {
            results.push(opp);
        }

        // Second: left NO + right YES
        if let Some(opp) = build_opportunity_from_legs(
            "cross_venue",
            style,
            &[no_left, yes_right],
            pair_fee,
            1.0,
            observed_at,
            match_key,
            match_score,
            metadata,
            min_threshold,
        ) {
            results.push(opp);
        }
    }

    results
}

fn find_cross_venue_from_mappings(
    quotes: &[QuoteInput],
    mappings: &[MappingInput],
    min_threshold: f64,
    styles: &[ExecutionStyle],
) -> Vec<OpportunityOutput> {
    // Group quotes by venue.
    let mut by_venue: HashMap<&str, Vec<&QuoteInput>> = HashMap::new();
    for q in quotes {
        by_venue.entry(q.venue.as_str()).or_default().push(q);
    }

    let kalshi_quotes = by_venue.get("kalshi").map(|v| v.as_slice()).unwrap_or(&[]);
    let poly_quotes = by_venue.get("polymarket").map(|v| v.as_slice()).unwrap_or(&[]);

    let mut results = Vec::new();

    for mapping in mappings {
        let kalshi_quote = match lookup_mapped_quote("kalshi", &mapping.kalshi, kalshi_quotes) {
            Some(q) => q,
            None => continue,
        };
        let poly_quote = match lookup_mapped_quote("polymarket", &mapping.polymarket, poly_quotes) {
            Some(q) => q,
            None => continue,
        };

        results.extend(cross_pair_opportunities(
            kalshi_quote,
            poly_quote,
            &mapping.group_id,
            1.0,
            min_threshold,
            styles,
        ));
    }

    results
}

fn find_cross_venue_fuzzy(
    quotes: &[QuoteInput],
    min_match_score: f64,
    min_threshold: f64,
    styles: &[ExecutionStyle],
) -> Vec<OpportunityOutput> {
    // Tokenize all quotes.
    struct Descriptor<'a> {
        quote: &'a QuoteInput,
        text: String,
        tokens: HashSet<String>,
    }

    let descriptors: Vec<Descriptor> = quotes
        .iter()
        .filter_map(|q| {
            let tokens = tokenize(&q.market_text);
            if tokens.is_empty() {
                return None;
            }
            Some(Descriptor {
                quote: q,
                text: q.market_text.clone(),
                tokens,
            })
        })
        .collect();

    let mut results = Vec::new();
    let n = descriptors.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let left = &descriptors[i];
            let right = &descriptors[j];

            if left.quote.venue == right.quote.venue {
                continue;
            }

            let score = jaccard_similarity(&left.tokens, &right.tokens);
            if score < min_match_score {
                continue;
            }

            let norm_left = normalize_text(&left.text);
            let norm_right = normalize_text(&right.text);
            let match_key = if !norm_left.is_empty() {
                norm_left
            } else if !norm_right.is_empty() {
                norm_right
            } else {
                format!("{}:{}", left.quote.market_id, right.quote.market_id)
            };

            results.extend(cross_pair_opportunities(
                left.quote,
                right.quote,
                &match_key,
                score,
                min_threshold,
                styles,
            ));
        }
    }

    results
}

fn find_cross_venue(
    quotes: &[QuoteInput],
    config: &FinderConfig,
    mappings: &[MappingInput],
    min_threshold: f64,
    styles: &[ExecutionStyle],
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    if !mappings.is_empty() {
        results.extend(find_cross_venue_from_mappings(
            quotes,
            mappings,
            min_threshold,
            styles,
        ));
        if config.cross_venue_mapping_required {
            return results;
        }
    }

    if config.enable_fuzzy_cross_venue_fallback {
        results.extend(find_cross_venue_fuzzy(
            quotes,
            config.cross_venue_min_match_score,
            min_threshold,
            styles,
        ));
    }

    results
}

// ---------------------------------------------------------------------------
// Lanes 3-5: Structural (buckets, event trees, parity)
// ---------------------------------------------------------------------------

type QuoteLookup<'a> = HashMap<(String, String), &'a QuoteInput>;

fn flip_ref(r: &MarketLegRefInput) -> MarketLegRefInput {
    let flipped = if r.side == "yes" { "no" } else { "yes" };
    MarketLegRefInput {
        venue: r.venue.clone(),
        market_id: r.market_id.clone(),
        side: flipped.to_string(),
    }
}

fn resolve_refs<'a>(
    refs: &[MarketLegRefInput],
    lookup: &QuoteLookup<'a>,
    style: ExecutionStyle,
) -> Option<(Vec<Leg>, Vec<&'a QuoteInput>)> {
    let mut legs = Vec::with_capacity(refs.len());
    let mut source_quotes = Vec::with_capacity(refs.len());

    for r in refs {
        let key = (r.venue.to_lowercase(), r.market_id.to_lowercase());
        let quote = lookup.get(&key)?;
        let side = Side::from_str(&r.side)?;
        let leg = leg_from_quote(quote, side, style)?;
        legs.push(leg);
        source_quotes.push(*quote);
    }

    Some((legs, source_quotes))
}

fn max_observed_at<'a>(quotes: &'a [&'a QuoteInput]) -> &'a str {
    quotes
        .iter()
        .map(|q| q.observed_at.as_str())
        .max()
        .unwrap_or("")
}

fn total_fees(quotes: &[&QuoteInput]) -> f64 {
    quotes.iter().map(|q| q.fee_per_contract).sum()
}

fn find_structural_buckets(
    lookup: &QuoteLookup,
    rules: &[BucketRuleInput],
    style: ExecutionStyle,
    min_threshold: f64,
    enabled_bucket_ids: &HashSet<String>,
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    for bucket in rules {
        if !enabled_bucket_ids.is_empty() && !enabled_bucket_ids.contains(&bucket.group_id) {
            continue;
        }

        let (legs, source_quotes) = match resolve_refs(&bucket.legs, lookup, style) {
            Some(r) => r,
            None => continue,
        };

        let observed_at = max_observed_at(&source_quotes);
        let fee = total_fees(&source_quotes);

        let mut metadata = HashMap::new();
        metadata.insert(
            "structural_class".to_string(),
            "mutually_exclusive_bucket".to_string(),
        );
        metadata.insert("bucket_group_id".to_string(), bucket.group_id.clone());

        if let Some(opp) = build_opportunity_from_legs(
            "structural_bucket",
            style,
            &legs,
            fee,
            bucket.payout_per_contract,
            observed_at,
            &bucket.group_id,
            1.0,
            metadata,
            min_threshold,
        ) {
            results.push(opp);
        }
    }

    results
}

fn find_structural_event_trees(
    lookup: &QuoteLookup,
    rules: &[EventTreeRuleInput],
    style: ExecutionStyle,
    min_threshold: f64,
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    for rule in rules {
        // Basket 1: parent=NO, children=YES
        {
            let mut refs = vec![flip_ref(&rule.parent)];
            refs.extend(rule.children.iter().cloned());

            if let Some((legs, source_quotes)) = resolve_refs(&refs, lookup, style) {
                let observed_at = max_observed_at(&source_quotes);
                let fee = total_fees(&source_quotes);
                let match_key = format!("{}:parent_no_children_yes", rule.group_id);

                let mut metadata = HashMap::new();
                metadata.insert(
                    "structural_class".to_string(),
                    "event_tree_parent_no_children_yes".to_string(),
                );

                if let Some(opp) = build_opportunity_from_legs(
                    "structural_event_tree",
                    style,
                    &legs,
                    fee,
                    1.0,
                    observed_at,
                    &match_key,
                    1.0,
                    metadata,
                    min_threshold,
                ) {
                    results.push(opp);
                }
            }
        }

        // Basket 2: parent=YES, children=NO
        {
            let mut refs = vec![rule.parent.clone()];
            refs.extend(rule.children.iter().map(flip_ref));

            if let Some((legs, source_quotes)) = resolve_refs(&refs, lookup, style) {
                let observed_at = max_observed_at(&source_quotes);
                let fee = total_fees(&source_quotes);
                let payout = rule.children.len() as f64;
                let match_key = format!("{}:parent_yes_children_no", rule.group_id);

                let mut metadata = HashMap::new();
                metadata.insert(
                    "structural_class".to_string(),
                    "event_tree_parent_yes_children_no".to_string(),
                );

                if let Some(opp) = build_opportunity_from_legs(
                    "structural_event_tree",
                    style,
                    &legs,
                    fee,
                    payout,
                    observed_at,
                    &match_key,
                    1.0,
                    metadata,
                    min_threshold,
                ) {
                    results.push(opp);
                }
            }
        }
    }

    results
}

fn find_structural_parity(
    lookup: &QuoteLookup,
    rules: &[ParityRuleInput],
    style: ExecutionStyle,
    min_threshold: f64,
) -> Vec<OpportunityOutput> {
    let mut results = Vec::new();

    for rule in rules {
        let pairings: Vec<(MarketLegRefInput, MarketLegRefInput, &str)> =
            if rule.relationship == "complement" {
                vec![
                    (rule.left.clone(), rule.right.clone(), "left_yes_right_yes"),
                    (flip_ref(&rule.left), flip_ref(&rule.right), "left_no_right_no"),
                ]
            } else {
                vec![
                    (rule.left.clone(), flip_ref(&rule.right), "left_yes_right_no"),
                    (flip_ref(&rule.left), rule.right.clone(), "left_no_right_yes"),
                ]
            };

        for (left_ref, right_ref, suffix) in &pairings {
            let refs = [left_ref.clone(), right_ref.clone()];
            if let Some((legs, source_quotes)) = resolve_refs(&refs, lookup, style) {
                let observed_at = max_observed_at(&source_quotes);
                let fee = total_fees(&source_quotes);
                let match_key = format!("{}:{}", rule.group_id, suffix);

                let mut metadata = HashMap::new();
                metadata.insert(
                    "structural_class".to_string(),
                    format!("parity_{}", rule.relationship),
                );

                if let Some(opp) = build_opportunity_from_legs(
                    "structural_parity",
                    style,
                    &legs,
                    fee,
                    1.0,
                    observed_at,
                    &match_key,
                    1.0,
                    metadata,
                    min_threshold,
                ) {
                    results.push(opp);
                }
            }
        }
    }

    results
}

fn find_structural(
    quotes: &[QuoteInput],
    rules: &RulesInput,
    min_threshold: f64,
    styles: &[ExecutionStyle],
    enabled_bucket_ids: &HashSet<String>,
) -> Vec<OpportunityOutput> {
    // Build lookup.
    let mut lookup: QuoteLookup = HashMap::new();
    for q in quotes {
        let key = (q.venue.to_lowercase(), q.market_id.to_lowercase());
        lookup.insert(key, q);
    }

    let mut results = Vec::new();

    for &style in styles {
        results.extend(find_structural_buckets(
            &lookup,
            &rules.buckets,
            style,
            min_threshold,
            enabled_bucket_ids,
        ));
        results.extend(find_structural_event_trees(
            &lookup,
            &rules.event_trees,
            style,
            min_threshold,
        ));
        results.extend(find_structural_parity(
            &lookup,
            &rules.parity_checks,
            style,
            min_threshold,
        ));
    }

    results
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Core detection function — called from Python.
pub fn find_opportunities(
    quotes: &[QuoteInput],
    config: &FinderConfig,
    rules: &RulesInput,
    mappings: &[MappingInput],
    enabled_bucket_ids: &HashSet<String>,
) -> Vec<OpportunityOutput> {
    let threshold = config
        .min_net_edge_override
        .unwrap_or(config.min_net_edge_per_contract);

    let selected_kinds: HashSet<String> = match &config.selected_kinds {
        Some(kinds) if !kinds.is_empty() => kinds.iter().cloned().collect(),
        _ => {
            let mut all = HashSet::new();
            all.insert("intra_venue".to_string());
            all.insert("cross_venue".to_string());
            all.insert("structural_bucket".to_string());
            all.insert("structural_event_tree".to_string());
            all.insert("structural_parity".to_string());
            all
        }
    };

    let styles: Vec<ExecutionStyle> = if config.enable_maker_estimates {
        vec![ExecutionStyle::Taker, ExecutionStyle::MakerEstimate]
    } else {
        vec![ExecutionStyle::Taker]
    };

    let mut opportunities = Vec::new();

    // Lane 1: Intra-venue
    if selected_kinds.contains("intra_venue") {
        opportunities.extend(find_intra_venue(quotes, threshold, &styles));
    }

    // Lane 2: Cross-venue
    if selected_kinds.contains("cross_venue") && config.enable_cross_venue {
        opportunities.extend(find_cross_venue(
            quotes, config, mappings, threshold, &styles,
        ));
    }

    // Lanes 3-5: Structural
    let structural_kinds: HashSet<&str> = [
        "structural_bucket",
        "structural_event_tree",
        "structural_parity",
    ]
    .iter()
    .copied()
    .collect();
    let has_structural = selected_kinds.iter().any(|k| structural_kinds.contains(k.as_str()));
    let rules_not_empty =
        !rules.buckets.is_empty() || !rules.event_trees.is_empty() || !rules.parity_checks.is_empty();

    if has_structural && config.enable_structural_arb && rules_not_empty {
        let structural = find_structural(quotes, rules, threshold, &styles, enabled_bucket_ids);
        // Filter by selected kinds.
        opportunities.extend(
            structural
                .into_iter()
                .filter(|opp| selected_kinds.contains(&opp.kind)),
        );
    }

    // Sort: net_edge DESC, match_score DESC, gross_edge DESC.
    opportunities.sort_by(|a, b| {
        b.net_edge_per_contract
            .partial_cmp(&a.net_edge_per_contract)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.match_score
                    .partial_cmp(&a.match_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.gross_edge_per_contract
                    .partial_cmp(&a.gross_edge_per_contract)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    opportunities
}

// ---------------------------------------------------------------------------
// PyO3 boundary
// ---------------------------------------------------------------------------

/// Register strategy PyO3 functions on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_opportunities_py, m)?)?;
    Ok(())
}

/// PyO3 wrapper: JSON-in, JSON-out.
///
/// Parameters:
///   quotes_json:  JSON array of QuoteInput
///   config_json:  JSON object of FinderConfig
///   rules_json:   JSON object of RulesInput
///   mappings_json: JSON array of MappingInput
///   enabled_bucket_ids_json: JSON array of strings (enabled bucket group_ids)
///
/// Returns: JSON string — array of OpportunityOutput.
#[pyfunction]
#[pyo3(name = "find_opportunities", signature = (quotes_json, config_json, rules_json, mappings_json, enabled_bucket_ids_json))]
pub fn find_opportunities_py(
    quotes_json: &str,
    config_json: &str,
    rules_json: &str,
    mappings_json: &str,
    enabled_bucket_ids_json: &str,
) -> PyResult<String> {
    let quotes: Vec<QuoteInput> = serde_json::from_str(quotes_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quotes JSON error: {}", e)))?;
    let config: FinderConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("config JSON error: {}", e)))?;
    let rules: RulesInput = serde_json::from_str(rules_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("rules JSON error: {}", e)))?;
    let mappings: Vec<MappingInput> = serde_json::from_str(mappings_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("mappings JSON error: {}", e)))?;
    let enabled_ids: Vec<String> = serde_json::from_str(enabled_bucket_ids_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("enabled_bucket_ids JSON error: {}", e)))?;

    let enabled_set: HashSet<String> = enabled_ids.into_iter().collect();

    let opportunities = find_opportunities(&quotes, &config, &rules, &mappings, &enabled_set);

    let result = serde_json::to_string(&opportunities)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialization error: {}", e)))?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Rust unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("Hello, World! 123"), "hello world 123");
        assert_eq!(normalize_text("BTC-USD $50K"), "btc usd 50k");
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Will BTC exceed 50K by March?");
        assert!(tokens.contains("btc"));
        assert!(tokens.contains("50k"));
        assert!(tokens.contains("march"));
        assert!(tokens.contains("exceed"));
        // "will" is a stopword
        assert!(!tokens.contains("will"));
        // "by" is a stopword
        assert!(!tokens.contains("by"));
    }

    #[test]
    fn test_tokenize_single_digit_kept() {
        let tokens = tokenize("option 1 2 3");
        assert!(tokens.contains("1"));
        assert!(tokens.contains("2"));
        assert!(tokens.contains("3"));
        assert!(tokens.contains("option"));
    }

    #[test]
    fn test_tokenize_single_letter_removed() {
        let tokens = tokenize("I am x y z test");
        // single letters that are not digits get removed
        assert!(!tokens.contains("x"));
        assert!(!tokens.contains("y"));
        assert!(!tokens.contains("z"));
        assert!(tokens.contains("am"));
        assert!(tokens.contains("test"));
    }

    #[test]
    fn test_jaccard_identical() {
        let s: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&s, &s) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_jaccard_partial() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_jaccard_empty() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_side_flip() {
        assert_eq!(Side::Yes.flip(), Side::No);
        assert_eq!(Side::No.flip(), Side::Yes);
    }

    fn make_quote(venue: &str, market_id: &str, yes_buy: f64, no_buy: f64) -> QuoteInput {
        QuoteInput {
            venue: venue.to_string(),
            market_id: market_id.to_string(),
            yes_buy_price: yes_buy,
            no_buy_price: no_buy,
            yes_buy_size: 100.0,
            no_buy_size: 100.0,
            yes_bid_price: None,
            no_bid_price: None,
            yes_bid_size: 0.0,
            no_bid_size: 0.0,
            yes_maker_buy_price: None,
            no_maker_buy_price: None,
            yes_maker_buy_size: 0.0,
            no_maker_buy_size: 0.0,
            fee_per_contract: 0.0,
            observed_at: "2024-01-01T00:00:00+00:00".to_string(),
            market_text: format!("{} {}", venue, market_id),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_intra_venue_arb() {
        let quotes = vec![make_quote("kalshi", "m1", 0.40, 0.40)]; // cost = 0.80, edge = 0.20
        let styles = vec![ExecutionStyle::Taker];
        let results = find_intra_venue(&quotes, 0.01, &styles);
        assert_eq!(results.len(), 1);
        assert!((results[0].net_edge_per_contract - 0.20).abs() < 1e-12);
    }

    #[test]
    fn test_intra_venue_no_arb() {
        let quotes = vec![make_quote("kalshi", "m1", 0.55, 0.55)]; // cost = 1.10, edge = -0.10
        let styles = vec![ExecutionStyle::Taker];
        let results = find_intra_venue(&quotes, 0.01, &styles);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fuzzy_cross_venue() {
        let q1 = QuoteInput {
            market_text: "Will BTC exceed 50K by March 2024".to_string(),
            ..make_quote("kalshi", "btc_50k", 0.40, 0.60)
        };
        let q2 = QuoteInput {
            market_text: "Bitcoin exceed 50K March 2024 end".to_string(),
            ..make_quote("polymarket", "btc50k", 0.45, 0.35)
        };
        let quotes = vec![q1, q2];
        let styles = vec![ExecutionStyle::Taker];

        let results = find_cross_venue_fuzzy(&quotes, 0.2, 0.01, &styles);
        assert!(!results.is_empty());
        assert_eq!(results[0].kind, "cross_venue");
    }

    #[test]
    fn test_structural_bucket() {
        let q1 = make_quote("kalshi", "m1", 0.30, 0.70);
        let q2 = make_quote("kalshi", "m2", 0.30, 0.70);
        let q3 = make_quote("kalshi", "m3", 0.30, 0.70);
        let quotes = vec![q1, q2, q3];

        let rules = vec![BucketRuleInput {
            group_id: "bucket1".to_string(),
            legs: vec![
                MarketLegRefInput { venue: "kalshi".to_string(), market_id: "m1".to_string(), side: "yes".to_string() },
                MarketLegRefInput { venue: "kalshi".to_string(), market_id: "m2".to_string(), side: "yes".to_string() },
                MarketLegRefInput { venue: "kalshi".to_string(), market_id: "m3".to_string(), side: "yes".to_string() },
            ],
            payout_per_contract: 1.0,
        }];

        let mut lookup: QuoteLookup = HashMap::new();
        for q in &quotes {
            lookup.insert((q.venue.to_lowercase(), q.market_id.to_lowercase()), q);
        }

        let enabled: HashSet<String> = HashSet::new(); // empty = allow all
        let results = find_structural_buckets(&lookup, &rules, ExecutionStyle::Taker, 0.01, &enabled);
        assert_eq!(results.len(), 1);
        // cost = 0.30 * 3 = 0.90, payout = 1.0, edge = 0.10
        assert!((results[0].net_edge_per_contract - 0.10).abs() < 1e-12);
    }

    #[test]
    fn test_find_opportunities_sorting() {
        let q1 = make_quote("kalshi", "m1", 0.40, 0.40); // edge = 0.20
        let q2 = make_quote("kalshi", "m2", 0.45, 0.45); // edge = 0.10

        let config = FinderConfig {
            min_net_edge_per_contract: 0.01,
            enable_cross_venue: false,
            cross_venue_min_match_score: 0.62,
            cross_venue_mapping_required: false,
            enable_fuzzy_cross_venue_fallback: false,
            enable_maker_estimates: false,
            enable_structural_arb: false,
            selected_kinds: None,
            min_net_edge_override: None,
        };

        let rules = RulesInput {
            buckets: vec![],
            event_trees: vec![],
            parity_checks: vec![],
        };

        let enabled: HashSet<String> = HashSet::new();
        let results = find_opportunities(&[q1, q2], &config, &rules, &[], &enabled);
        assert_eq!(results.len(), 2);
        // Sorted by net_edge DESC.
        assert!(results[0].net_edge_per_contract >= results[1].net_edge_per_contract);
    }
}
