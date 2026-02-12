from __future__ import annotations

import csv
import glob
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_sort_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class BucketAccumulator:
    detections: int = 0
    opened: int = 0
    settled: int = 0
    positive_outcomes: int = 0
    detected_edge_sum: float = 0.0
    fill_probability_sum: float = 0.0
    expected_realized_profit_sum: float = 0.0
    realized_profit_sum: float = 0.0
    slippage_sum: float = 0.0
    latency_sum_ms: float = 0.0

    def average_detected_edge(self) -> float:
        if self.detections <= 0:
            return 0.0
        return self.detected_edge_sum / self.detections

    def average_fill_probability(self) -> float:
        if self.detections <= 0:
            return 0.0
        return self.fill_probability_sum / self.detections

    def open_rate(self) -> float:
        if self.detections <= 0:
            return 0.0
        return self.opened / self.detections

    def positive_outcome_rate(self) -> float:
        if self.opened <= 0:
            return 0.0
        return self.positive_outcomes / self.opened

    def average_slippage(self) -> float:
        if self.opened <= 0:
            return 0.0
        return self.slippage_sum / self.opened

    def average_latency_ms(self) -> float:
        if self.opened <= 0:
            return 0.0
        return self.latency_sum_ms / self.opened

    def posterior_expected_realized_profit(
        self,
        prior_mean: float,
        prior_strength: float,
    ) -> float:
        if self.opened <= 0:
            return prior_mean
        return (
            self.realized_profit_sum + prior_mean * prior_strength
        ) / (self.opened + prior_strength)


@dataclass(frozen=True)
class BucketScore:
    group_id: str
    score: float
    detections: int
    opened: int
    posterior_expected_realized_profit: float
    average_detected_edge: float
    average_fill_probability: float
    open_rate: float
    positive_outcome_rate: float


class BucketQualityModel:
    """Ranks structural bucket rules via Thompson Sampling.

    Instead of a deterministic explore/exploit split, each bucket
    maintains a Beta(alpha, beta) posterior over its success
    probability.  At each recompute we sample from each bucket's
    posterior and enable the top-k by sampled value.  This naturally
    balances exploration (uncertain buckets have high-variance samples)
    and exploitation (proven buckets have high-mean samples).

    Reference: Agrawal & Goyal (2012), "Analysis of Thompson Sampling
    for the Multi-armed Bandit Problem", COLT.
    """

    def __init__(
        self,
        bucket_leg_counts: dict[str, int],
        enabled: bool = True,
        history_glob: str | None = None,
        history_max_files: int = 50,
        min_observations: int = 8,
        max_active_buckets: int = 0,
        explore_fraction: float = 0.15,
        prior_mean_realized_profit: float = 0.002,
        prior_strength: float = 12.0,
        min_score: float = -0.02,
        leg_count_penalty: float = 0.00025,
        live_update_interval: int = 25,
        thompson_prior_alpha: float = 1.0,
        thompson_prior_beta: float = 1.0,
        use_thompson_sampling: bool = True,
    ) -> None:
        self._enabled = enabled
        self._bucket_leg_counts = dict(bucket_leg_counts)
        self._history_glob = history_glob or ""
        self._history_max_files = max(1, history_max_files)
        self._min_observations = max(1, min_observations)
        self._max_active_buckets = max(0, max_active_buckets)
        self._explore_fraction = min(0.95, max(0.0, explore_fraction))
        self._prior_mean_realized_profit = prior_mean_realized_profit
        self._prior_strength = max(0.1, prior_strength)
        self._min_score = min_score
        self._leg_count_penalty = max(0.0, leg_count_penalty)
        self._live_update_interval = max(1, live_update_interval)
        self._live_updates_since_recompute = 0
        self._thompson_prior_alpha = max(0.01, thompson_prior_alpha)
        self._thompson_prior_beta = max(0.01, thompson_prior_beta)
        self._use_thompson_sampling = use_thompson_sampling

        self._accumulators: dict[str, BucketAccumulator] = {}
        self._scores: dict[str, BucketScore] = {}
        self._active_bucket_ids: set[str] = set(self._bucket_leg_counts.keys())
        self._history_files_loaded = 0
        self._history_rows_loaded = 0

        if not self._enabled:
            return

        self._load_history()
        self._recompute_active_set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active_bucket_ids(self) -> set[str]:
        return set(self._active_bucket_ids)

    @property
    def history_files_loaded(self) -> int:
        return self._history_files_loaded

    @property
    def history_rows_loaded(self) -> int:
        return self._history_rows_loaded

    @property
    def score_count(self) -> int:
        return len(self._scores)

    def score_for(self, group_id: str) -> float | None:
        score = self._scores.get(group_id)
        return score.score if score is not None else None

    def should_enable_bucket(self, group_id: str) -> bool:
        if not self._enabled:
            return True
        if not self._active_bucket_ids:
            return True
        return group_id in self._active_bucket_ids

    def observe_decision(
        self,
        *,
        group_id: str,
        action: str,
        detected_edge_per_contract: float,
        fill_probability: float,
        expected_realized_profit: float,
        realized_profit: float | None,
        expected_slippage_per_contract: float,
        execution_latency_ms: float | None,
    ) -> None:
        if not self._enabled:
            return
        if group_id not in self._bucket_leg_counts:
            return
        self._ingest_bucket_observation(
            group_id=group_id,
            action=action,
            detected_edge_per_contract=detected_edge_per_contract,
            fill_probability=fill_probability,
            expected_realized_profit=expected_realized_profit,
            realized_profit=realized_profit,
            expected_slippage_per_contract=expected_slippage_per_contract,
            execution_latency_ms=execution_latency_ms,
        )
        self._live_updates_since_recompute += 1
        if self._live_updates_since_recompute >= self._live_update_interval:
            self._live_updates_since_recompute = 0
            self._recompute_active_set()

    def summary(self) -> str:
        total = len(self._bucket_leg_counts)
        active = len(self._active_bucket_ids)
        scored = len(self._scores)
        return (
            f"bucket_quality enabled={self._enabled} active={active}/{total} scored={scored} "
            f"history_files={self._history_files_loaded} history_rows={self._history_rows_loaded}"
        )

    def _load_history(self) -> None:
        files = self._resolve_history_files()
        self._history_files_loaded = len(files)
        rows_loaded = 0
        for path in files:
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("kind") or "").strip().lower() != "structural_bucket":
                            continue
                        group_id = str(row.get("match_key") or "").strip()
                        if not group_id:
                            continue
                        self._ingest_row_dict(group_id, row)
                        rows_loaded += 1
            except Exception as exc:
                LOGGER.warning("bucket quality failed to read history %s: %s", path, exc)
        self._history_rows_loaded = rows_loaded

    def _resolve_history_files(self) -> list[Path]:
        pattern = self._history_glob.strip()
        if not pattern:
            return []

        candidates = [Path(path) for path in glob.glob(pattern)]
        candidates = [path for path in candidates if path.exists() and path.is_file()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[: self._history_max_files]

    def _ingest_row_dict(self, group_id: str, row: dict[str, Any]) -> None:
        action = str(row.get("action") or "").strip().lower()
        detected_edge = _to_float(row.get("detected_edge_per_contract"), 0.0)
        fill_probability = _to_float(row.get("fill_probability"), 0.0)
        expected_realized_profit = _to_float(row.get("expected_realized_profit"), 0.0)
        raw_realized = row.get("realized_profit")
        realized_profit = _to_float(raw_realized, 0.0) if raw_realized not in (None, "", "None") else None
        slippage = _to_float(row.get("expected_slippage_per_contract"), 0.0)
        latency = _to_float(row.get("execution_latency_ms"), 0.0)
        self._ingest_bucket_observation(
            group_id=group_id,
            action=action,
            detected_edge_per_contract=detected_edge,
            fill_probability=fill_probability,
            expected_realized_profit=expected_realized_profit,
            realized_profit=realized_profit,
            expected_slippage_per_contract=slippage,
            execution_latency_ms=latency,
        )

    def _ingest_bucket_observation(
        self,
        *,
        group_id: str,
        action: str,
        detected_edge_per_contract: float,
        fill_probability: float,
        expected_realized_profit: float,
        realized_profit: float | None,
        expected_slippage_per_contract: float,
        execution_latency_ms: float | None,
    ) -> None:
        acc = self._accumulators.setdefault(group_id, BucketAccumulator())
        acc.detections += 1
        acc.detected_edge_sum += detected_edge_per_contract
        acc.fill_probability_sum += max(0.0, min(1.0, fill_probability))

        if action == "skipped":
            return

        acc.opened += 1
        if action == "settled":
            acc.settled += 1

        acc.expected_realized_profit_sum += expected_realized_profit
        if realized_profit is None:
            realized_value = expected_realized_profit
        else:
            realized_value = realized_profit
        acc.realized_profit_sum += realized_value
        if realized_value > 0:
            acc.positive_outcomes += 1

        acc.slippage_sum += max(0.0, expected_slippage_per_contract)
        if execution_latency_ms is not None and execution_latency_ms > 0:
            acc.latency_sum_ms += execution_latency_ms

    def _recompute_active_set(self) -> None:
        all_ids = list(self._bucket_leg_counts.keys())
        if not all_ids:
            self._active_bucket_ids = set()
            self._scores = {}
            return

        scored: dict[str, BucketScore] = {}
        for group_id in all_ids:
            acc = self._accumulators.get(group_id)
            if acc is None:
                continue
            score_obj = self._score_bucket(group_id, acc)
            scored[group_id] = score_obj
        self._scores = scored

        # If we have no usable score history for this ruleset, keep all buckets
        # active instead of selecting a sampled subset that may miss live edges.
        if not scored:
            self._active_bucket_ids = set(all_ids)
            return

        if not self._enabled:
            self._active_bucket_ids = set(all_ids)
            return

        capacity = len(all_ids) if self._max_active_buckets <= 0 else min(self._max_active_buckets, len(all_ids))
        if capacity <= 0:
            self._active_bucket_ids = set(all_ids)
            return

        if self._use_thompson_sampling:
            self._active_bucket_ids = self._thompson_select(all_ids, capacity)
        else:
            self._active_bucket_ids = self._deterministic_select(all_ids, capacity)

    def _thompson_select(self, all_ids: list[str], capacity: int) -> set[str]:
        """Thompson Sampling: sample from each bucket's Beta posterior,
        enable top-k by sampled value.

        Beta(alpha, beta) where:
            alpha = positive_outcomes + prior_alpha
            beta  = (opened - positive_outcomes) + prior_beta

        Buckets with no history use the prior (uniform by default),
        giving them high variance = natural exploration.
        """
        sampled: list[tuple[float, str]] = []
        for group_id in all_ids:
            acc = self._accumulators.get(group_id)
            if acc is None or acc.opened == 0:
                # No history — sample from prior (high variance = exploration).
                alpha = self._thompson_prior_alpha
                beta_param = self._thompson_prior_beta
            else:
                alpha = acc.positive_outcomes + self._thompson_prior_alpha
                beta_param = (acc.opened - acc.positive_outcomes) + self._thompson_prior_beta

            # Hard filter: if we have enough data and score is terrible, skip.
            score = self._scores.get(group_id)
            if score is not None and acc is not None and acc.detections >= self._min_observations:
                if score.score < self._min_score:
                    continue

            sampled_value = random.betavariate(alpha, beta_param)
            sampled.append((sampled_value, group_id))

        # Sort by sampled value descending, take top capacity.
        sampled.sort(key=lambda x: x[0], reverse=True)
        active = {group_id for _, group_id in sampled[:capacity]}

        if not active:
            active = set(all_ids)
        return active

    def _deterministic_select(self, all_ids: list[str], capacity: int) -> set[str]:
        """Legacy deterministic explore/exploit selection (fallback)."""
        ready: list[str] = []
        uncertain: list[str] = []
        for group_id in all_ids:
            acc = self._accumulators.get(group_id)
            if acc is None or acc.detections < self._min_observations:
                uncertain.append(group_id)
                continue
            score = self._scores.get(group_id)
            if score is None:
                uncertain.append(group_id)
                continue
            if score.score < self._min_score:
                continue
            ready.append(group_id)

        ready.sort(key=lambda group_id: self._scores[group_id].score, reverse=True)
        uncertain.sort(
            key=lambda group_id: (
                self._accumulators.get(group_id).detections if self._accumulators.get(group_id) else 0,
                _stable_sort_key(group_id),
            )
        )

        active: list[str] = []
        if ready:
            explore_slots = 0
            if uncertain and self._explore_fraction > 0:
                explore_slots = min(len(uncertain), int(round(capacity * self._explore_fraction)))
            exploit_slots = max(1, capacity - explore_slots)

            active.extend(ready[:exploit_slots])
            if len(active) < capacity and uncertain:
                need = capacity - len(active)
                active.extend(uncertain[:need])
            if len(active) < capacity:
                remaining_ready = [group_id for group_id in ready if group_id not in active]
                need = capacity - len(active)
                active.extend(remaining_ready[:need])
        else:
            active.extend(uncertain[:capacity] if uncertain else all_ids[:capacity])

        if not active:
            active = list(all_ids)

        return set(active)

    def _score_bucket(self, group_id: str, acc: BucketAccumulator) -> BucketScore:
        avg_edge = acc.average_detected_edge()
        avg_fill = acc.average_fill_probability()
        open_rate = acc.open_rate()
        persistence = acc.positive_outcome_rate()
        avg_slippage = acc.average_slippage()
        avg_latency_ms = acc.average_latency_ms()
        posterior_profit = acc.posterior_expected_realized_profit(
            prior_mean=self._prior_mean_realized_profit,
            prior_strength=self._prior_strength,
        )
        latency_penalty = min(0.05, avg_latency_ms / 10_000.0)
        leg_count = max(2, int(self._bucket_leg_counts.get(group_id, 2)))
        leg_penalty = max(0, leg_count - 2) * self._leg_count_penalty

        score = (
            posterior_profit
            + (avg_edge * avg_fill * open_rate * persistence)
            - avg_slippage
            - latency_penalty
            - leg_penalty
        )

        return BucketScore(
            group_id=group_id,
            score=score,
            detections=acc.detections,
            opened=acc.opened,
            posterior_expected_realized_profit=posterior_profit,
            average_detected_edge=avg_edge,
            average_fill_probability=avg_fill,
            open_rate=open_rate,
            positive_outcome_rate=persistence,
        )


def bucket_leg_count_map(group_ids_and_leg_counts: Iterable[tuple[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group_id, leg_count in group_ids_and_leg_counts:
        if not group_id:
            continue
        if group_id in counts:
            counts[group_id] = max(counts[group_id], max(2, int(leg_count)))
        else:
            counts[group_id] = max(2, int(leg_count))
    return counts
