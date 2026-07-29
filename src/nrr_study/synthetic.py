"""Deterministic synthetic benchmarks for prospective gate calibration.

The generators are observation-location benchmarks, not SGS.  They create
repeatable synthetic responses at supplied composite coordinates and never
produce blocks or resource realizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

Scenario = Literal["null", "stationary", "hard_boundary", "transitional"]


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    simulations_per_scenario: int = 500
    random_seed: int = 20260728
    n_features: int = 64
    practical_range: float | None = None
    structured_variance: float = 1.0
    nugget_variance: float = 0.25
    boundary_effect: float = 1.0
    transition_width: float | None = None


@dataclass(frozen=True)
class GateCalibration:
    threshold: float
    direction: Literal["higher", "lower"]
    null_false_pass_rate: float
    positive_detection_rates: Mapping[str, float]
    maximum_null_false_pass_rate: float
    minimum_positive_detection_rate: float
    meets_targets: bool


@dataclass(frozen=True)
class SyntheticBenchmarkReport:
    scores: Mapping[str, np.ndarray]
    calibration: GateCalibration
    simulations_per_scenario: int


def _coordinates(coordinates: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    if coords.ndim != 2 or len(coords) < 2:
        raise ValueError("coordinates must be an (n, d) array with n >= 2")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contain non-finite values")
    return coords


def _resolved_range(coords: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        if requested <= 0:
            raise ValueError("practical_range must be positive")
        return float(requested)
    diagonal = float(np.linalg.norm(np.ptp(coords, axis=0)))
    if diagonal <= 0:
        raise ValueError("coordinates have no positive spatial extent")
    return diagonal / 3.0


def _standardize_columns(matrix: np.ndarray) -> np.ndarray:
    centred = matrix - np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(centred, axis=0, keepdims=True)
    scale[scale <= np.finfo(float).eps] = 1.0
    return centred / scale


def _random_feature_basis(
    coords: np.ndarray,
    *,
    practical_range: float,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_features < 4:
        raise ValueError("n_features must be at least 4")
    centred = coords - np.mean(coords, axis=0, keepdims=True)
    # Gaussian random Fourier features give a smooth positive benchmark.  The
    # empirical gate is still fitted with the declared candidate variograms.
    frequencies = rng.normal(
        scale=np.sqrt(3.0) / practical_range,
        size=(coords.shape[1], n_features),
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(centred @ frequencies + phase)


def generate_synthetic_matrix(
    coordinates: np.ndarray | Sequence[Sequence[float]],
    *,
    scenario: Scenario,
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
    domains: Sequence[object] | None = None,
    signed_contact_distance: Sequence[float] | None = None,
    seed_offset: int = 0,
) -> np.ndarray:
    """Generate ``(n_observations, n_simulations)`` synthetic responses."""

    coords = _coordinates(coordinates)
    simulations = int(config.simulations_per_scenario)
    if simulations < 1:
        raise ValueError("simulations_per_scenario must be positive")
    if config.structured_variance < 0 or config.nugget_variance < 0:
        raise ValueError("variance parameters must be nonnegative")
    rng = np.random.default_rng(config.random_seed + int(seed_offset))
    n = len(coords)
    nugget = rng.normal(
        scale=np.sqrt(config.nugget_variance),
        size=(n, simulations),
    )

    if scenario == "null":
        return nugget

    practical_range = _resolved_range(coords, config.practical_range)
    if scenario == "stationary":
        basis = _random_feature_basis(
            coords,
            practical_range=practical_range,
            n_features=config.n_features,
            rng=rng,
        )
        weights = rng.normal(size=(config.n_features, simulations))
        structured = _standardize_columns(basis @ weights)
        return np.sqrt(config.structured_variance) * structured + nugget

    if scenario == "hard_boundary":
        if domains is None:
            raise ValueError("hard_boundary requires domains")
        domain_array = np.asarray(domains, dtype=object).astype(str)
        if len(domain_array) != n:
            raise ValueError("domains must match coordinates")
        levels = np.asarray(sorted(np.unique(domain_array)), dtype=object)
        if len(levels) < 2:
            raise ValueError("hard_boundary requires at least two domains")
        structured = np.zeros((n, simulations), dtype=float)
        offsets = np.linspace(
            -config.boundary_effect,
            config.boundary_effect,
            len(levels),
        )
        for index, level in enumerate(levels):
            use = domain_array == level
            basis = _random_feature_basis(
                coords[use],
                practical_range=practical_range,
                n_features=config.n_features,
                rng=rng,
            )
            weights = rng.normal(size=(config.n_features, simulations))
            local = _standardize_columns(basis @ weights)
            structured[use] = (
                np.sqrt(config.structured_variance) * local + offsets[index]
            )
        return structured + nugget

    if scenario == "transitional":
        if signed_contact_distance is None:
            distance = coords[:, 0] - np.median(coords[:, 0])
        else:
            distance = np.asarray(signed_contact_distance, dtype=float)
            if len(distance) != n or not np.all(np.isfinite(distance)):
                raise ValueError(
                    "signed_contact_distance must be finite and match coordinates"
                )
        width = (
            config.transition_width
            if config.transition_width is not None
            else max(float(np.std(distance)) / 3.0, np.finfo(float).eps)
        )
        if width <= 0:
            raise ValueError("transition_width must be positive")
        basis = _random_feature_basis(
            coords,
            practical_range=practical_range,
            n_features=config.n_features,
            rng=rng,
        )
        weights = rng.normal(size=(config.n_features, simulations))
        structured = _standardize_columns(basis @ weights)
        transition = (
            config.boundary_effect * np.tanh(distance / width)
        )[:, None]
        return (
            np.sqrt(config.structured_variance) * structured
            + transition
            + nugget
        )

    raise ValueError(f"unsupported synthetic scenario: {scenario}")


def generate_all_scenarios(
    coordinates: np.ndarray | Sequence[Sequence[float]],
    *,
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
    domains: Sequence[object] | None = None,
    signed_contact_distance: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Generate the frozen null and three positive benchmark families."""

    scenarios: tuple[Scenario, ...] = (
        "null",
        "stationary",
        "hard_boundary",
        "transitional",
    )
    return {
        scenario: generate_synthetic_matrix(
            coordinates,
            scenario=scenario,
            config=config,
            domains=domains,
            signed_contact_distance=signed_contact_distance,
            seed_offset=10_000 * index,
        )
        for index, scenario in enumerate(scenarios)
    }


def calibrate_gate_threshold(
    null_scores: Sequence[float],
    positive_scores: Mapping[str, Sequence[float]],
    *,
    direction: Literal["higher", "lower"] = "higher",
    maximum_null_false_pass_rate: float = 0.05,
    minimum_positive_detection_rate: float = 0.80,
) -> GateCalibration:
    """Calibrate a threshold on nulls and audit positive detection.

    The null distribution alone fixes the threshold.  Positive scenarios are
    used only to test whether the declared design has adequate power.
    """

    null = np.asarray(null_scores, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) < 2:
        raise ValueError("at least two finite null scores are required")
    if not 0 < maximum_null_false_pass_rate < 1:
        raise ValueError("maximum_null_false_pass_rate must lie in (0, 1)")
    if not 0 < minimum_positive_detection_rate <= 1:
        raise ValueError("minimum_positive_detection_rate must lie in (0, 1]")

    if direction == "higher":
        boundary = np.quantile(
            null,
            1.0 - maximum_null_false_pass_rate,
            method="higher",
        )
        threshold = float(np.nextafter(boundary, np.inf))
        passes = lambda values: values >= threshold
    elif direction == "lower":
        boundary = np.quantile(
            null,
            maximum_null_false_pass_rate,
            method="lower",
        )
        threshold = float(np.nextafter(boundary, -np.inf))
        passes = lambda values: values <= threshold
    else:
        raise ValueError(f"unsupported direction: {direction}")

    null_rate = float(np.mean(passes(null)))
    detection: dict[str, float] = {}
    for name, raw in positive_scores.items():
        values = np.asarray(raw, dtype=float)
        values = values[np.isfinite(values)]
        detection[name] = float(np.mean(passes(values))) if len(values) else 0.0
    meets = (
        null_rate <= maximum_null_false_pass_rate
        and bool(detection)
        and all(
            rate >= minimum_positive_detection_rate
            for rate in detection.values()
        )
    )
    return GateCalibration(
        threshold=threshold,
        direction=direction,
        null_false_pass_rate=null_rate,
        positive_detection_rates=detection,
        maximum_null_false_pass_rate=maximum_null_false_pass_rate,
        minimum_positive_detection_rate=minimum_positive_detection_rate,
        meets_targets=meets,
    )


def benchmark_gate(
    coordinates: np.ndarray | Sequence[Sequence[float]],
    score_function: Callable[[np.ndarray], float],
    *,
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
    domains: Sequence[object] | None = None,
    signed_contact_distance: Sequence[float] | None = None,
    matrix_transform_function: Callable[[np.ndarray], np.ndarray] | None = None,
    direction: Literal["higher", "lower"] = "higher",
    maximum_null_false_pass_rate: float = 0.05,
    minimum_positive_detection_rate: float = 0.80,
) -> SyntheticBenchmarkReport:
    """Run the declared simulations and calibrate a scalar gate score.

    ``matrix_transform_function`` applies the same deterministic preprocessing
    to every simulated response matrix before scoring. It is intended for
    fitting and removing the same geology/depth trend used for the observed
    residuals. The transform must preserve the ``(observations, simulations)``
    shape so null and positive scenarios remain directly comparable.
    """

    matrices = generate_all_scenarios(
        coordinates,
        config=config,
        domains=domains,
        signed_contact_distance=signed_contact_distance,
    )
    transformed: dict[str, np.ndarray] = {}
    for scenario, matrix in matrices.items():
        candidate = (
            matrix
            if matrix_transform_function is None
            else np.asarray(
                matrix_transform_function(np.array(matrix, copy=True)),
                dtype=float,
            )
        )
        if candidate.shape != matrix.shape:
            raise ValueError(
                "matrix_transform_function must preserve the synthetic "
                f"matrix shape; {scenario!r} changed {matrix.shape} to "
                f"{candidate.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(
                "matrix_transform_function returned non-finite values for "
                f"{scenario!r}"
            )
        transformed[scenario] = candidate
    scores = {
        scenario: np.asarray(
            [score_function(matrix[:, i]) for i in range(matrix.shape[1])],
            dtype=float,
        )
        for scenario, matrix in transformed.items()
    }
    calibration = calibrate_gate_threshold(
        scores["null"],
        {name: values for name, values in scores.items() if name != "null"},
        direction=direction,
        maximum_null_false_pass_rate=maximum_null_false_pass_rate,
        minimum_positive_detection_rate=minimum_positive_detection_rate,
    )
    return SyntheticBenchmarkReport(
        scores=scores,
        calibration=calibration,
        simulations_per_scenario=config.simulations_per_scenario,
    )


@lru_cache(maxsize=16)
def _balanced_between_hole_pair_indices(
    holes: tuple[str, ...],
    domains: tuple[str, ...] | None,
    pair_count: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache deterministic equal-hole-pair samples across simulations."""

    hole_array = np.asarray(holes, dtype=object)
    domain_array = (
        np.asarray(domains, dtype=object) if domains is not None else None
    )
    rng = np.random.default_rng(random_state)
    hole_levels = tuple(sorted(np.unique(hole_array).tolist()))
    hole_rows = {
        hole: np.flatnonzero(hole_array == hole) for hole in hole_levels
    }
    eligible: list[tuple[str, str, tuple[str, ...]]] = []
    for first_position, first_hole in enumerate(hole_levels[:-1]):
        first_domains = (
            set(domain_array[hole_rows[first_hole]].tolist())
            if domain_array is not None
            else set()
        )
        for second_hole in hole_levels[first_position + 1 :]:
            if domain_array is None:
                common_domains: tuple[str, ...] = ()
            else:
                second_domains = set(
                    domain_array[hole_rows[second_hole]].tolist()
                )
                common_domains = tuple(sorted(first_domains & second_domains))
                if not common_domains:
                    continue
            eligible.append((first_hole, second_hole, common_domains))
    if not eligible:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    effective_count = max(pair_count, len(eligible))
    base_draws, remainder = divmod(effective_count, len(eligible))
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for pair_position, (first_hole, second_hole, common_domains) in enumerate(
        eligible
    ):
        draws = base_draws + int(pair_position < remainder)
        first_candidates = hole_rows[first_hole]
        second_candidates = hole_rows[second_hole]
        if domain_array is None:
            first_parts.append(rng.choice(first_candidates, size=draws))
            second_parts.append(rng.choice(second_candidates, size=draws))
            continue
        domain_weights = np.asarray(
            [
                np.sum(domain_array[first_candidates] == level)
                * np.sum(domain_array[second_candidates] == level)
                for level in common_domains
            ],
            dtype=float,
        )
        domain_weights /= np.sum(domain_weights)
        selected_domains = rng.choice(
            len(common_domains), size=draws, p=domain_weights
        )
        first_draw = np.empty(draws, dtype=int)
        second_draw = np.empty(draws, dtype=int)
        for domain_position, level in enumerate(common_domains):
            use = selected_domains == domain_position
            count = int(np.sum(use))
            if not count:
                continue
            first_draw[use] = rng.choice(
                first_candidates[domain_array[first_candidates] == level],
                size=count,
            )
            second_draw[use] = rng.choice(
                second_candidates[domain_array[second_candidates] == level],
                size=count,
            )
        first_parts.append(first_draw)
        second_parts.append(second_draw)
    first = np.concatenate(first_parts)
    second = np.concatenate(second_parts)
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


def sampled_short_lag_signal_score(
    values: Sequence[float],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    *,
    domains: Sequence[object] | None = None,
    holes: Sequence[object] | None = None,
    between_hole_only: bool = False,
    balance_hole_pairs: bool = False,
    pair_count: int = 20_000,
    short_lag_quantile: float = 0.20,
    random_state: int = 20260728,
) -> float:
    """Fast benchmark score: one minus short-lag/overall semivariance.

    It is intended for synthetic power calibration, not as a fitted range or a
    substitute for the full empirical variogram audit. Supplying ``domains``
    restricts the calculation to same-domain pairs. With
    ``balance_hole_pairs=True``, every eligible independent hole pair receives
    the same target number of sampled row pairs before the score is aggregated
    with equal hole-pair weight.
    """

    y = np.asarray(values, dtype=float)
    coords = _coordinates(coordinates)
    if len(y) != len(coords) or not np.all(np.isfinite(y)):
        raise ValueError("values must be finite and match coordinates")
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if not 0.0 < short_lag_quantile < 1.0:
        raise ValueError("short_lag_quantile must lie in (0, 1)")
    if (between_hole_only or balance_hole_pairs) and holes is None:
        raise ValueError(
            "holes are required for between-hole filtering or hole-pair "
            "balancing"
        )
    if balance_hole_pairs and not between_hole_only:
        raise ValueError(
            "hole-pair balancing requires between_hole_only=True"
        )
    hole_array: np.ndarray | None = None
    if holes is not None:
        hole_array = np.asarray(holes, dtype=object).astype(str)
        if len(hole_array) != len(y):
            raise ValueError("holes must match values")
    domain_array: np.ndarray | None = None
    if domains is not None:
        domain_array = np.asarray(domains, dtype=object).astype(str)
        if len(domain_array) != len(y):
            raise ValueError("domains must match values")
    rng = np.random.default_rng(random_state)
    if balance_hole_pairs:
        assert hole_array is not None
        first, second = _balanced_between_hole_pair_indices(
            tuple(hole_array.tolist()),
            (
                tuple(domain_array.tolist())
                if domain_array is not None
                else None
            ),
            pair_count,
            random_state,
        )
    else:
        first = rng.integers(0, len(y), size=pair_count * 2)
        second = rng.integers(0, len(y), size=pair_count * 2)
        keep = first != second
        if between_hole_only:
            assert hole_array is not None
            keep &= hole_array[first] != hole_array[second]
        if domain_array is not None:
            keep &= domain_array[first] == domain_array[second]
        first, second = first[keep][:pair_count], second[keep][:pair_count]
    if len(first) < 10:
        return float("nan")
    distance = np.linalg.norm(coords[first] - coords[second], axis=1)
    positive = np.isfinite(distance) & (distance > 0)
    first, second, distance = first[positive], second[positive], distance[positive]
    if len(first) < 10:
        return float("nan")
    semivariance = 0.5 * (y[first] - y[second]) ** 2
    cutoff = np.quantile(distance, short_lag_quantile)
    short_mask = distance <= cutoff
    if not np.any(short_mask):
        return float("nan")
    if balance_hole_pairs:
        assert hole_array is not None
        first_hole = hole_array[first]
        second_hole = hole_array[second]
        keys = np.asarray(
            [
                f"{a}|{b}" if a <= b else f"{b}|{a}"
                for a, b in zip(first_hole, second_hole)
            ],
            dtype=object,
        )
        _, inverse = np.unique(keys, return_inverse=True)
        group_count = int(np.max(inverse)) + 1
        overall_counts = np.bincount(inverse, minlength=group_count)
        overall_sums = np.bincount(
            inverse, weights=semivariance, minlength=group_count
        )
        overall = float(np.mean(overall_sums / overall_counts))
        short_inverse = inverse[short_mask]
        short_counts = np.bincount(short_inverse, minlength=group_count)
        short_sums = np.bincount(
            short_inverse,
            weights=semivariance[short_mask],
            minlength=group_count,
        )
        supported = short_counts > 0
        short_mean = float(
            np.mean(short_sums[supported] / short_counts[supported])
        )
    else:
        overall = float(np.mean(semivariance))
        short_mean = float(np.mean(semivariance[short_mask]))
    if overall <= np.finfo(float).eps:
        return 0.0
    return float(1.0 - short_mean / overall)
