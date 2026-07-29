"""Variogram estimation, fitting, and prospective stability gates.

The functions in this module operate on canonical point-support composites.
They deliberately do not create grids, blocks, resources, or directional
interpretations.  Pair accounting is kept explicit so a visually smooth
variogram cannot hide domination by one hole or one parent assay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

Estimator = Literal["classical", "robust"]
VariogramKind = Literal["downhole", "omnidirectional"]
ModelKind = Literal["exponential", "spherical", "gaussian"]
PairMode = Literal["combined_raw", "between_hole_balanced"]
PairDomainPolicy = Literal["pooled", "same_domain"]


@dataclass(frozen=True)
class EmpiricalVariogram:
    """Binned semivariogram plus auditable pair-support information."""

    kind: VariogramKind
    estimator: Estimator
    bin_edges: np.ndarray
    lag: np.ndarray
    semivariance: np.ndarray
    raw_pairs: np.ndarray
    unique_hole_pairs: np.ndarray
    same_hole_pairs: np.ndarray
    excluded_same_parent_pairs: int
    hole_pair_contributions: tuple[Mapping[str, int], ...] = field(
        repr=False
    )
    pair_mode: PairMode = "combined_raw"
    pair_domain_policy: PairDomainPolicy = "pooled"
    excluded_cross_domain_pairs: int = 0

    @property
    def supported(self) -> np.ndarray:
        return np.isfinite(self.semivariance) & (self.raw_pairs > 0)

    @property
    def fit_support(self) -> np.ndarray:
        """Support basis used to weight model fitting.

        Corrected between-hole variograms use independent hole-pair support;
        the legacy combined branch retains raw-pair support as a sensitivity.
        """

        if self.pair_mode == "between_hole_balanced":
            return self.unique_hole_pairs.astype(float)
        return self.raw_pairs.astype(float)


@dataclass(frozen=True)
class VariogramModel:
    """A fitted isotropic semivariogram model.

    ``sill`` is the partial sill.  The total variance is ``nugget + sill``.
    ``range`` is the practical range for exponential and Gaussian models and
    the finite range for the spherical model.
    """

    model: ModelKind
    range: float
    sill: float
    nugget: float
    rmse: float
    normalized_rmse: float
    n_bins: int
    success: bool = True
    message: str = ""

    @property
    def total_sill(self) -> float:
        return float(self.sill + self.nugget)


@dataclass(frozen=True)
class StabilityThresholds:
    """Prospective thresholds, intended to be calibrated synthetically."""

    min_successful_fits: int = 4
    min_supported_bins: int = 4
    min_unique_hole_pairs_per_bin: int = 5
    max_range_ratio: float = 3.0
    max_range_cv: float = 0.50
    max_normalized_rmse: float = 0.35
    max_nugget_fraction: float = 0.80


@dataclass(frozen=True)
class VariogramStabilityResult:
    """Decision record for the residual-variogram gate."""

    passed: bool
    decision: Literal["accept", "abstain"]
    reasons: tuple[str, ...]
    successful_fits: int
    supported_bins: int
    median_range: float
    range_ratio: float
    range_cv: float
    worst_normalized_rmse: float
    worst_nugget_fraction: float


def _as_2d_coordinates(coordinates: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    if coords.ndim != 2 or coords.shape[1] < 1:
        raise ValueError("coordinates must be an (n, d) numeric array")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contain non-finite values")
    return coords


def _canonical_hole_pair(first: object, second: object) -> str:
    a, b = str(first), str(second)
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _parent_memberships(
    values: Sequence[object],
) -> tuple[list[frozenset[str]], dict[str, set[int]]]:
    memberships: list[frozenset[str]] = []
    token_rows: dict[str, set[int]] = {}
    for row_index, value in enumerate(values):
        raw = "" if value is None else str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "<na>"}:
            tokens = frozenset()
        else:
            tokens = frozenset(
                token.strip() for token in raw.split("|") if token.strip()
            )
        memberships.append(tokens)
        for token in tokens:
            token_rows.setdefault(token, set()).add(row_index)
    return memberships, token_rows


def _default_maxlag(coords: np.ndarray) -> float:
    span = np.ptp(coords, axis=0)
    diagonal = float(np.linalg.norm(span))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("coordinates have no positive spatial extent")
    return 0.5 * diagonal


def empirical_variogram(
    coordinates: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float],
    holes: Sequence[object],
    parent_ids: Sequence[object] | None = None,
    *,
    kind: VariogramKind = "omnidirectional",
    estimator: Estimator = "classical",
    n_lags: int = 10,
    maxlag: float | None = None,
    bin_edges: Sequence[float] | None = None,
    alonghole: Sequence[float] | None = None,
    pair_mode: PairMode = "combined_raw",
    domains: Sequence[object] | None = None,
    pair_domain_policy: PairDomainPolicy = "pooled",
) -> EmpiricalVariogram:
    """Compute a classical or Cressie-Hawkins robust semivariogram.

    Same-parent pairs are always excluded.  Downhole variograms retain only
    within-hole pairs and use ``alonghole`` separation when supplied.
    Omnidirectional legacy variograms retain both within- and between-hole
    pairs. ``between_hole_balanced`` excludes within-hole pairs and gives each
    independent hole pair equal weight within a lag.
    """

    coords = _as_2d_coordinates(coordinates)
    vals = np.asarray(values, dtype=float)
    hole_arr = np.asarray(holes, dtype=object)
    n = len(vals)
    if len(coords) != n or len(hole_arr) != n:
        raise ValueError("coordinates, values, and holes must have equal length")
    if n < 2:
        raise ValueError("at least two observations are required")
    if not np.all(np.isfinite(vals)):
        raise ValueError("values contain non-finite entries")
    if kind not in {"downhole", "omnidirectional"}:
        raise ValueError(f"unsupported variogram kind: {kind}")
    if estimator not in {"classical", "robust"}:
        raise ValueError(f"unsupported estimator: {estimator}")
    if pair_mode not in {"combined_raw", "between_hole_balanced"}:
        raise ValueError(f"unsupported pair mode: {pair_mode}")
    if pair_domain_policy not in {"pooled", "same_domain"}:
        raise ValueError(
            f"unsupported pair-domain policy: {pair_domain_policy}"
        )
    if kind == "downhole" and pair_mode != "combined_raw":
        raise ValueError("downhole variograms require combined_raw pair mode")
    domain_arr: np.ndarray | None = None
    if domains is not None:
        if len(domains) != n:
            raise ValueError("domains must have the same length as values")
        domain_arr = np.asarray(domains, dtype=object).astype(str)
    if pair_domain_policy == "same_domain" and domain_arr is None:
        raise ValueError("same_domain pair policy requires domains")

    if parent_ids is None:
        parent_sets = [frozenset((f"__row_{idx}",)) for idx in range(n)]
        parent_token_rows = {f"__row_{idx}": {idx} for idx in range(n)}
    else:
        if len(parent_ids) != n:
            raise ValueError("parent_ids must have the same length as values")
        parent_sets, parent_token_rows = _parent_memberships(parent_ids)

    along = None
    if alonghole is not None:
        along = np.asarray(alonghole, dtype=float)
        if len(along) != n or not np.all(np.isfinite(along)):
            raise ValueError("alonghole must be finite and match values")

    if bin_edges is None:
        if n_lags < 2:
            raise ValueError("n_lags must be at least 2")
        resolved_maxlag = _default_maxlag(coords) if maxlag is None else float(maxlag)
        if resolved_maxlag <= 0:
            raise ValueError("maxlag must be positive")
        edges = np.linspace(0.0, resolved_maxlag, n_lags + 1)
    else:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 3:
            raise ValueError("bin_edges must contain at least three ordered edges")
        if not np.all(np.diff(edges) > 0) or edges[0] < 0:
            raise ValueError("bin_edges must be strictly increasing and nonnegative")
        resolved_maxlag = float(edges[-1])

    bins = len(edges) - 1
    raw_pairs = np.zeros(bins, dtype=np.int64)
    same_hole_pairs = np.zeros(bins, dtype=np.int64)
    sum_distance = np.zeros(bins, dtype=float)
    sum_squared_difference = np.zeros(bins, dtype=float)
    sum_root_absolute_difference = np.zeros(bins, dtype=float)
    contributions: list[dict[str, int]] = [dict() for _ in range(bins)]
    independent_pairs: list[set[str]] = [set() for _ in range(bins)]
    pair_distance_sum: list[dict[str, float]] = [dict() for _ in range(bins)]
    pair_squared_sum: list[dict[str, float]] = [dict() for _ in range(bins)]
    pair_root_sum: list[dict[str, float]] = [dict() for _ in range(bins)]
    excluded_same_parent = 0
    excluded_cross_domain = 0

    # A row-at-a-time vectorization avoids materializing all O(n^2) pairs.
    for i in range(n - 1):
        js = np.arange(i + 1, n)
        linked_rows: set[int] = set()
        for parent_token in parent_sets[i]:
            linked_rows.update(parent_token_rows[parent_token])
        same_parent = np.fromiter(
            (int(row_index) in linked_rows for row_index in js),
            dtype=bool,
            count=len(js),
        )
        excluded_same_parent += int(np.sum(same_parent))
        keep = ~same_parent
        if kind == "downhole":
            keep &= hole_arr[js] == hole_arr[i]
        elif pair_mode == "between_hole_balanced":
            keep &= hole_arr[js] != hole_arr[i]
        if pair_domain_policy == "same_domain":
            cross_domain = domain_arr[js] != domain_arr[i]
            excluded_cross_domain += int(np.sum(keep & cross_domain))
            keep &= ~cross_domain
        if not np.any(keep):
            continue
        js = js[keep]
        if kind == "downhole" and along is not None:
            distance = np.abs(along[js] - along[i])
        else:
            distance = np.linalg.norm(coords[js] - coords[i], axis=1)
        delta = vals[js] - vals[i]
        valid = np.isfinite(distance) & (distance > 0) & (distance <= resolved_maxlag)
        if not np.any(valid):
            continue
        js, distance, delta = js[valid], distance[valid], delta[valid]
        index = np.searchsorted(edges, distance, side="right") - 1
        index[index == bins] = bins - 1
        valid_bin = (index >= 0) & (index < bins)
        js, distance, delta, index = (
            js[valid_bin],
            distance[valid_bin],
            delta[valid_bin],
            index[valid_bin],
        )
        np.add.at(raw_pairs, index, 1)
        np.add.at(sum_distance, index, distance)
        np.add.at(sum_squared_difference, index, delta * delta)
        np.add.at(sum_root_absolute_difference, index, np.sqrt(np.abs(delta)))

        for j, b, d, difference in zip(
            js.tolist(), index.tolist(), distance.tolist(), delta.tolist()
        ):
            key = _canonical_hole_pair(hole_arr[i], hole_arr[j])
            contributions[b][key] = contributions[b].get(key, 0) + 1
            if hole_arr[i] == hole_arr[j]:
                same_hole_pairs[b] += 1
            else:
                independent_pairs[b].add(key)
                pair_distance_sum[b][key] = (
                    pair_distance_sum[b].get(key, 0.0) + float(d)
                )
                pair_squared_sum[b][key] = (
                    pair_squared_sum[b].get(key, 0.0)
                    + float(difference) * float(difference)
                )
                pair_root_sum[b][key] = (
                    pair_root_sum[b].get(key, 0.0)
                    + float(np.sqrt(abs(difference)))
                )

    lag = np.full(bins, np.nan, dtype=float)
    gamma = np.full(bins, np.nan, dtype=float)
    nonempty = raw_pairs > 0
    if pair_mode == "between_hole_balanced":
        for b in range(bins):
            keys = tuple(independent_pairs[b])
            if not keys:
                continue
            counts = contributions[b]
            lag[b] = float(
                np.mean(
                    [pair_distance_sum[b][key] / counts[key] for key in keys]
                )
            )
            if estimator == "classical":
                gamma[b] = float(
                    np.mean(
                        [
                            0.5 * pair_squared_sum[b][key] / counts[key]
                            for key in keys
                        ]
                    )
                )
            else:
                pair_estimates: list[float] = []
                for key in keys:
                    count = float(counts[key])
                    mean_root = pair_root_sum[b][key] / count
                    correction = 0.457 + 0.494 / count + 0.045 / (count * count)
                    pair_estimates.append(0.5 * float(mean_root**4) / correction)
                gamma[b] = float(np.mean(pair_estimates))
    else:
        lag[nonempty] = sum_distance[nonempty] / raw_pairs[nonempty]
        if estimator == "classical":
            gamma[nonempty] = 0.5 * (
                sum_squared_difference[nonempty] / raw_pairs[nonempty]
            )
        else:
            count = raw_pairs[nonempty].astype(float)
            mean_root = sum_root_absolute_difference[nonempty] / count
            correction = 0.457 + 0.494 / count + 0.045 / (count * count)
            gamma[nonempty] = 0.5 * np.power(mean_root, 4) / correction

    return EmpiricalVariogram(
        kind=kind,
        estimator=estimator,
        bin_edges=edges,
        lag=lag,
        semivariance=gamma,
        raw_pairs=raw_pairs,
        unique_hole_pairs=np.asarray([len(x) for x in independent_pairs], dtype=int),
        same_hole_pairs=same_hole_pairs,
        excluded_same_parent_pairs=excluded_same_parent,
        hole_pair_contributions=tuple(dict(x) for x in contributions),
        pair_mode=pair_mode,
        pair_domain_policy=pair_domain_policy,
        excluded_cross_domain_pairs=excluded_cross_domain,
    )


def directional_pair_support(
    coordinates: np.ndarray | Sequence[Sequence[float]],
    holes: Sequence[object],
    parent_ids: Sequence[object] | None = None,
    *,
    azimuths: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    tolerances: Sequence[float] = (15.0, 30.0),
    lag_edges: Sequence[float],
) -> tuple[dict[str, object], ...]:
    """Audit grade-blind horizontal directional support.

    Azimuth is axial (0--180 degrees) and measured clockwise from north.
    Only between-hole pairs are counted.  The result is a feasibility audit;
    it does not estimate a directional semivariogram or select anisotropy.
    """

    coords = _as_2d_coordinates(coordinates)
    if coords.shape[1] < 2:
        raise ValueError("directional support requires easting and northing")
    hole_arr = np.asarray(holes, dtype=object)
    if len(hole_arr) != len(coords):
        raise ValueError("holes must have the same length as coordinates")
    edges = np.asarray(lag_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("lag_edges must contain at least two ordered edges")
    if edges[0] < 0 or not np.all(np.diff(edges) > 0):
        raise ValueError("lag_edges must be strictly increasing and nonnegative")
    axes = np.mod(np.asarray(azimuths, dtype=float), 180.0)
    widths = np.asarray(tolerances, dtype=float)
    if not len(axes) or not len(widths) or np.any(widths <= 0) or np.any(widths > 90):
        raise ValueError("azimuths and tolerances must define positive axial sectors")

    if parent_ids is None:
        parent_sets = [frozenset((f"__row_{idx}",)) for idx in range(len(coords))]
        parent_token_rows = {
            f"__row_{idx}": {idx} for idx in range(len(coords))
        }
    else:
        if len(parent_ids) != len(coords):
            raise ValueError("parent_ids must have the same length as coordinates")
        parent_sets, parent_token_rows = _parent_memberships(parent_ids)

    bins = len(edges) - 1
    counters: dict[tuple[float, float, int], dict[str, int]] = {}
    for tolerance in widths.tolist():
        for azimuth in axes.tolist():
            for bin_index in range(bins):
                counters[(tolerance, azimuth, bin_index)] = {}

    for i in range(len(coords) - 1):
        js = np.arange(i + 1, len(coords))
        linked_rows: set[int] = set()
        for token in parent_sets[i]:
            linked_rows.update(parent_token_rows[token])
        keep = np.fromiter(
            (
                hole_arr[j] != hole_arr[i] and int(j) not in linked_rows
                for j in js
            ),
            dtype=bool,
            count=len(js),
        )
        if not np.any(keep):
            continue
        js = js[keep]
        delta_e = coords[js, 0] - coords[i, 0]
        delta_n = coords[js, 1] - coords[i, 1]
        distance = np.hypot(delta_e, delta_n)
        valid = (distance > 0) & (distance <= edges[-1])
        if not np.any(valid):
            continue
        js = js[valid]
        distance = distance[valid]
        axial = np.mod(np.degrees(np.arctan2(delta_e[valid], delta_n[valid])), 180.0)
        bin_index = np.searchsorted(edges, distance, side="right") - 1
        bin_index[bin_index == bins] = bins - 1
        for j, direction, lag_bin in zip(js.tolist(), axial.tolist(), bin_index.tolist()):
            if lag_bin < 0 or lag_bin >= bins:
                continue
            pair_key = _canonical_hole_pair(hole_arr[i], hole_arr[j])
            for tolerance in widths.tolist():
                for azimuth in axes.tolist():
                    angular_difference = abs(direction - azimuth)
                    angular_difference = min(angular_difference, 180.0 - angular_difference)
                    if angular_difference <= tolerance:
                        bucket = counters[(tolerance, azimuth, lag_bin)]
                        bucket[pair_key] = bucket.get(pair_key, 0) + 1

    rows: list[dict[str, object]] = []
    for tolerance in widths.tolist():
        for azimuth in axes.tolist():
            for bin_index in range(bins):
                contributions = counters[(tolerance, azimuth, bin_index)]
                raw_pairs = int(sum(contributions.values()))
                unique_pairs = int(len(contributions))
                maximum_share = (
                    float(max(contributions.values()) / raw_pairs)
                    if raw_pairs
                    else float("nan")
                )
                rows.append(
                    {
                        "azimuth_deg": float(azimuth),
                        "tolerance_deg": float(tolerance),
                        "lag_from_m": float(edges[bin_index]),
                        "lag_to_m": float(edges[bin_index + 1]),
                        "raw_pairs": raw_pairs,
                        "unique_hole_pairs": unique_pairs,
                        "maximum_hole_pair_share": maximum_share,
                        "hole_pair_contributions_json": contributions,
                    }
                )
    return tuple(rows)


def semivariogram_values(
    distance: np.ndarray | Sequence[float],
    model: VariogramModel | ModelKind,
    *,
    range_: float | None = None,
    sill: float | None = None,
    nugget: float | None = None,
) -> np.ndarray:
    """Evaluate an isotropic semivariogram.

    The nugget is applied only at positive separation; gamma(0) is exactly zero.
    """

    h = np.asarray(distance, dtype=float)
    if isinstance(model, VariogramModel):
        model_name = model.model
        resolved_range, resolved_sill, resolved_nugget = (
            model.range,
            model.sill,
            model.nugget,
        )
    else:
        model_name = model
        if range_ is None or sill is None or nugget is None:
            raise ValueError("range_, sill, and nugget are required")
        resolved_range, resolved_sill, resolved_nugget = range_, sill, nugget
    if resolved_range <= 0 or resolved_sill < 0 or resolved_nugget < 0:
        raise ValueError("range must be positive and variance parameters nonnegative")

    ratio = np.maximum(h, 0.0) / resolved_range
    if model_name == "exponential":
        structure = 1.0 - np.exp(-3.0 * ratio)
    elif model_name == "gaussian":
        structure = 1.0 - np.exp(-3.0 * ratio * ratio)
    elif model_name == "spherical":
        structure = np.where(
            ratio < 1.0,
            1.5 * ratio - 0.5 * ratio**3,
            1.0,
        )
    else:
        raise ValueError(f"unsupported variogram model: {model_name}")
    result = resolved_nugget + resolved_sill * structure
    return np.where(h <= 0.0, 0.0, result)


def fit_variogram(
    empirical: EmpiricalVariogram,
    model: ModelKind = "exponential",
    *,
    min_pairs_per_bin: int = 5,
    min_unique_hole_pairs: int = 0,
) -> VariogramModel:
    """Fit a bounded isotropic model to supported empirical bins."""

    support = empirical.fit_support
    use = (
        empirical.supported
        & (support >= min_pairs_per_bin)
        & (empirical.unique_hole_pairs >= min_unique_hole_pairs)
    )
    lag = empirical.lag[use]
    gamma = empirical.semivariance[use]
    pairs = support[use].astype(float)
    if len(lag) < 3:
        return VariogramModel(
            model=model,
            range=np.nan,
            sill=np.nan,
            nugget=np.nan,
            rmse=np.nan,
            normalized_rmse=np.nan,
            n_bins=len(lag),
            success=False,
            message="fewer than three supported bins",
        )

    variance_scale = max(float(np.nanmax(gamma)), float(np.nanvar(gamma)), 1e-12)
    positive_lags = lag[lag > 0]
    minimum_range = max(float(np.min(positive_lags)) * 0.25, 1e-9)
    maximum_range = max(float(np.max(lag)) * 3.0, minimum_range * 2.0)
    initial = np.asarray(
        [
            np.median(positive_lags),
            max(float(np.nanmax(gamma) - np.nanmin(gamma)), variance_scale * 0.5),
            max(float(np.nanmin(gamma)), 0.0),
        ]
    )
    lower = np.asarray([minimum_range, 0.0, 0.0])
    upper = np.asarray([maximum_range, variance_scale * 5.0, variance_scale * 2.0])
    initial = np.clip(initial, lower + 1e-12, upper - 1e-12)
    weights = np.sqrt(pairs / np.max(pairs))

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = semivariogram_values(
            lag,
            model,
            range_=parameters[0],
            sill=parameters[1],
            nugget=parameters[2],
        )
        return weights * (predicted - gamma)

    try:
        optimized = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=2500,
        )
        predicted = semivariogram_values(
            lag,
            model,
            range_=optimized.x[0],
            sill=optimized.x[1],
            nugget=optimized.x[2],
        )
        rmse = float(np.sqrt(np.average((predicted - gamma) ** 2, weights=pairs)))
        normalizer = max(float(np.nanmax(gamma) - np.nanmin(gamma)), variance_scale)
        return VariogramModel(
            model=model,
            range=float(optimized.x[0]),
            sill=float(optimized.x[1]),
            nugget=float(optimized.x[2]),
            rmse=rmse,
            normalized_rmse=rmse / normalizer,
            n_bins=len(lag),
            success=bool(optimized.success),
            message=str(optimized.message),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return VariogramModel(
            model=model,
            range=np.nan,
            sill=np.nan,
            nugget=np.nan,
            rmse=np.nan,
            normalized_rmse=np.nan,
            n_bins=len(lag),
            success=False,
            message=str(exc),
        )


def variogram_sensitivity(
    coordinates: np.ndarray,
    values: Sequence[float],
    holes: Sequence[object],
    parent_ids: Sequence[object] | None = None,
    *,
    lag_counts: Iterable[int] = (8, 10, 12),
    maxlag_fractions: Iterable[float] = (0.4, 0.5, 0.6),
    estimators: Iterable[Estimator] = ("classical", "robust"),
    models: Iterable[ModelKind] = ("exponential", "spherical"),
    kind: VariogramKind = "omnidirectional",
    alonghole: Sequence[float] | None = None,
    min_pairs_per_bin: int = 5,
    min_unique_hole_pairs: int = 1,
    pair_mode: PairMode = "combined_raw",
    domains: Sequence[object] | None = None,
    pair_domain_policy: PairDomainPolicy = "pooled",
) -> tuple[tuple[EmpiricalVariogram, VariogramModel], ...]:
    """Run a frozen sensitivity family without selecting by favourable result."""

    coords = _as_2d_coordinates(coordinates)
    diagonal = float(np.linalg.norm(np.ptp(coords, axis=0)))
    if diagonal <= 0:
        raise ValueError("coordinates have no positive extent")
    results: list[tuple[EmpiricalVariogram, VariogramModel]] = []
    for lag_count in lag_counts:
        for fraction in maxlag_fractions:
            if fraction <= 0:
                raise ValueError("maxlag fractions must be positive")
            for estimator in estimators:
                empirical = empirical_variogram(
                    coords,
                    values,
                    holes,
                    parent_ids,
                    kind=kind,
                    estimator=estimator,
                    n_lags=int(lag_count),
                    maxlag=fraction * diagonal,
                    alonghole=alonghole,
                    pair_mode=pair_mode,
                    domains=domains,
                    pair_domain_policy=pair_domain_policy,
                )
                for model in models:
                    fitted = fit_variogram(
                        empirical,
                        model,
                        min_pairs_per_bin=min_pairs_per_bin,
                        min_unique_hole_pairs=(
                            0 if kind == "downhole" else min_unique_hole_pairs
                        ),
                    )
                    results.append((empirical, fitted))
    return tuple(results)


def assess_variogram_stability(
    sensitivity_results: Sequence[tuple[EmpiricalVariogram, VariogramModel]],
    thresholds: StabilityThresholds,
) -> VariogramStabilityResult:
    """Apply a predeclared stability gate to a sensitivity family."""

    successful = [
        (empirical, fitted)
        for empirical, fitted in sensitivity_results
        if fitted.success
        and np.isfinite(fitted.range)
        and np.isfinite(fitted.normalized_rmse)
        and fitted.range > 0
    ]
    reasons: list[str] = []
    if len(successful) < thresholds.min_successful_fits:
        reasons.append(
            f"{len(successful)} successful fits < "
            f"{thresholds.min_successful_fits} required"
        )

    supported_counts = [
        int(
            np.sum(
                empirical.supported
                & (
                    empirical.unique_hole_pairs
                    >= thresholds.min_unique_hole_pairs_per_bin
                )
            )
        )
        for empirical, _ in sensitivity_results
    ]
    supported_bins = min(supported_counts, default=0)
    if supported_bins < thresholds.min_supported_bins:
        reasons.append(
            f"{supported_bins} independently supported bins < "
            f"{thresholds.min_supported_bins} required"
        )

    if successful:
        ranges = np.asarray([fit.range for _, fit in successful], dtype=float)
        median_range = float(np.median(ranges))
        minimum_range = float(np.min(ranges))
        range_ratio = (
            float(np.max(ranges) / minimum_range)
            if minimum_range > 0
            else float("inf")
        )
        range_cv = (
            float(np.std(ranges, ddof=1) / np.mean(ranges))
            if len(ranges) > 1 and np.mean(ranges) > 0
            else 0.0
        )
        worst_rmse = float(max(fit.normalized_rmse for _, fit in successful))
        nugget_fractions = [
            fit.nugget / fit.total_sill if fit.total_sill > 0 else 1.0
            for _, fit in successful
        ]
        worst_nugget = float(max(nugget_fractions))
    else:
        median_range = range_ratio = range_cv = worst_rmse = worst_nugget = float(
            "inf"
        )

    if range_ratio > thresholds.max_range_ratio:
        reasons.append(
            f"range ratio {range_ratio:.3g} > {thresholds.max_range_ratio:.3g}"
        )
    if range_cv > thresholds.max_range_cv:
        reasons.append(f"range CV {range_cv:.3g} > {thresholds.max_range_cv:.3g}")
    if worst_rmse > thresholds.max_normalized_rmse:
        reasons.append(
            f"normalized RMSE {worst_rmse:.3g} > "
            f"{thresholds.max_normalized_rmse:.3g}"
        )
    if worst_nugget > thresholds.max_nugget_fraction:
        reasons.append(
            f"nugget fraction {worst_nugget:.3g} > "
            f"{thresholds.max_nugget_fraction:.3g}"
        )

    passed = not reasons
    return VariogramStabilityResult(
        passed=passed,
        decision="accept" if passed else "abstain",
        reasons=tuple(reasons) if reasons else ("all prospective checks passed",),
        successful_fits=len(successful),
        supported_bins=supported_bins,
        median_range=median_range,
        range_ratio=range_ratio,
        range_cv=range_cv,
        worst_normalized_rmse=worst_rmse,
        worst_nugget_fraction=worst_nugget,
    )


def residual_variogram_gate(
    coordinates: np.ndarray,
    residuals: Sequence[float],
    holes: Sequence[object],
    parent_ids: Sequence[object] | None,
    thresholds: StabilityThresholds,
    **sensitivity_kwargs: object,
) -> tuple[VariogramStabilityResult, tuple[tuple[EmpiricalVariogram, VariogramModel], ...]]:
    """Run and assess the frozen residual-variogram sensitivity family."""

    results = variogram_sensitivity(
        coordinates,
        residuals,
        holes,
        parent_ids,
        **sensitivity_kwargs,
    )
    return assess_variogram_stability(results, thresholds), results

