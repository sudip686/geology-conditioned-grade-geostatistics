"""Prediction models for canonical drillhole composites.

All models predict observed composite support only.  They do not construct
blocks or resource outputs.  Spatial parameters are constructor arguments so
the validation layer can tune them inside grouped training folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .geostat import (
    Estimator,
    ModelKind,
    PairDomainPolicy,
    PairMode,
    StabilityThresholds,
    VariogramModel,
    VariogramStabilityResult,
    assess_variogram_stability,
    empirical_variogram,
    fit_variogram,
    semivariogram_values,
    variogram_sensitivity,
)


@dataclass(frozen=True)
class PredictionResult:
    """Mean, variance, and per-row prediction success."""

    mean: np.ndarray
    variance: np.ndarray
    success: np.ndarray


class CompositeRegressor(Protocol):
    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "CompositeRegressor": ...

    def predict(self, data: pd.DataFrame) -> PredictionResult: ...


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _coordinates(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    _require_columns(data, columns)
    coordinates = data.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("spatial coordinates contain non-finite values")
    return coordinates


def _parent_sets(values: Sequence[object]) -> list[frozenset[str]]:
    result: list[frozenset[str]] = []
    for value in values:
        if pd.isna(value):
            result.append(frozenset())
        else:
            result.append(
                frozenset(
                    token.strip()
                    for token in str(value).split("|")
                    if token.strip()
                )
            )
    return result


class GlobalMeanRegressor:
    """Weighted global-mean baseline."""

    def __init__(self, weight_col: str | None = "support_m") -> None:
        self.weight_col = weight_col

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "GlobalMeanRegressor":
        _require_columns(data, [target_col])
        y = data[target_col].to_numpy(dtype=float)
        if self.weight_col is not None and self.weight_col in data:
            weights = data[self.weight_col].to_numpy(dtype=float)
        else:
            weights = np.ones(len(data), dtype=float)
        valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        if not np.any(valid):
            raise ValueError("no valid training observations")
        self.mean_ = float(np.average(y[valid], weights=weights[valid]))
        self.variance_ = float(
            np.average((y[valid] - self.mean_) ** 2, weights=weights[valid])
        )
        return self

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "mean_"):
            raise RuntimeError("model is not fitted")
        return PredictionResult(
            mean=np.full(len(data), self.mean_, dtype=float),
            variance=np.full(len(data), self.variance_, dtype=float),
            success=np.ones(len(data), dtype=bool),
        )


class GeologyRegressionRegressor:
    """Ridge regression with categorical geology and numeric context.

    A near-zero ``alpha`` is the geology-regression comparator.  Positive
    ``alpha`` provides partial pooling for sparse lithology levels.
    """

    def __init__(
        self,
        *,
        categorical_cols: Sequence[str] = (
            "canonical_lithology",
            "weathering",
        ),
        numeric_cols: Sequence[str] = ("mid_tvd", "mid_rl"),
        alpha: float = 1.0,
        weight_col: str | None = "support_m",
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.categorical_cols = tuple(categorical_cols)
        self.numeric_cols = tuple(numeric_cols)
        self.alpha = float(alpha)
        self.weight_col = weight_col

    def _make_pipeline(self) -> Pipeline:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("categorical", categorical, list(self.categorical_cols)),
                ("numeric", numeric, list(self.numeric_cols)),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("features", transformer),
                ("regressor", Ridge(alpha=max(self.alpha, 1e-12))),
            ]
        )

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "GeologyRegressionRegressor":
        features = self.categorical_cols + self.numeric_cols
        _require_columns(data, [target_col, *features])
        y = data[target_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("training target contains non-finite values")
        self.pipeline_ = self._make_pipeline()
        fit_kwargs: dict[str, np.ndarray] = {}
        if self.weight_col is not None and self.weight_col in data:
            weights = data[self.weight_col].to_numpy(dtype=float)
            if not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("sample weights must be finite and positive")
            fit_kwargs["regressor__sample_weight"] = weights
        self.pipeline_.fit(data.loc[:, list(features)], y, **fit_kwargs)
        fitted = self.pipeline_.predict(data.loc[:, list(features)])
        residual = y - fitted
        if fit_kwargs:
            weights = fit_kwargs["regressor__sample_weight"]
            self.residual_variance_ = float(
                np.average(residual * residual, weights=weights)
            )
        else:
            self.residual_variance_ = float(np.mean(residual * residual))
        self.target_col_ = target_col
        return self

    def residualize_matrix(
        self,
        data: pd.DataFrame,
        response_matrix: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Fit the declared trend to every response column and return residuals.

        The same feature pipeline and optional support weights used by
        :meth:`fit` are applied in one multi-response ridge fit. This permits
        synthetic null and positive fields to undergo the identical
        geology/depth trend removal used for the observed residuals without
        mutating the fitted state of this regressor.
        """

        features = self.categorical_cols + self.numeric_cols
        _require_columns(data, features)
        responses = np.asarray(response_matrix, dtype=float)
        if responses.ndim != 2 or responses.shape[0] != len(data):
            raise ValueError(
                "response_matrix must be a two-dimensional array with one "
                "row per data record"
            )
        if responses.shape[1] < 1 or not np.all(np.isfinite(responses)):
            raise ValueError(
                "response_matrix must contain at least one finite response"
            )
        pipeline = self._make_pipeline()
        fit_kwargs: dict[str, np.ndarray] = {}
        if self.weight_col is not None and self.weight_col in data:
            weights = data[self.weight_col].to_numpy(dtype=float)
            if not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("sample weights must be finite and positive")
            fit_kwargs["regressor__sample_weight"] = weights
        feature_frame = data.loc[:, list(features)]
        pipeline.fit(feature_frame, responses, **fit_kwargs)
        fitted = np.asarray(pipeline.predict(feature_frame), dtype=float)
        if fitted.shape != responses.shape:
            raise RuntimeError("trend pipeline changed the response-matrix shape")
        return responses - fitted

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("model is not fitted")
        features = self.categorical_cols + self.numeric_cols
        _require_columns(data, features)
        mean = np.asarray(
            self.pipeline_.predict(data.loc[:, list(features)]), dtype=float
        )
        return PredictionResult(
            mean=mean,
            variance=np.full(len(data), self.residual_variance_, dtype=float),
            success=np.isfinite(mean),
        )


class IDWRegressor:
    """Inverse-distance weighting comparator with optional hard domains."""

    def __init__(
        self,
        *,
        coordinate_cols: Sequence[str] = (
            "mid_easting",
            "mid_northing",
            "mid_rl",
        ),
        power: float = 2.0,
        max_neighbors: int | None = 32,
        search_radius: float | None = None,
        min_neighbors: int = 1,
        domain_col: str | None = None,
        parent_col: str = "parent_assay_ids",
    ) -> None:
        if power <= 0:
            raise ValueError("power must be positive")
        if max_neighbors is not None and max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        if search_radius is not None and search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if min_neighbors < 1:
            raise ValueError("min_neighbors must be positive")
        self.coordinate_cols = tuple(coordinate_cols)
        self.power = float(power)
        self.max_neighbors = max_neighbors
        self.search_radius = search_radius
        self.min_neighbors = int(min_neighbors)
        self.domain_col = domain_col
        self.parent_col = parent_col

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "IDWRegressor":
        required = [target_col, *self.coordinate_cols]
        if self.domain_col is not None:
            required.append(self.domain_col)
        _require_columns(data, required)
        self.coordinates_ = _coordinates(data, self.coordinate_cols)
        self.values_ = data[target_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(self.values_)):
            raise ValueError("training target contains non-finite values")
        self.domains_ = (
            data[self.domain_col].astype(str).to_numpy()
            if self.domain_col is not None
            else None
        )
        self.parents_ = (
            _parent_sets(data[self.parent_col])
            if self.parent_col in data
            else [frozenset() for _ in range(len(data))]
        )
        return self

    def _eligible(
        self,
        distances: np.ndarray,
        domain: str | None,
        parents: frozenset[str],
    ) -> np.ndarray:
        eligible = np.isfinite(distances) & (distances > 0)
        if self.search_radius is not None:
            eligible &= distances <= self.search_radius
        if self.domains_ is not None:
            eligible &= self.domains_ == str(domain)
        if parents:
            eligible &= np.asarray(
                [not bool(parents & item) for item in self.parents_], dtype=bool
            )
        return np.flatnonzero(eligible)

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "coordinates_"):
            raise RuntimeError("model is not fitted")
        required = list(self.coordinate_cols)
        if self.domain_col is not None:
            required.append(self.domain_col)
        _require_columns(data, required)
        query = _coordinates(data, self.coordinate_cols)
        distance_matrix = cdist(query, self.coordinates_)
        domains = (
            data[self.domain_col].astype(str).to_numpy()
            if self.domain_col is not None
            else np.full(len(data), None, dtype=object)
        )
        parents = (
            _parent_sets(data[self.parent_col])
            if self.parent_col in data
            else [frozenset() for _ in range(len(data))]
        )
        mean = np.full(len(data), np.nan)
        variance = np.full(len(data), np.nan)
        success = np.zeros(len(data), dtype=bool)
        for row, distances in enumerate(distance_matrix):
            eligible = self._eligible(distances, domains[row], parents[row])
            if self.max_neighbors is not None and len(eligible) > self.max_neighbors:
                eligible = eligible[
                    np.argsort(distances[eligible], kind="mergesort")[
                        : self.max_neighbors
                    ]
                ]
            if len(eligible) < self.min_neighbors:
                continue
            d = np.maximum(distances[eligible], np.finfo(float).eps)
            weights = np.power(d, -self.power)
            weights /= np.sum(weights)
            local = self.values_[eligible]
            estimate = float(np.dot(weights, local))
            mean[row] = estimate
            variance[row] = float(np.dot(weights, (local - estimate) ** 2))
            success[row] = True
        return PredictionResult(mean=mean, variance=variance, success=success)


class OrdinaryKrigingRegressor(IDWRegressor):
    """Isotropic ordinary kriging at held-out composite locations."""

    def __init__(
        self,
        variogram: VariogramModel,
        *,
        coordinate_cols: Sequence[str] = (
            "mid_easting",
            "mid_northing",
            "mid_rl",
        ),
        max_neighbors: int | None = 32,
        search_radius: float | None = None,
        min_neighbors: int = 3,
        domain_col: str | None = None,
        parent_col: str = "parent_assay_ids",
        regularization: float = 1e-10,
    ) -> None:
        if not variogram.success:
            raise ValueError("ordinary kriging requires a successful variogram fit")
        super().__init__(
            coordinate_cols=coordinate_cols,
            power=2.0,
            max_neighbors=max_neighbors,
            search_radius=search_radius,
            min_neighbors=min_neighbors,
            domain_col=domain_col,
            parent_col=parent_col,
        )
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        self.variogram = variogram
        self.regularization = float(regularization)

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "coordinates_"):
            raise RuntimeError("model is not fitted")
        required = list(self.coordinate_cols)
        if self.domain_col is not None:
            required.append(self.domain_col)
        _require_columns(data, required)
        query = _coordinates(data, self.coordinate_cols)
        distance_matrix = cdist(query, self.coordinates_)
        domains = (
            data[self.domain_col].astype(str).to_numpy()
            if self.domain_col is not None
            else np.full(len(data), None, dtype=object)
        )
        parents = (
            _parent_sets(data[self.parent_col])
            if self.parent_col in data
            else [frozenset() for _ in range(len(data))]
        )
        mean = np.full(len(data), np.nan)
        variance = np.full(len(data), np.nan)
        success = np.zeros(len(data), dtype=bool)

        for row, distances in enumerate(distance_matrix):
            eligible = self._eligible(distances, domains[row], parents[row])
            if self.max_neighbors is not None and len(eligible) > self.max_neighbors:
                eligible = eligible[
                    np.argsort(distances[eligible], kind="mergesort")[
                        : self.max_neighbors
                    ]
                ]
            if len(eligible) < self.min_neighbors:
                continue

            local_coords = self.coordinates_[eligible]
            pair_distance = cdist(local_coords, local_coords)
            gamma_matrix = semivariogram_values(pair_distance, self.variogram)
            if self.regularization:
                gamma_matrix = gamma_matrix.copy()
                gamma_matrix.flat[:: len(eligible) + 1] += self.regularization
            gamma_query = semivariogram_values(distances[eligible], self.variogram)
            system = np.empty((len(eligible) + 1, len(eligible) + 1), dtype=float)
            system[:-1, :-1] = gamma_matrix
            system[:-1, -1] = 1.0
            system[-1, :-1] = 1.0
            system[-1, -1] = 0.0
            right = np.concatenate([gamma_query, [1.0]])
            try:
                solution = np.linalg.solve(system, right)
            except np.linalg.LinAlgError:
                solution, *_ = np.linalg.lstsq(system, right, rcond=None)
            weights, multiplier = solution[:-1], solution[-1]
            if not np.all(np.isfinite(weights)):
                continue
            mean[row] = float(np.dot(weights, self.values_[eligible]))
            variance[row] = max(
                float(np.dot(weights, gamma_query) + multiplier), 0.0
            )
            success[row] = np.isfinite(mean[row]) and np.isfinite(variance[row])
        return PredictionResult(mean=mean, variance=variance, success=success)


class RegressionKrigingRegressor:
    """Geological trend plus residual ordinary kriging.

    A successful prospective residual-variogram gate must be passed explicitly.
    Failed residual kriging falls back to the trend mean while retaining a
    ``False`` success flag so validation reports the failure rate.
    """

    def __init__(
        self,
        trend_model: CompositeRegressor,
        residual_variogram: VariogramModel,
        *,
        stability_gate_passed: bool,
        coordinate_cols: Sequence[str] = (
            "mid_easting",
            "mid_northing",
            "mid_rl",
        ),
        max_neighbors: int | None = 32,
        search_radius: float | None = None,
        min_neighbors: int = 3,
        domain_col: str | None = None,
        parent_col: str = "parent_assay_ids",
    ) -> None:
        if not stability_gate_passed:
            raise ValueError(
                "regression kriging is disabled because the residual "
                "variogram gate did not pass"
            )
        self.trend_model = trend_model
        self.residual_variogram = residual_variogram
        self.coordinate_cols = tuple(coordinate_cols)
        self.max_neighbors = max_neighbors
        self.search_radius = search_radius
        self.min_neighbors = min_neighbors
        self.domain_col = domain_col
        self.parent_col = parent_col

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "RegressionKrigingRegressor":
        self.trend_model.fit(data, target_col)
        trend = self.trend_model.predict(data)
        residual = data[target_col].to_numpy(dtype=float) - trend.mean
        residual_data = data.copy()
        residual_data["_nrr_residual"] = residual
        self.residual_model_ = OrdinaryKrigingRegressor(
            self.residual_variogram,
            coordinate_cols=self.coordinate_cols,
            max_neighbors=self.max_neighbors,
            search_radius=self.search_radius,
            min_neighbors=self.min_neighbors,
            domain_col=self.domain_col,
            parent_col=self.parent_col,
        ).fit(residual_data, "_nrr_residual")
        return self

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "residual_model_"):
            raise RuntimeError("model is not fitted")
        trend = self.trend_model.predict(data)
        residual = self.residual_model_.predict(data)
        residual_mean = np.where(residual.success, residual.mean, 0.0)
        mean = trend.mean + residual_mean
        variance = np.where(
            residual.success & np.isfinite(residual.variance),
            residual.variance,
            0.0,
        )
        return PredictionResult(
            mean=mean,
            variance=variance,
            success=trend.success & residual.success,
        )


class FoldLocalVariogramError(RuntimeError):
    """A fold could not support the declared covariance model."""

    def __init__(self, stage: str, reasons: Sequence[str]) -> None:
        self.stage = str(stage)
        self.reasons = tuple(str(reason) for reason in reasons)
        message = "; ".join(self.reasons) or "unspecified fold-local failure"
        super().__init__(f"fold-local variogram {self.stage} failed: {message}")


class FoldLocalKrigingRegressor:
    """Ordinary or regression kriging with covariance fitted per train fold.

    Every call to :meth:`fit` constructs the residual (or raw-grade)
    variogram exclusively from the supplied training rows, applies the
    declared stability gate, and then fits one fixed operational variogram
    specification. It never selects a favourable model fitted to the full
    analysis cohort. Prospective mode remains fail-closed. Explicit diagnostic
    mode may omit the repeated sensitivity family and continue past a failed
    fixed-fit stability check, but it still raises when the operational fit or
    minimum independent-hole/bin support is unavailable.
    """

    def __init__(
        self,
        *,
        trend_model: CompositeRegressor | None = None,
        stability_thresholds: StabilityThresholds = StabilityThresholds(),
        coordinate_cols: Sequence[str] = (
            "mid_easting",
            "mid_northing",
            "mid_rl",
        ),
        hole_col: str = "BHID",
        parent_col: str = "parent_assay_ids",
        pair_mode: PairMode = "between_hole_balanced",
        pair_domain_policy: PairDomainPolicy = "same_domain",
        pair_domain_col: str | None = "canonical_lithology",
        prediction_domain_col: str | None = None,
        sensitivity_lag_counts: Sequence[int] = (8, 10),
        sensitivity_maxlag_fractions: Sequence[float] = (0.4, 0.5),
        sensitivity_estimators: Sequence[Estimator] = (
            "classical",
            "robust",
        ),
        sensitivity_models: Sequence[ModelKind] = (
            "exponential",
            "spherical",
        ),
        operational_n_lags: int = 10,
        operational_maxlag_fraction: float = 0.5,
        operational_estimator: Estimator = "classical",
        operational_model: ModelKind = "exponential",
        min_pairs_per_bin: int = 5,
        min_unique_hole_pairs: int = 5,
        max_neighbors: int | None = 32,
        search_radius_multiplier: float = 1.0,
        min_neighbors: int = 3,
        execution_policy: Literal[
            "prospective_gate", "diagnostic_sensitivity"
        ] = "prospective_gate",
        fit_sensitivity_family: bool = True,
    ) -> None:
        if pair_domain_policy == "same_domain" and pair_domain_col is None:
            raise ValueError(
                "same_domain covariance requires pair_domain_col"
            )
        if operational_n_lags < 2:
            raise ValueError("operational_n_lags must be at least 2")
        if operational_maxlag_fraction <= 0:
            raise ValueError("operational_maxlag_fraction must be positive")
        if min_pairs_per_bin < 1 or min_unique_hole_pairs < 1:
            raise ValueError("minimum pair supports must be positive")
        if search_radius_multiplier <= 0:
            raise ValueError("search_radius_multiplier must be positive")
        if execution_policy not in {
            "prospective_gate",
            "diagnostic_sensitivity",
        }:
            raise ValueError(
                "execution_policy must be prospective_gate or "
                "diagnostic_sensitivity"
            )
        if execution_policy == "prospective_gate" and not fit_sensitivity_family:
            raise ValueError(
                "prospective_gate requires the complete fold sensitivity family"
            )
        self.trend_model = trend_model
        self.stability_thresholds = stability_thresholds
        self.coordinate_cols = tuple(coordinate_cols)
        self.hole_col = hole_col
        self.parent_col = parent_col
        self.pair_mode = pair_mode
        self.pair_domain_policy = pair_domain_policy
        self.pair_domain_col = pair_domain_col
        self.prediction_domain_col = prediction_domain_col
        self.sensitivity_lag_counts = tuple(int(x) for x in sensitivity_lag_counts)
        self.sensitivity_maxlag_fractions = tuple(
            float(x) for x in sensitivity_maxlag_fractions
        )
        self.sensitivity_estimators = tuple(sensitivity_estimators)
        self.sensitivity_models = tuple(sensitivity_models)
        self.operational_n_lags = int(operational_n_lags)
        self.operational_maxlag_fraction = float(operational_maxlag_fraction)
        self.operational_estimator = operational_estimator
        self.operational_model = operational_model
        self.min_pairs_per_bin = int(min_pairs_per_bin)
        self.min_unique_hole_pairs = int(min_unique_hole_pairs)
        self.max_neighbors = max_neighbors
        self.search_radius_multiplier = float(search_radius_multiplier)
        self.min_neighbors = int(min_neighbors)
        self.execution_policy = execution_policy
        self.fit_sensitivity_family = bool(fit_sensitivity_family)

    def _covariance_target(
        self, data: pd.DataFrame, target_col: str
    ) -> np.ndarray:
        target = data[target_col].to_numpy(dtype=float)
        if self.trend_model is None:
            return target
        self.trend_model.fit(data, target_col)
        trend = self.trend_model.predict(data)
        if not np.all(trend.success & np.isfinite(trend.mean)):
            raise FoldLocalVariogramError(
                "trend",
                ("training-fold trend did not predict every training row",),
            )
        return target - trend.mean

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "FoldLocalKrigingRegressor":
        required = [
            target_col,
            self.hole_col,
            self.parent_col,
            *self.coordinate_cols,
        ]
        if self.pair_domain_col is not None:
            required.append(self.pair_domain_col)
        if self.prediction_domain_col is not None:
            required.append(self.prediction_domain_col)
        _require_columns(data, required)
        coordinates = _coordinates(data, self.coordinate_cols)
        target = data[target_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError("training target contains non-finite values")
        holes = data[self.hole_col].astype(str).to_numpy()
        if len(pd.unique(holes)) < 2:
            raise FoldLocalVariogramError(
                "support", ("fewer than two independent training holes",)
            )
        parents = data[self.parent_col].astype(str).to_numpy()
        domains = (
            data[self.pair_domain_col].astype(str).to_numpy()
            if self.pair_domain_col is not None
            else None
        )
        covariance_target = self._covariance_target(data, target_col)

        diagonal = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise FoldLocalVariogramError(
                "support", ("training coordinates have no positive extent",)
            )
        operational_empirical = empirical_variogram(
            coordinates,
            covariance_target,
            holes,
            parents,
            kind="omnidirectional",
            estimator=self.operational_estimator,
            n_lags=self.operational_n_lags,
            maxlag=self.operational_maxlag_fraction * diagonal,
            pair_mode=self.pair_mode,
            domains=domains,
            pair_domain_policy=self.pair_domain_policy,
        )
        operational = fit_variogram(
            operational_empirical,
            self.operational_model,
            min_pairs_per_bin=self.min_pairs_per_bin,
            min_unique_hole_pairs=self.min_unique_hole_pairs,
        )
        if not operational.success or not np.isfinite(operational.range):
            raise FoldLocalVariogramError(
                "operational_fit",
                (operational.message or "fixed variogram fit was unsupported",),
            )
        operational_supported_bins = int(
            np.sum(
                operational_empirical.supported
                & (
                    operational_empirical.unique_hole_pairs
                    >= self.min_unique_hole_pairs
                )
            )
        )
        if operational_supported_bins < self.stability_thresholds.min_supported_bins:
            raise FoldLocalVariogramError(
                "support",
                (
                    f"{operational_supported_bins} independently supported bins < "
                    f"{self.stability_thresholds.min_supported_bins} required",
                ),
            )

        if self.fit_sensitivity_family:
            try:
                sensitivity = variogram_sensitivity(
                    coordinates,
                    covariance_target,
                    holes,
                    parents,
                    lag_counts=self.sensitivity_lag_counts,
                    maxlag_fractions=self.sensitivity_maxlag_fractions,
                    estimators=self.sensitivity_estimators,
                    models=self.sensitivity_models,
                    kind="omnidirectional",
                    min_pairs_per_bin=self.min_pairs_per_bin,
                    min_unique_hole_pairs=self.min_unique_hole_pairs,
                    pair_mode=self.pair_mode,
                    domains=domains,
                    pair_domain_policy=self.pair_domain_policy,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                raise FoldLocalVariogramError("support", (str(exc),)) from exc
            gate = assess_variogram_stability(
                sensitivity, self.stability_thresholds
            )
            self.fold_sensitivity_ = sensitivity
        else:
            nugget_fraction = (
                operational.nugget / operational.total_sill
                if operational.total_sill > 0
                else 1.0
            )
            reasons = []
            if operational.normalized_rmse > self.stability_thresholds.max_normalized_rmse:
                reasons.append(
                    f"normalized RMSE {operational.normalized_rmse:.3g} > "
                    f"{self.stability_thresholds.max_normalized_rmse:.3g}"
                )
            if nugget_fraction > self.stability_thresholds.max_nugget_fraction:
                reasons.append(
                    f"nugget fraction {nugget_fraction:.3g} > "
                    f"{self.stability_thresholds.max_nugget_fraction:.3g}"
                )
            gate = VariogramStabilityResult(
                passed=not reasons,
                decision="accept" if not reasons else "abstain",
                reasons=tuple(reasons) if reasons else ("fixed operational fit passed",),
                successful_fits=1,
                supported_bins=operational_supported_bins,
                median_range=operational.range,
                range_ratio=float("nan"),
                range_cv=float("nan"),
                worst_normalized_rmse=operational.normalized_rmse,
                worst_nugget_fraction=nugget_fraction,
            )
            self.fold_sensitivity_ = ()
        self.fold_gate_: VariogramStabilityResult = gate
        self.minimum_support_passed_ = True
        self.continued_past_failed_stability_ = False
        if not gate.passed and self.execution_policy == "prospective_gate":
            raise FoldLocalVariogramError("stability", gate.reasons)
        if not gate.passed:
            self.continued_past_failed_stability_ = True
        self.fold_variogram_ = operational
        self.operational_empirical_ = operational_empirical
        spatial_data = data.copy()
        spatial_data["_nrr_fold_covariance_target"] = covariance_target
        self.spatial_model_ = OrdinaryKrigingRegressor(
            operational,
            coordinate_cols=self.coordinate_cols,
            max_neighbors=self.max_neighbors,
            search_radius=(
                self.search_radius_multiplier * operational.range
            ),
            min_neighbors=self.min_neighbors,
            domain_col=self.prediction_domain_col,
            parent_col=self.parent_col,
        ).fit(spatial_data, "_nrr_fold_covariance_target")
        self.target_col_ = target_col
        return self

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "spatial_model_"):
            raise RuntimeError("model is not fitted")
        spatial = self.spatial_model_.predict(data)
        if self.trend_model is None:
            return spatial
        trend = self.trend_model.predict(data)
        spatial_mean = np.where(spatial.success, spatial.mean, 0.0)
        mean = trend.mean + spatial_mean
        spatial_variance = np.where(
            spatial.success & np.isfinite(spatial.variance),
            spatial.variance,
            0.0,
        )
        return PredictionResult(
            mean=mean,
            variance=spatial_variance,
            success=trend.success & spatial.success,
        )
