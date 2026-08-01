"""Geology-conditioned descriptive statistics for the active Tanga study.

The functions in this module are deliberately descriptive and decision-gated.
They preserve hole-level dependence, treat logged contacts as observations, and
do not convert along-hole distances into perpendicular distance or true
thickness.  Nothing here infers a weathering mechanism, metamorphic P-T
conditions, a graphitisation temperature, or deposit genesis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


CONTACT_WINDOWS_M = (1.0, 2.0, 5.0, 10.0)
VALID_RQ_RESULTS = {
    "supported",
    "unsupported",
    "insufficient evidence/abstain",
}


def decompose_depth_by_hole(
    data: pd.DataFrame,
    *,
    hole_col: str = "BHID",
    depth_col: str = "MIDPOINT",
) -> pd.DataFrame:
    """Add within-hole and between-hole depth components.

    ``depth_within_hole`` is the interval depth minus its hole mean and therefore
    sums to zero within each hole (apart from floating-point error).
    ``depth_between_hole`` is the hole mean repeated on every row.  Retaining
    both terms prevents an across-hole depth association from being presented as
    an interval-scale, within-hole relationship.
    """

    _require_columns(data, [hole_col, depth_col])
    out = data.copy()
    depth = pd.to_numeric(out[depth_col], errors="coerce")
    out["hole_mean_depth"] = depth.groupby(out[hole_col], dropna=False).transform(
        "mean"
    )
    out["depth_within_hole"] = depth - out["hole_mean_depth"]
    out["depth_between_hole"] = out["hole_mean_depth"]
    return out


def cell_declustered_mean(
    data: pd.DataFrame,
    *,
    grade_col: str = "tgc_pct",
    support_col: str = "support_m",
    easting_col: str = "mid_easting",
    northing_col: str = "mid_northing",
    cell_size_m: float,
    origin_fraction: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Return an equal-cell mean with support-weighted means inside cells.

    Grid origins are expressed as fractions of the cell size, allowing a
    deterministic origin sensitivity.  This is a clustering sensitivity, not a
    claim that any cell size is geologically optimal.
    """

    _require_columns(
        data, [grade_col, support_col, easting_col, northing_col]
    )
    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be a finite positive number")
    if len(origin_fraction) != 2:
        raise ValueError("origin_fraction must contain easting and northing")

    work = _numeric_subset(
        data,
        [grade_col, support_col, easting_col, northing_col],
    )
    work = work[work[support_col] > 0].copy()
    if work.empty:
        return float("nan")

    ox = float(origin_fraction[0]) * cell_size_m
    oy = float(origin_fraction[1]) * cell_size_m
    work["_cell_x"] = np.floor((work[easting_col] - ox) / cell_size_m)
    work["_cell_y"] = np.floor((work[northing_col] - oy) / cell_size_m)
    work["_mass"] = work[grade_col] * work[support_col]
    cell = work.groupby(["_cell_x", "_cell_y"], sort=True).agg(
        mass=("_mass", "sum"),
        support=(support_col, "sum"),
    )
    valid = cell["support"] > 0
    return float((cell.loc[valid, "mass"] / cell.loc[valid, "support"]).mean())


def summarize_support_sensitivities(
    frames: Mapping[str, pd.DataFrame],
    *,
    grade_col: str = "tgc_pct",
    support_col: str = "support_m",
    hole_col: str = "BHID",
    censor_fraction_col: str = "censored_support_fraction",
    base_censor_substitution: float = 0.025,
    censor_substitutions: Sequence[float] = (0.0, 0.05),
    cap_quantiles: Sequence[float] = (0.99, 0.995),
    cell_sizes_m: Sequence[float] = (100.0, 200.0, 400.0),
    grid_origin_fractions: Sequence[tuple[float, float]] = (
        (0.0, 0.0),
        (0.5, 0.5),
    ),
    easting_col: str = "mid_easting",
    northing_col: str = "mid_northing",
) -> pd.DataFrame:
    """Summarize support, censoring, top-cap, and declustering sensitivities.

    The supplied grade is treated as the base result using a 0.025% substitution
    for results below 0.05% TGC.  Alternative substitutions are propagated
    through each composite using its censored-support fraction.  Top caps are
    influence diagnostics only and are never silently substituted for the base
    analysis.
    """

    records: list[dict[str, Any]] = []
    for support_name, frame in frames.items():
        _require_columns(frame, [grade_col, support_col, hole_col])
        work = frame.copy()
        work[grade_col] = pd.to_numeric(work[grade_col], errors="coerce")
        work[support_col] = pd.to_numeric(work[support_col], errors="coerce")
        if censor_fraction_col in work:
            censor_fraction = pd.to_numeric(
                work[censor_fraction_col], errors="coerce"
            ).fillna(0.0)
        else:
            censor_fraction = pd.Series(0.0, index=work.index)
        work = work[
            np.isfinite(work[grade_col])
            & np.isfinite(work[support_col])
            & (work[support_col] > 0)
            & work[hole_col].notna()
        ].copy()
        censor_fraction = censor_fraction.reindex(work.index).clip(0.0, 1.0)
        if work.empty:
            continue

        variants: list[tuple[str, pd.Series, float | None]] = [
            ("base", work[grade_col].copy(), None)
        ]
        for substitution in censor_substitutions:
            adjusted = (
                work[grade_col]
                + (float(substitution) - base_censor_substitution)
                * censor_fraction
            ).clip(lower=0.0)
            variants.append(
                (f"censor_{float(substitution):g}", adjusted, float(substitution))
            )
        for quantile in cap_quantiles:
            if not 0 < float(quantile) < 1:
                raise ValueError("cap quantiles must lie strictly between 0 and 1")
            cap = float(work[grade_col].quantile(float(quantile)))
            variants.append(
                (
                    f"cap_p{100 * float(quantile):g}",
                    work[grade_col].clip(upper=cap),
                    cap,
                )
            )

        for scenario, values, parameter_value in variants:
            variant = work.copy()
            variant["_grade_variant"] = values
            common = {
                "support_name": str(support_name),
                "scenario": scenario,
                "parameter_value": parameter_value,
                "n_records": int(len(variant)),
                "n_holes": int(variant[hole_col].nunique()),
                "total_support_m": float(variant[support_col].sum()),
            }
            records.append(
                {
                    **common,
                    "estimator": "length_weighted",
                    "cell_size_m": np.nan,
                    "origin_e_fraction": np.nan,
                    "origin_n_fraction": np.nan,
                    "mean_tgc_pct": _weighted_mean(
                        variant["_grade_variant"], variant[support_col]
                    ),
                }
            )
            hole_means = _hole_means(
                variant,
                hole_col=hole_col,
                grade_col="_grade_variant",
                support_col=support_col,
            )
            records.append(
                {
                    **common,
                    "estimator": "equal_hole",
                    "cell_size_m": np.nan,
                    "origin_e_fraction": np.nan,
                    "origin_n_fraction": np.nan,
                    "mean_tgc_pct": float(hole_means.mean()),
                }
            )
            if {easting_col, northing_col}.issubset(variant.columns):
                for cell_size in cell_sizes_m:
                    for origin in grid_origin_fractions:
                        records.append(
                            {
                                **common,
                                "estimator": "equal_cell",
                                "cell_size_m": float(cell_size),
                                "origin_e_fraction": float(origin[0]),
                                "origin_n_fraction": float(origin[1]),
                                "mean_tgc_pct": cell_declustered_mean(
                                    variant,
                                    grade_col="_grade_variant",
                                    support_col=support_col,
                                    easting_col=easting_col,
                                    northing_col=northing_col,
                                    cell_size_m=float(cell_size),
                                    origin_fraction=origin,
                                ),
                            }
                        )
    return pd.DataFrame.from_records(records)


def grouped_hole_bootstrap_stats(
    data: pd.DataFrame,
    *,
    group_cols: str | Sequence[str],
    grade_col: str = "tgc_pct",
    support_col: str = "support_m",
    hole_col: str = "BHID",
    n_boot: int = 2000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return canonical-domain summaries with a hole-aware bootstrap.

    Each hole first contributes one support-weighted mean per group.  Holes, not
    intervals, are then resampled.  The bootstrap therefore targets the
    equal-hole mean and does not pretend that intervals from one hole are
    independent replicates.
    """

    if isinstance(group_cols, str):
        groups = [group_cols]
    else:
        groups = list(group_cols)
    _require_columns(data, [*groups, grade_col, support_col, hole_col])
    if n_boot < 1:
        raise ValueError("n_boot must be at least one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")

    work = data.copy()
    for col in (grade_col, support_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[
        np.isfinite(work[grade_col])
        & np.isfinite(work[support_col])
        & (work[support_col] > 0)
        & work[hole_col].notna()
    ].copy()
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    records: list[dict[str, Any]] = []

    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for key, subset in work.groupby(grouper, dropna=False, sort=True):
        keys = (key,) if len(groups) == 1 else tuple(key)
        hole_means = _hole_means(
            subset,
            hole_col=hole_col,
            grade_col=grade_col,
            support_col=support_col,
        )
        bootstrap = _bootstrap_mean(hole_means.to_numpy(), n_boot, rng)
        record = {col: value for col, value in zip(groups, keys)}
        record.update(
            {
                "n_records": int(len(subset)),
                "n_holes": int(hole_means.size),
                "total_support_m": float(subset[support_col].sum()),
                "length_weighted_mean_tgc_pct": _weighted_mean(
                    subset[grade_col], subset[support_col]
                ),
                "equal_hole_mean_tgc_pct": float(hole_means.mean()),
                "median_tgc_pct": float(subset[grade_col].median()),
                "hole_bootstrap_ci_low": float(
                    np.quantile(bootstrap, alpha)
                ),
                "hole_bootstrap_ci_high": float(
                    np.quantile(bootstrap, 1.0 - alpha)
                ),
                "bootstrap_target": "equal_hole_mean",
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_adjacent_contact_registry(
    geology: pd.DataFrame,
    *,
    hole_col: str = "BHID",
    from_col: str = "FROM",
    to_col: str = "TO",
    lithology_col: str = "canonical_lithology",
    interval_id_col: str = "geology_interval_id",
    adjacency_tolerance_m: float = 1e-6,
) -> pd.DataFrame:
    """Build verified adjacent QFR--graphitic-schist logged contacts.

    Only consecutive intervals sharing a boundary within the stated tolerance
    are retained.  A logged contact is an observed along-hole boundary, not a
    structural orientation, perpendicular distance, or true body thickness.
    The boundary policy remains ``untested`` until contact profiles and held-out
    prediction provide evidence for hard, soft, transitional, or pooled use.
    """

    required = [hole_col, from_col, to_col, lithology_col]
    _require_columns(geology, required)
    if adjacency_tolerance_m < 0:
        raise ValueError("adjacency_tolerance_m cannot be negative")
    work = geology.copy()
    work[from_col] = pd.to_numeric(work[from_col], errors="coerce")
    work[to_col] = pd.to_numeric(work[to_col], errors="coerce")
    work = work[
        work[hole_col].notna()
        & np.isfinite(work[from_col])
        & np.isfinite(work[to_col])
        & (work[to_col] > work[from_col])
    ].sort_values([hole_col, from_col, to_col], kind="mergesort")

    records: list[dict[str, Any]] = []
    for hole, intervals in work.groupby(hole_col, sort=True):
        rows = list(intervals.to_dict("records"))
        for shallow, deep in zip(rows[:-1], rows[1:]):
            gap = float(deep[from_col]) - float(shallow[to_col])
            if abs(gap) > adjacency_tolerance_m:
                continue
            pair = {
                str(shallow[lithology_col]).strip().lower(),
                str(deep[lithology_col]).strip().lower(),
            }
            if pair != {"qfr", "graphitic_schist"}:
                continue
            shallow_lith = str(shallow[lithology_col]).strip().lower()
            deep_lith = str(deep[lithology_col]).strip().lower()
            stable = not any(
                _truthy(shallow.get(flag)) or _truthy(deep.get(flag))
                for flag in (
                    "interval_key_difference",
                    "lithology_difference",
                )
            )
            records.append(
                {
                    "BHID": hole,
                    "contact_depth": (
                        float(shallow[to_col]) + float(deep[from_col])
                    )
                    / 2.0,
                    "adjacency_gap_m": gap,
                    "shallow_from": float(shallow[from_col]),
                    "shallow_to": float(shallow[to_col]),
                    "deep_from": float(deep[from_col]),
                    "deep_to": float(deep[to_col]),
                    "shallow_lithology": shallow_lith,
                    "deep_lithology": deep_lith,
                    "shallow_geology_interval_id": shallow.get(
                        interval_id_col, ""
                    ),
                    "deep_geology_interval_id": deep.get(interval_id_col, ""),
                    "graphitic_position": (
                        "shallow"
                        if shallow_lith == "graphitic_schist"
                        else "deep"
                    ),
                    "qfr_position": (
                        "shallow" if shallow_lith == "qfr" else "deep"
                    ),
                    "source_version_stable": stable,
                    "boundary_policy": "untested",
                    "contact_observation": "adjacent_logged_interval_boundary",
                    "distance_limitation": (
                        "along-hole only; not perpendicular distance or true "
                        "thickness"
                    ),
                }
            )
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "contact_id",
                "BHID",
                "contact_depth",
                "adjacency_gap_m",
                "shallow_from",
                "shallow_to",
                "deep_from",
                "deep_to",
                "shallow_lithology",
                "deep_lithology",
                "shallow_geology_interval_id",
                "deep_geology_interval_id",
                "graphitic_position",
                "qfr_position",
                "source_version_stable",
                "boundary_policy",
                "contact_observation",
                "distance_limitation",
            ]
        )
    out.insert(0, "contact_id", [f"CONTACT_{i:05d}" for i in range(1, len(out) + 1)])
    return out


def assign_signed_contact_distances(
    composites: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    hole_col: str = "BHID",
    midpoint_col: str = "MIDPOINT",
    lithology_col: str = "canonical_lithology",
    interval_tolerance_m: float = 1e-6,
    keep_unmatched: bool = True,
) -> pd.DataFrame:
    """Assign the nearest compatible logged contact to each composite.

    Positive values are graphitic-schist-side along-hole distances; negative
    values are QFR-side distances.  Compatibility requires the composite
    midpoint to lie inside the specific logged interval forming that contact,
    preventing an unrelated QFR or graphitic interval elsewhere in the hole
    from being linked to the boundary.
    """

    _require_columns(composites, [hole_col, midpoint_col, lithology_col])
    _require_columns(
        contacts,
        [
            "contact_id",
            "BHID",
            "contact_depth",
            "shallow_from",
            "shallow_to",
            "deep_from",
            "deep_to",
            "shallow_lithology",
            "deep_lithology",
        ],
    )
    out = composites.copy()
    assigned: list[dict[str, Any]] = []
    contacts_by_hole = {
        hole: group.sort_values("contact_id", kind="mergesort")
        for hole, group in contacts.groupby("BHID", sort=True)
    }
    for idx, row in out.iterrows():
        hole_contacts = contacts_by_hole.get(row[hole_col])
        midpoint = pd.to_numeric(pd.Series([row[midpoint_col]]), errors="coerce").iloc[0]
        lithology = str(row[lithology_col]).strip().lower()
        best: dict[str, Any] | None = None
        if (
            hole_contacts is not None
            and np.isfinite(midpoint)
            and lithology in {"qfr", "graphitic_schist"}
        ):
            for contact in hole_contacts.to_dict("records"):
                for position in ("shallow", "deep"):
                    if lithology != contact[f"{position}_lithology"]:
                        continue
                    lower = float(contact[f"{position}_from"])
                    upper = float(contact[f"{position}_to"])
                    if not (
                        lower - interval_tolerance_m
                        <= float(midpoint)
                        <= upper + interval_tolerance_m
                    ):
                        continue
                    absolute = abs(float(midpoint) - float(contact["contact_depth"]))
                    signed = absolute if lithology == "graphitic_schist" else -absolute
                    candidate = {
                        "contact_id": contact["contact_id"],
                        "contact_depth": float(contact["contact_depth"]),
                        "contact_side": lithology,
                        "signed_alonghole_contact_distance_m": signed,
                        "absolute_alonghole_contact_distance_m": absolute,
                        "contact_source_version_stable": contact.get(
                            "source_version_stable", np.nan
                        ),
                        "contact_boundary_policy": contact.get(
                            "boundary_policy", "untested"
                        ),
                    }
                    if best is None or (
                        absolute,
                        str(candidate["contact_id"]),
                    ) < (
                        best["absolute_alonghole_contact_distance_m"],
                        str(best["contact_id"]),
                    ):
                        best = candidate
        if best is None:
            best = {
                "contact_id": pd.NA,
                "contact_depth": np.nan,
                "contact_side": pd.NA,
                "signed_alonghole_contact_distance_m": np.nan,
                "absolute_alonghole_contact_distance_m": np.nan,
                "contact_source_version_stable": pd.NA,
                "contact_boundary_policy": pd.NA,
            }
        best["_index"] = idx
        assigned.append(best)

    assignment = pd.DataFrame.from_records(assigned).set_index("_index")
    out = out.join(assignment)
    out["contact_distance_limitation"] = (
        "along-hole only; not perpendicular distance or true thickness"
    )
    if not keep_unmatched:
        out = out[out["contact_id"].notna()].copy()
    return out


def summarize_contact_profiles(
    contact_composites: pd.DataFrame,
    *,
    windows_m: Sequence[float] = CONTACT_WINDOWS_M,
    grade_col: str = "tgc_pct",
    support_col: str = "support_m",
    hole_col: str = "BHID",
    n_boot: int = 2000,
    seed: int = 20260728,
    min_independent_holes: int = 8,
) -> pd.DataFrame:
    """Summarize cumulative signed along-hole profiles on each contact side."""

    _require_columns(
        contact_composites,
        [
            grade_col,
            support_col,
            hole_col,
            "contact_id",
            "contact_side",
            "absolute_alonghole_contact_distance_m",
        ],
    )
    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows_m):
        if float(window) <= 0:
            raise ValueError("contact windows must be positive")
        local = contact_composites[
            pd.to_numeric(
                contact_composites["absolute_alonghole_contact_distance_m"],
                errors="coerce",
            )
            <= float(window)
        ].copy()
        for side_index, side in enumerate(("qfr", "graphitic_schist")):
            subset = local[local["contact_side"] == side]
            if subset.empty:
                rows.append(
                    {
                        "window_m": float(window),
                        "side": side,
                        "n_composites": 0,
                        "n_holes": 0,
                        "n_contacts": 0,
                        "total_support_m": 0.0,
                        "length_weighted_mean_tgc_pct": np.nan,
                        "equal_hole_mean_tgc_pct": np.nan,
                        "hole_bootstrap_ci_low": np.nan,
                        "hole_bootstrap_ci_high": np.nan,
                        "evidence_status": "insufficient evidence/abstain",
                        "distance_limitation": (
                            "along-hole only; not perpendicular distance or "
                            "true thickness"
                        ),
                    }
                )
                continue
            stats = grouped_hole_bootstrap_stats(
                subset.assign(_profile_side=side),
                group_cols="_profile_side",
                grade_col=grade_col,
                support_col=support_col,
                hole_col=hole_col,
                n_boot=n_boot,
                seed=seed + 10 * window_index + side_index,
            ).iloc[0]
            n_holes = int(stats["n_holes"])
            rows.append(
                {
                    "window_m": float(window),
                    "side": side,
                    "n_composites": int(stats["n_records"]),
                    "n_holes": n_holes,
                    "n_contacts": int(subset["contact_id"].nunique()),
                    "total_support_m": float(stats["total_support_m"]),
                    "length_weighted_mean_tgc_pct": float(
                        stats["length_weighted_mean_tgc_pct"]
                    ),
                    "equal_hole_mean_tgc_pct": float(
                        stats["equal_hole_mean_tgc_pct"]
                    ),
                    "hole_bootstrap_ci_low": float(
                        stats["hole_bootstrap_ci_low"]
                    ),
                    "hole_bootstrap_ci_high": float(
                        stats["hole_bootstrap_ci_high"]
                    ),
                    "evidence_status": (
                        "descriptive"
                        if n_holes >= min_independent_holes
                        else "insufficient evidence/abstain"
                    ),
                    "distance_limitation": (
                        "along-hole only; not perpendicular distance or true "
                        "thickness"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def build_rq_evidence_table(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate and order concise research-question evidence records.

    Required fields are ``rq``, ``question``, ``evidence``, ``result``, and
    ``limitation``.  Results are intentionally restricted to supported,
    unsupported, or insufficient evidence/abstain.
    """

    required = ["rq", "question", "evidence", "result", "limitation"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"RQ record is missing fields: {missing}")
        row = dict(record)
        rq = str(row["rq"]).strip()
        if not rq or rq in seen:
            raise ValueError("RQ identifiers must be non-empty and unique")
        seen.add(rq)
        result = str(row["result"]).strip().lower()
        if result == "abstain":
            result = "insufficient evidence/abstain"
        if result not in VALID_RQ_RESULTS:
            raise ValueError(
                f"Unsupported RQ result {result!r}; expected one of "
                f"{sorted(VALID_RQ_RESULTS)}"
            )
        row["rq"] = rq
        row["result"] = result
        row.setdefault("decision", "")
        rows.append(row)
    columns = [*required, "decision"]
    extras = sorted({key for row in rows for key in row} - set(columns))
    return pd.DataFrame.from_records(rows, columns=[*columns, *extras])


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _numeric_subset(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = data.loc[:, list(columns)].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weights_array = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array > 0)
    )
    if not np.any(valid):
        return float("nan")
    return float(np.average(values_array[valid], weights=weights_array[valid]))


def _hole_means(
    data: pd.DataFrame,
    *,
    hole_col: str,
    grade_col: str,
    support_col: str,
) -> pd.Series:
    work = data[[hole_col, grade_col, support_col]].copy()
    work["_mass"] = work[grade_col] * work[support_col]
    grouped = work.groupby(hole_col, sort=True).agg(
        mass=("_mass", "sum"),
        support=(support_col, "sum"),
    )
    grouped = grouped[grouped["support"] > 0]
    return grouped["mass"] / grouped["support"]


def _bootstrap_mean(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(n_boot, np.nan)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    return values[indices].mean(axis=1)


def _truthy(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False
