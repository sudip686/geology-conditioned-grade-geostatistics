"""Run a deterministic synthetic end-to-end validation workflow.

All identifiers, coordinates, grades, and outputs made here are synthetic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nrr_study.synthetic import (
    SyntheticBenchmarkConfig,
    benchmark_gate,
    sampled_short_lag_signal_score,
)


def make_parent_intervals(seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hole_index in range(16):
        column, row = hole_index % 4, hole_index // 4
        x_m, y_m = 200.0 * column, 200.0 * row
        domain = "domain_a" if column < 2 else "domain_b"
        for parent_index in range(3):
            top = 4.0 * parent_index
            grade = (
                2.0
                + (1.4 if domain == "domain_b" else 0.0)
                + 0.06 * (top + 2.0)
                + rng.normal(0.0, 0.18)
            )
            rows.append(
                {
                    "synthetic_hole": f"SYN_H{hole_index:02d}",
                    "x_m": x_m,
                    "y_m": y_m,
                    "parent_group": f"SYN_H{hole_index:02d}_P{parent_index:02d}",
                    "from_m": top,
                    "to_m": top + 4.0,
                    "support_m": 4.0,
                    "grade_pct": grade,
                    "domain": domain,
                }
            )
    return pd.DataFrame(rows)


def composite_to_common_support(
    parents: pd.DataFrame, support_m: float = 2.0
) -> pd.DataFrame:
    if support_m <= 0:
        raise ValueError("support_m must be positive")
    children = []
    for parent in parents.to_dict(orient="records"):
        length = float(parent["to_m"] - parent["from_m"])
        pieces = int(round(length / support_m))
        if pieces < 1 or not np.isclose(pieces * support_m, length):
            raise ValueError("parent support must be an exact multiple")
        for piece in range(pieces):
            child = dict(parent)
            child["from_m"] = parent["from_m"] + piece * support_m
            child["to_m"] = child["from_m"] + support_m
            child["support_m"] = support_m
            child["synthetic_composite"] = f"{parent['parent_group']}_C{piece:02d}"
            children.append(child)
    return pd.DataFrame(children)


def support_conservation(parents: pd.DataFrame, children: pd.DataFrame) -> pd.DataFrame:
    parent = parents.assign(grade_mass=lambda x: x.grade_pct * x.support_m)
    child = children.assign(grade_mass=lambda x: x.grade_pct * x.support_m)
    expected = parent.set_index("parent_group")[["support_m", "grade_mass"]]
    observed = child.groupby("parent_group")[["support_m", "grade_mass"]].sum()
    audit = expected.join(observed, lsuffix="_expected", rsuffix="_observed")
    audit["support_conserved"] = np.isclose(
        audit.support_m_expected, audit.support_m_observed
    )
    audit["grade_mass_conserved"] = np.isclose(
        audit.grade_mass_expected, audit.grade_mass_observed
    )
    return audit.reset_index()


def grouped_buffered_registry(
    composites: pd.DataFrame, buffer_m: float = 225.0
) -> pd.DataFrame:
    holes = composites[["synthetic_hole", "x_m", "y_m"]].drop_duplicates()
    holes = holes.sort_values("synthetic_hole").reset_index(drop=True)
    holes["spatial_group"] = (holes.y_m / 200.0).round().astype(int)
    rows = []
    for group in sorted(holes.spatial_group.unique()):
        validation = holes.loc[holes.spatial_group == group]
        validation_xy = validation[["x_m", "y_m"]].to_numpy(float)
        for record in holes.to_dict(orient="records"):
            xy = np.asarray([record["x_m"], record["y_m"]], dtype=float)
            distance = float(np.min(np.linalg.norm(validation_xy - xy, axis=1)))
            if record["spatial_group"] == group:
                role = "validation"
            elif distance <= buffer_m:
                role = "excluded_buffer"
            else:
                role = "training"
            rows.append(
                {
                    **record,
                    "validation_group": int(group),
                    "role": role,
                    "nearest_validation_distance_m": distance,
                    "buffer_m": buffer_m,
                }
            )
    registry = pd.DataFrame(rows)
    return composites[["synthetic_composite", "parent_group", "synthetic_hole"]].merge(
        registry, on="synthetic_hole", validate="many_to_many"
    )


def compare_models(composites: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group in sorted(registry.validation_group.unique()):
        roles = registry.loc[registry.validation_group == group]
        role_by_hole = roles.drop_duplicates("synthetic_hole").set_index(
            "synthetic_hole"
        ).role
        train = composites[composites.synthetic_hole.map(role_by_hole) == "training"]
        test = composites[composites.synthetic_hole.map(role_by_hole) == "validation"]
        if train.empty or test.empty:
            raise RuntimeError("synthetic fold has no training or validation support")
        global_mean = float(train.grade_pct.mean())
        domain_means = train.groupby("domain").grade_pct.mean().to_dict()
        predictions = {
            "global_mean": np.full(len(test), global_mean),
            "domain_mean": test.domain.map(domain_means).fillna(global_mean).to_numpy(float),
        }
        for model, prediction in predictions.items():
            records.append(
                {
                    "model": model,
                    "validation_group": int(group),
                    "rows": int(len(test)),
                    "mae_pct": float(np.mean(np.abs(test.grade_pct - prediction))),
                    "bias_pct": float(np.mean(prediction - test.grade_pct)),
                }
            )
    per_group = pd.DataFrame(records)
    return (
        per_group.groupby("model", as_index=False)
        .agg(groups=("validation_group", "nunique"), rows=("rows", "sum"),
             mean_group_mae_pct=("mae_pct", "mean"),
             mean_group_bias_pct=("bias_pct", "mean"))
        .sort_values("model")
        .reset_index(drop=True)
    )


def equal_hole_pair_semivariance(composites: pd.DataFrame) -> dict[str, float]:
    rows = composites.reset_index(drop=True)
    pair_values: dict[str, list[float]] = {}
    for first in range(len(rows) - 1):
        for second in range(first + 1, len(rows)):
            hole_a = str(rows.at[first, "synthetic_hole"])
            hole_b = str(rows.at[second, "synthetic_hole"])
            if hole_a == hole_b:
                continue
            key = "|".join(sorted((hole_a, hole_b)))
            delta = float(rows.at[first, "grade_pct"] - rows.at[second, "grade_pct"])
            pair_values.setdefault(key, []).append(0.5 * delta * delta)
    pair_means = np.asarray([np.mean(values) for values in pair_values.values()])
    raw = np.asarray([value for values in pair_values.values() for value in values])
    return {
        "independent_hole_pairs": int(len(pair_values)),
        "equal_hole_pair_semivariance": float(np.mean(pair_means)),
        "raw_row_pair_semivariance": float(np.mean(raw)),
    }


def synthetic_gate(composites: pd.DataFrame) -> dict[str, object]:
    locations = composites[["x_m", "y_m"]].to_numpy(float)
    domains = composites.domain.to_numpy(object)
    holes = composites.synthetic_hole.to_numpy(object)

    def score(values: np.ndarray) -> float:
        return sampled_short_lag_signal_score(
            values,
            locations,
            domains=domains,
            holes=holes,
            between_hole_only=True,
            balance_hole_pairs=True,
            pair_count=1_200,
            random_state=19,
        )

    report = benchmark_gate(
        locations,
        score,
        config=SyntheticBenchmarkConfig(
            simulations_per_scenario=24,
            random_seed=19,
            n_features=12,
        ),
        domains=domains,
    )
    return asdict(report.calibration)


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parents = make_parent_intervals()
    composites = composite_to_common_support(parents)
    conservation = support_conservation(parents, composites)
    if not conservation[["support_conserved", "grade_mass_conserved"]].all().all():
        raise RuntimeError("synthetic common-support conservation failed")
    registry = grouped_buffered_registry(composites)
    comparison = compare_models(composites, registry)
    pair_audit = equal_hole_pair_semivariance(composites)
    gate = synthetic_gate(composites)
    parents.to_csv(output_dir / "synthetic_parent_intervals.csv", index=False)
    composites.to_csv(output_dir / "synthetic_common_support.csv", index=False)
    conservation.to_csv(output_dir / "synthetic_support_audit.csv", index=False)
    registry.to_csv(output_dir / "synthetic_grouped_buffer_registry.csv", index=False)
    comparison.to_csv(output_dir / "synthetic_model_comparison.csv", index=False)
    (output_dir / "synthetic_pair_weighting.json").write_text(
        json.dumps(pair_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "synthetic_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "scope": "synthetic_only",
        "parents": int(len(parents)),
        "composites": int(len(composites)),
        "support_conservation_passed": True,
        "parent_grouping_passed": bool(
            registry.groupby(["validation_group", "parent_group"]).role.nunique().max() == 1
        ),
        "buffering_passed": bool(
            (registry.loc[registry.role == "training", "nearest_validation_distance_m"]
             > registry.loc[registry.role == "training", "buffer_m"]).all()
        ),
        "model_comparison": comparison.to_dict(orient="records"),
        "pair_weighting": pair_audit,
        "gate": gate,
    }
    (output_dir / "synthetic_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_outputs"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
