"""Run a small deterministic synthetic gate calibration."""

from dataclasses import asdict
import json

import numpy as np

from nrr_study.synthetic import (
    SyntheticBenchmarkConfig,
    benchmark_gate,
    sampled_short_lag_signal_score,
)


def main() -> None:
    locations = np.column_stack(
        [np.linspace(0.0, 120.0, 24), np.zeros(24)]
    )
    domains = np.where(locations[:, 0] < 60.0, "domain_a", "domain_b")
    groups = np.asarray([f"H{index:02d}" for index in range(len(locations))])

    def score(values: np.ndarray) -> float:
        return sampled_short_lag_signal_score(
            values,
            locations,
            domains=domains,
            holes=groups,
            between_hole_only=True,
            balance_hole_pairs=True,
            pair_count=2_000,
            random_state=19,
        )

    report = benchmark_gate(
        locations,
        score,
        config=SyntheticBenchmarkConfig(
            simulations_per_scenario=20,
            random_seed=19,
            n_features=16,
        ),
        domains=domains,
    )
    print(json.dumps(asdict(report.calibration), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
