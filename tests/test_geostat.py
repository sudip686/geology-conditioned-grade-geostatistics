import numpy as np

from nrr_study.geostat import (
    EmpiricalVariogram,
    StabilityThresholds,
    VariogramModel,
    assess_variogram_stability,
    directional_pair_support,
    empirical_variogram,
    fit_variogram,
    semivariogram_values,
)
from nrr_study.synthetic import (
    SyntheticBenchmarkConfig,
    benchmark_gate,
    calibrate_gate_threshold,
    generate_all_scenarios,
    sampled_short_lag_signal_score,
)


def test_pair_accounting_excludes_same_parent_and_tracks_holes():
    coordinates = np.column_stack([np.arange(6.0), np.zeros(6)])
    values = np.asarray([0.0, 0.1, 0.5, 1.0, 1.3, 1.8])
    holes = np.asarray(["A", "A", "A", "B", "B", "B"])
    parents = np.asarray(["p1|shared", "shared|p2", "p7", "p3", "p4", "p5"])

    omni = empirical_variogram(
        coordinates,
        values,
        holes,
        parents,
        n_lags=3,
        maxlag=10.0,
    )
    assert omni.excluded_same_parent_pairs == 1
    assert int(omni.raw_pairs.sum()) == 14
    assert int(omni.same_hole_pairs.sum()) == 5
    assert any(
        "A|B" in contribution
        for contribution in omni.hole_pair_contributions
    )

    downhole = empirical_variogram(
        coordinates,
        values,
        holes,
        parents,
        kind="downhole",
        alonghole=np.arange(6.0),
        n_lags=3,
        maxlag=10.0,
    )
    assert int(downhole.raw_pairs.sum()) == 5
    assert int(downhole.same_hole_pairs.sum()) == 5
    assert int(downhole.unique_hole_pairs.sum()) == 0


def test_classical_and_robust_variograms_are_finite():
    coordinates = np.column_stack([np.arange(12.0), np.zeros(12)])
    values = np.sin(np.arange(12.0) / 3.0)
    values[-1] += 8.0
    holes = np.repeat(["A", "B", "C"], 4)
    parents = np.asarray([f"p{i}" for i in range(12)])
    classical = empirical_variogram(
        coordinates,
        values,
        holes,
        parents,
        n_lags=4,
        maxlag=11.0,
        estimator="classical",
    )
    robust = empirical_variogram(
        coordinates,
        values,
        holes,
        parents,
        n_lags=4,
        maxlag=11.0,
        estimator="robust",
    )
    assert np.all(np.isfinite(classical.semivariance[classical.supported]))
    assert np.all(np.isfinite(robust.semivariance[robust.supported]))
    assert not np.allclose(
        classical.semivariance[classical.supported],
        robust.semivariance[robust.supported],
    )


def test_between_hole_balancing_gives_each_hole_pair_equal_lag_weight():
    coordinates = np.column_stack(
        [np.asarray([0.0, 0.1, 0.2, 0.3, 1.0, 2.0]), np.zeros(6)]
    )
    values = np.asarray([0.0, 0.0, 0.0, 0.0, 2.0, 10.0])
    holes = np.asarray(["A", "A", "A", "A", "B", "C"])
    parents = np.asarray([f"p{i}" for i in range(6)])
    empirical = empirical_variogram(
        coordinates,
        values,
        holes,
        parents,
        bin_edges=(0.0, 5.0, 10.0),
        pair_mode="between_hole_balanced",
    )
    # A-B, A-C, and B-C contribute semivariances 2, 50, and 32. Equal
    # hole-pair weighting therefore gives 28 rather than a row-pair-weighted
    # value dominated by the four A observations.
    assert np.isclose(empirical.semivariance[0], 28.0)
    assert empirical.raw_pairs[0] == 9
    assert empirical.unique_hole_pairs[0] == 3
    assert empirical.same_hole_pairs[0] == 0
    assert empirical.fit_support[0] == 3


def test_same_domain_between_hole_variogram_excludes_cross_domain_pairs():
    coordinates = np.column_stack([np.arange(3.0), np.zeros(3)])
    empirical = empirical_variogram(
        coordinates,
        [0.0, 2.0, 10.0],
        ["A", "B", "C"],
        ["p1", "p2", "p3"],
        bin_edges=(0.0, 5.0, 10.0),
        pair_mode="between_hole_balanced",
        domains=["X", "X", "Y"],
        pair_domain_policy="same_domain",
    )
    assert empirical.excluded_cross_domain_pairs == 2
    assert empirical.raw_pairs[0] == 1
    assert empirical.unique_hole_pairs[0] == 1
    assert np.isclose(empirical.semivariance[0], 2.0)


def test_directional_pair_support_is_grade_blind_and_deterministic():
    coordinates = np.asarray(
        [
            [0.0, 0.0, 100.0],
            [0.0, 10.0, 90.0],
            [10.0, 0.0, 90.0],
            [10.0, 10.0, 80.0],
        ]
    )
    arguments = dict(
        holes=["A", "B", "C", "D"],
        parent_ids=["p1", "p2", "p3", "p4"],
        azimuths=(0.0, 45.0, 90.0, 135.0),
        tolerances=(10.0,),
        lag_edges=(0.0, 11.0, 20.0),
    )
    first = directional_pair_support(coordinates, **arguments)
    second = directional_pair_support(coordinates, **arguments)
    for first_row, second_row in zip(first, second):
        assert first_row.keys() == second_row.keys()
        for key in first_row:
            first_value, second_value = first_row[key], second_row[key]
            if isinstance(first_value, float) and np.isnan(first_value):
                assert isinstance(second_value, float) and np.isnan(second_value)
            else:
                assert first_value == second_value
    north_short = next(
        row
        for row in first
        if row["azimuth_deg"] == 0.0 and row["lag_from_m"] == 0.0
    )
    assert north_short["raw_pairs"] == 2
    assert north_short["unique_hole_pairs"] == 2
    assert np.isclose(north_short["maximum_hole_pair_share"], 0.5)


def _manual_empirical(model="exponential"):
    lag = np.linspace(5.0, 80.0, 12)
    truth = VariogramModel(
        model=model,
        range=40.0,
        sill=1.5,
        nugget=0.2,
        rmse=0.0,
        normalized_rmse=0.0,
        n_bins=len(lag),
    )
    gamma = semivariogram_values(lag, truth)
    return EmpiricalVariogram(
        kind="omnidirectional",
        estimator="classical",
        bin_edges=np.linspace(0.0, 85.0, len(lag) + 1),
        lag=lag,
        semivariance=gamma,
        raw_pairs=np.full(len(lag), 100),
        unique_hole_pairs=np.full(len(lag), 20),
        same_hole_pairs=np.zeros(len(lag), dtype=int),
        excluded_same_parent_pairs=0,
        hole_pair_contributions=tuple({"A|B": 100} for _ in lag),
    )


def test_exponential_fit_recovers_synthetic_parameters():
    fitted = fit_variogram(
        _manual_empirical("exponential"),
        "exponential",
        min_unique_hole_pairs=5,
    )
    assert fitted.success
    assert np.isclose(fitted.range, 40.0, rtol=0.03)
    assert np.isclose(fitted.sill, 1.5, rtol=0.03)
    assert np.isclose(fitted.nugget, 0.2, rtol=0.05)


def test_stability_gate_accepts_stable_and_rejects_range_instability():
    empirical = _manual_empirical()
    stable = [
        (
            empirical,
            VariogramModel(
                model="exponential" if i % 2 else "spherical",
                range=40.0 + i,
                sill=1.5,
                nugget=0.2,
                rmse=0.05,
                normalized_rmse=0.03,
                n_bins=12,
            ),
        )
        for i in range(4)
    ]
    thresholds = StabilityThresholds(
        min_successful_fits=4,
        min_supported_bins=4,
        min_unique_hole_pairs_per_bin=5,
        max_range_ratio=2.0,
        max_range_cv=0.3,
        max_normalized_rmse=0.2,
        max_nugget_fraction=0.5,
    )
    accepted = assess_variogram_stability(stable, thresholds)
    assert accepted.passed
    unstable = list(stable)
    unstable[-1] = (
        empirical,
        VariogramModel(
            model="exponential",
            range=250.0,
            sill=1.5,
            nugget=0.2,
            rmse=0.05,
            normalized_rmse=0.03,
            n_bins=12,
        ),
    )
    rejected = assess_variogram_stability(unstable, thresholds)
    assert not rejected.passed
    assert any("range" in reason for reason in rejected.reasons)


def test_synthetic_scenarios_are_deterministic_and_calibration_targets_work():
    assert SyntheticBenchmarkConfig().simulations_per_scenario == 500
    coordinates = np.column_stack(
        [np.linspace(0.0, 100.0, 20), np.zeros(20)]
    )
    domains = np.where(coordinates[:, 0] < 50.0, "A", "B")
    config = SyntheticBenchmarkConfig(
        simulations_per_scenario=12,
        random_seed=7,
        n_features=8,
    )
    first = generate_all_scenarios(
        coordinates,
        config=config,
        domains=domains,
    )
    second = generate_all_scenarios(
        coordinates,
        config=config,
        domains=domains,
    )
    assert set(first) == {"null", "stationary", "hard_boundary", "transitional"}
    assert first["null"].shape == (20, 12)
    assert np.allclose(first["stationary"], second["stationary"])

    calibration = calibrate_gate_threshold(
        np.linspace(0.0, 1.0, 100),
        {
            "stationary": np.linspace(1.5, 2.0, 100),
            "hard_boundary": np.linspace(1.6, 2.1, 100),
            "transitional": np.linspace(1.4, 1.9, 100),
        },
    )
    assert calibration.null_false_pass_rate <= 0.05
    assert min(calibration.positive_detection_rates.values()) >= 0.80
    assert calibration.meets_targets


def test_benchmark_gate_applies_one_shape_preserving_matrix_transform():
    coordinates = np.column_stack(
        [np.linspace(0.0, 10.0, 12), np.zeros(12)]
    )
    calls = []

    def centre_columns(matrix):
        calls.append(matrix.shape)
        return matrix - np.mean(matrix, axis=0, keepdims=True)

    report = benchmark_gate(
        coordinates,
        lambda values: float(abs(np.mean(values))),
        config=SyntheticBenchmarkConfig(
            simulations_per_scenario=4,
            random_seed=11,
            n_features=8,
        ),
        domains=np.where(coordinates[:, 0] < 5.0, "A", "B"),
        matrix_transform_function=centre_columns,
    )
    assert calls == [(12, 4)] * 4
    for scores in report.scores.values():
        assert np.all(scores < 1e-12)


def test_short_lag_score_balances_between_hole_pairs_deterministically():
    coordinates = np.column_stack(
        [np.linspace(0.0, 40.0, 15), np.zeros(15)]
    )
    holes = np.asarray(["A"] * 6 + ["B"] * 5 + ["C"] * 4)
    domains = np.asarray(["X"] * 11 + ["Y"] * 4)
    values = np.sin(coordinates[:, 0] / 6.0)
    kwargs = dict(
        domains=domains,
        holes=holes,
        between_hole_only=True,
        balance_hole_pairs=True,
        pair_count=5_000,
        random_state=17,
    )
    first = sampled_short_lag_signal_score(values, coordinates, **kwargs)
    second = sampled_short_lag_signal_score(values, coordinates, **kwargs)
    assert np.isfinite(first)
    assert first == second

    try:
        sampled_short_lag_signal_score(
            values,
            coordinates,
            between_hole_only=True,
            pair_count=100,
        )
    except ValueError as exc:
        assert "holes are required" in str(exc)
    else:
        raise AssertionError("missing holes should fail between-hole scoring")
