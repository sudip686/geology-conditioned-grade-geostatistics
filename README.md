# Geology-conditioned grade geostatistics

This repository snapshot contains reusable statistical code, a synthetic
benchmark, selected tests, and a minimal public configuration for a
support-aware geology-conditioned grade-geostatistics study.

The release is intentionally data-minimal. It contains no raw or interval-level
data, coordinates, sample or drillhole identifiers, author metadata, reports,
manuscript, document, table, or figure-generation code, prediction rows, fold
records, derived study outputs, or diagnostic-kriging outputs. Reproducing the
restricted empirical analysis requires owner-approved inputs that are not
distributed here.

The scientific decision rule retains supported, unsupported, mixed, and
insufficient-evidence/abstention outcomes. Geological conditioning is evaluated
as an interpretive and predictive hypothesis; this package does not establish
strong local grade-prediction performance and is not a resource-estimation
workflow.

## Install and test

Python 3.10 or newer is required.

```console
python -m pip install -e ".[test]"
pytest
```

Run the synthetic-only example with:

```console
python examples/synthetic_gate_demo.py
```

## Repository layout

- `src/nrr_study/`: reusable analysis, validation, geostatistical, geological,
  sparse-evidence, and synthetic-benchmark modules.
- `tests/`: tests that use synthetic or in-memory toy data only.
- `examples/`: a deterministic synthetic gate demonstration.
- `config/public_study_config.json`: methodological settings with all input
  locations, identifiers, CRS metadata, and unapproved diagnostic settings
  omitted.
- `INTERNAL_CONFIDENTIALITY_SCAN.json`: the machine-readable pre-release scan.
- `MANIFEST.sha256`: SHA-256 checksums for every other release file.

## Rights and citation

No open-source license is granted. See `NOTICE.md` before use and
`CITATION.cff` for citation metadata.
