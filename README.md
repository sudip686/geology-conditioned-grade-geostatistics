# Geology-conditioned grade geostatistics

This versioned snapshot contains reusable statistical code, selected tests, a
minimal public configuration, and a complete synthetic end-to-end workflow for
support-aware geology-conditioned grade geostatistics.

The release is intentionally data-minimal. It contains no empirical raw or
interval-level data, coordinates, sample or drillhole identifiers, restricted
reports, manuscript/document/table/figure-generation code, prediction rows, or
fold records. Data-owner-authorized, non-identifying aggregate evidence is
included under `results/` so the reported numerical decisions can be audited.
`results/EVIDENCE_AVAILABILITY.json` records the release boundary. Full
empirical reproduction still requires owner-approved inputs and private
pipeline components that are not distributed.

The decision rule retains supported, unsupported, mixed, and insufficient-
evidence/abstention outcomes. Geological conditioning is evaluated as a
hypothesis; this package does not establish strong local prediction performance
and is not a resource-estimation workflow.

## Install and test

Python 3.10 or newer is required. A fully pinned environment specification is
provided in `requirements-lock.txt`; a clean locked installation remains a final release check.

```console
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
pytest
```

Run the complete synthetic workflow with:

```console
python examples/synthetic_end_to_end.py --output-dir synthetic_outputs
```

The demonstration creates synthetic parent intervals and common-support
composites, checks support and grade-mass conservation, assigns parent-safe
grouped spatial folds with grade-blind buffers, compares a global-mean and a
geological-domain model, audits equal hole-pair weighting, and reports a
synthetic structural gate. Nothing it generates is an empirical project result.

## Repository layout

- `src/nrr_study/`: reusable analysis and validation modules.
- `tests/`: synthetic and in-memory tests.
- `examples/synthetic_end_to_end.py`: deterministic synthetic workflow.
- `config/public_study_config.json`: public methodological settings.
- `results/EVIDENCE_AVAILABILITY.json`: public-evidence governance status.
- `results/AGGREGATE_EVIDENCE_INDEX.csv`: mapping from paper items to released aggregate files.
- Other files under `results/`: approved non-identifying aggregate evidence.
- `results/SYNTHETIC_OUTPUT_SCHEMA.json`: synthetic output data dictionary.
- `RELEASE_METADATA.json`: version, upstream state, and publication status.
- `INTERNAL_CONFIDENTIALITY_SCAN.json`: machine-readable pre-release scan.
- `MANIFEST.sha256`: SHA-256 checksums for every other release file.

## Rights, status, and citation

The MIT License applies only to software under `src/`, tests, and examples. It
does not grant rights in data, aggregate evidence, reports, configuration,
manuscript, table, or figure content. See `LICENSE` and `NOTICE.md`.

Version 0.3.1 is prepared as the GitHub tag v0.3.1. No archive DOI has been assigned. RELEASE_METADATA.json records the tag and the absence of a persistent identifier; CITATION.cff supplies author, version, and repository metadata.
