# Audited benchmark v1

## Inputs and scope

The 14 outer folds use held-out target classes and group-separated shared reference records. Reports certify only the declared observed-session/association grouping; capture and host independence are not claimed. See audit/ and configs/. Historical flags in frozen reports refer to their original preparation stage, not the later experiment status.

Actual source implementations are in source_snapshot/. Some archived runners retain original server paths and are not promised as portable training entrypoints. The new evaluation and table scripts below take explicit artifact paths. All baseline training paths have not yet been independently relocated and tested. Third-party code retains its own licenses.

## Extract Release assets

Extract all required packages into one `artifacts/` directory. They provide `partitions/`, `preprocessing/`, `cache/`, `runs/`, and `source_records/`. For a .tar.gz split into .part001 etc, concatenate parts in numeric order before extracting. Verify SHA256SUMS before extraction.

## Install

Use Python3.9 and the reference requirements/environment from the repository root. Install a PyTorch wheel appropriate for the local machine.

## Recompute saved-result tables

```bash
python benchmarks/r1_5/scripts/regenerate_tables.py --artifacts artifacts
```

The results package includes completed-run metrics/configs/plans for all reported models. Recalculation is not checkpoint evaluation.

## Evaluate an AdaFuse-Net checkpoint

```bash
python benchmarks/r1_5/scripts/evaluate_adafuse.py --artifacts artifacts --dataset USTC --fold round0 --seed 42 --shot 1 --device cpu --output evaluation/USTC
```

Default evaluation uses all 100 episodes for each Val/Test and Clean/IBP30 condition. `--episodes 2` performs a short check, not complete benchmark verification. Original source paths in effective_config.json are provenance only; this evaluator does not read them. See verification/ for the actually tested scope.

## Limitations

Pre-Contrast-Local is not the authors' official implementation. MFF/GP-VAE/MetaTraffic/FlowPic/Trafficformer-Light adaptations are described in source_snapshot/native/. Native FlowPic/MetaTraffic inputs have representation aliases and no byte-replacement robustness evaluation. RF seeds are estimator repeats. No external preregistration or new 50-trial Optuna search is claimed. Full training relocation, per-class group-count summaries and exclusion reconciliation still need explicit verification; source-level records are provided to inspect these.

## Cross-platform asset extraction

```bash
python benchmarks/r1_5/scripts/extract_assets.py --assets release_assets --output artifacts
```

This checks all listed hashes, joins split archives and extracts into a shared artifact root. Allow sufficient disk space for the uncompressed dataset and temporary joined archive.
