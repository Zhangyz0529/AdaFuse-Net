# AdaFuse-Net Reproducibility Package

A standalone, relocatable source release for multimodal few-shot traffic classification. Version: **0.1.0**. This directory is a local release candidate; preparing it does not publish a public repository or establish a public URL.

The package contains model/loss implementations, preprocessing procedures, explicit dataset configurations, model-independent episodic plans, fixed-length corruption operators,16 local neural baseline adapters plus a support-fitted RandomForest reference, fixed-budget training and validation-only Optuna, matched evaluation and conditional bootstrap, and a separate NF57/IP512 offline deployment pipeline.

**No datasets, processed samples, pretrained checkpoints, original sample-ID plans, paper result tables, figures, experiment logs, manuscripts or review correspondence are distributed.** Runtime outputs are private and ignored. Generating outputs locally is supported and necessary for independent verification.

## Install

Tested reference: Python3.9.25, PyTorch2.1.2, NumPy1.26.4. See pinned requirements and environment.yml. CPU/GPU PyTorch wheels must match the user's platform; the reference environment used a CUDA12.1 wheel. NFStream is optional for core training.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
# Only for NFStream extraction/deployment:
python -m pip install -r requirements-deployment.txt
python -m unittest discover -s tests -v
```

Without installation, use `PYTHONPATH=src python -m adafuse.cli.main --help` from this directory. Test fixtures are generated in temporary directories; they are not scientific benchmark results.

## Reproduce the core workflow

1. Download data from the original providers: [data/README.md](data/README.md).
2. Build trusted multimodal records using the explicit dataset-specific procedures: [docs/data_preparation.md](docs/data_preparation.md). Statistical dimensions are never padded to pretend cross-domain compatibility.
3. Inspect the dataset YAML and output path. All paths are relative to the current working directory, or supplied as CLI arguments.
4. Generate the common split/scaler/episode plans before training:

```bash
adafuse prepare --config configs/default.yaml
adafuse train --config configs/default.yaml
adafuse evaluate --checkpoint results/cic_ids2017/checkpoints/round0/shot1/model.pt --prepared results/cic_ids2017/prepared --output results/cic_ids2017/clean
adafuse evaluate --checkpoint results/cic_ids2017/checkpoints/round0/shot1/model.pt --prepared results/cic_ids2017/prepared --operator IBP --rho 0.3 --output results/cic_ids2017/ibp30
```

The training entry runs every configured round and shot. Replace `--dataset-config` with one of the five dataset YAMLs and select a fresh `--output`; use `--seed 42`, `3407` or `2026` for independent training repeats. Common episode_seed and noise_seed do not become training replicates.

For validation-only hyperparameter search, prepare first and run:

```bash
adafuse search --config configs/default.yaml --trials 200
```

Selection averages held-out validation tasks and configured shots. Test scoring happens only after the selected trial is written. See docs/reproduction.md for matrix runs, ablation switches, alternative fusion, noisy training controls and result aggregation. Fixed-configuration training and Optuna results are separate protocols, not interchangeable estimates.

## Deployment and recent IoT subset

- [NF57/IP512 deployment](docs/deployment_pipeline.md): same native fields, source-only scaler, frozen inference and stage timing. This is a separately trained57D variant, not a77D checkpoint adapter.
- [CICIoT2023 construction](docs/ciciot2023_construction.md): Stat39/Eth512,100-packet windows and raw10000-packet block partitions. It is not a native-flow dataset or a TLS1.3/QUIC composition validation.
- [Frozen-source comparison](docs/evaluation_protocols.md): one frozen source model, target support prototypes, same target test scores under fixed/source/target thresholds.

## Layout

`configs/` settings; `src/adafuse/` implementation; `scripts/` executable workflows; `tests/` minimal correctness and end-to-end checks; `docs/` protocols and limitations; `data/` placement instructions and empty directories; `results/` and `checkpoints/` empty runtime roots.

## Release checks

```bash
python scripts/audit_release.py
python -m unittest discover -s tests -v
```

The source manifest hashes code/configuration only. Historical numeric results are intentionally absent. A clean package is evidence that code is available locally, not evidence that every historical number has been independently regenerated. Known gaps are stated in [docs/known_limitations.md](docs/known_limitations.md). License: MIT for project code; see THIRD_PARTY.md and dataset terms separately.
