#!/usr/bin/env bash
set -euo pipefail
mode="${1:-fixed}"
for dataset in cic_ids2017 cic_ddos2019 ustc_tfc2016 iscx_vpn cic_doh; do
  for seed in 42 3407 2026; do
    output="results/main/${dataset}/seed${seed}"
    python -m adafuse.cli.main prepare --dataset-config "configs/datasets/${dataset}.yaml" --seed "$seed" --output "$output"
    if [[ "$mode" == "search" ]]; then
      python -m adafuse.cli.main search --dataset-config "configs/datasets/${dataset}.yaml" --seed "$seed" --output "$output" --trials "${OPTUNA_TRIALS:-200}"
    else
      python -m adafuse.cli.main train --dataset-config "configs/datasets/${dataset}.yaml" --seed "$seed" --output "$output"
    fi
  done
done
