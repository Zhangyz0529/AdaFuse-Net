# Evaluation and statistical reporting

## Frozen thresholds

calibrate takes only label/score calibration rows. It selects the minimum-FPR threshold subject to calibration Attack Recall>=0.95, choosing the highest threshold on ties. The test set never selects a threshold. No-positive precision is undefined scientifically; numerical summaries use zero_division=0 and retain an explicit flag. MacroF1 and attack-positive F1 are distinct metrics.

```bash
adafuse calibrate --scores data/processed/source_calibration.csv --test-scores data/processed/source_test.csv --output results/threshold.json
```

## Cross-domain native pilot

Only use a source model explicitly trained on NF57/IP512. Both source and target must be extracted using the same frontend,57 field order and masking contract; target inputs use the source checkpoint scaler. Canonical target groups are hash-split into support/calibration/test roles. Select target prototypes from labeled support only; no gradient updates occur. Target calibration is a labeled reference with the same frozen model, not an upper bound. The three thresholds apply to identical target scores per support draw.

```bash
adafuse frozen-pilot --checkpoint results/nf_source/checkpoints/round0/shot10/model.pt --source-calibration data/processed/source_calibration.csv --target data/processed/nf_target/dataset.pkl --normal-label BENIGN --output results/frozen_pilot
```

The default is100 support draws,1500 calibration and1500 test queries per class,10-shot. The module requires group metadata and input-contract sidecars. Its output documents protocol mechanics; independently certify label provenance and calibration independence before drawing scientific conclusions. A historical77D target diagnostic is not automatically a verified native-input transfer experiment.

## Matched bootstrap

Use per-episode records with full(dataset,training_seed,shot,task,episode,condition) keys, common group identifiers and matching input_sha256. Reject duplicates, missing pairs or input mismatches before merging. Run separately per dataset/training_seed/shot. Within each task, resample episodes with replacement using common indices for both models and Clean/IBP30. Compute task means, then equal-task means. Recompute Delta_Drop=Delta_IBP30-Delta_Clean inside each replicate. Default10000 replicates,2.5/97.5 percentile endpoints. These intervals describe evaluation-sampling uncertainty conditional on fixed checkpoints and pools, not independent training variability or session-level population uncertainty. The function is pairwise per shot; callers comparing several baselines/shots jointly must explicitly share indices if claiming a joint resampling experiment.

```bash
adafuse bootstrap --ours results/ours_episode_metrics.csv --baseline results/baseline_episode_metrics.csv --output results/paired_interval.json
```

## Prevalence

The prevalence sweep fixes15 attack queries and score/model/prototype conditions while nesting BENIGN queries from a fixed pool at beta1/10/50/100. It reports FP absolute counts and N_BENIGN, not just FPR. FPR is a conditional rate; observed finite-sample values can vary. Counts of repeated episode prediction occurrences are not independent deployment-flow counts.

```bash
adafuse prevalence --scores data/processed/fixed_query_scores.csv --threshold 0.5 --output results/prevalence.csv
```

For native binary source calibration, evaluate the validation role using its fixed reserved support, then export distinct query scores. Do not pool scores from changing prototypes into one calibration distribution:

```bash
adafuse evaluate --checkpoint results/nf_source/checkpoints/round0/shot10/model.pt --prepared results/nf_source/prepared --role validation --output results/nf_source/calibration
python scripts/export_scores.py --predictions results/nf_source/calibration/predictions.jsonl.gz --output data/processed/source_calibration.csv
```

This exports only the sampled validation queries, not every available validation record. The subset policy and whether it was used for selection must be stated. Reusing a selection validation subset does not create independent calibration.
