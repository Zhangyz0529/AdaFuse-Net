# Execution recipes and output boundaries

Start at README.md. Every command is intended to run from the repository root, with data paths controlled by YAML/CLI. Use fresh output directories to prevent accidental checkpoint reuse. Output files are generated locally and excluded from the release.

## Five-dataset matrices

Run prepare/train for each dataset YAML and each training seed. Use common episode_seed42/noise_seed20261028,shots1/3/5/10,100 evaluation episodes/task. The preserved loader fixes split_seed42 independently of training_seed. prepare persists scaler/split/episode data before training. The portable main protocol is the verified class-rotation protocol, not an assertion that every historical baseline used it. Historical unaligned results cannot be turned into paired results by matching numeric episode IDs.

## Module ablation

Copy the complete default configuration and toggle use_adef,use_gating,use_dec to enumerate eight variants. This reproduces the factorial computation switches, with inactive modules frozen. Do not confuse the resulting module ablation with the legacy nine-row SE/DST variants or with differently normalized historical regularizers.

## Alternative fusion and denoising controls

Use alternative_fusion.yaml for AdaFuse-Net,Concatenation,FiLM,Cross-Attention; update variant and choose a fresh output for each seed/variant. Train all on Clean inputs,50updates,shots1/10,then evaluate Clean/INP30/IBP30 under common IDs. Rec0/rec0.1 use denoising_rec0.yaml and denoising_rec01.yaml. Both keep the training reconstruction head and identical alternating Query corruption; only reconstruction_weight changes. Decoder logits reconstruct clean integer tokens from h_seq_raw with257-class crossentropy at selected corruption positions, not clean features. Inference never executes the reconstruction head. Checkpoints retain full training state; to export an encoder-only artifact remove reconstruction_head.* and load into the matching Concat model.

## Result aggregation

Per-episode records contain task/shot/training_seed/condition IDs and input hashes. Average episodes within each task,then equally across tasks,then across independent training seeds with ddof1. Keep FP prediction-occurrence sums separate from per-episode FP means. AttackRecall uses all attack queries; attack ASR denominators, if computed separately, use clean-correct attack queries. Bootstrap intervals are conditional, not training-seed inference.

## Exactness boundary

The public code preserves audited model/loss equations and retained reference sampling/noise mechanics. Path removal and module packaging do not authenticate a missing historical input file. Raw source record order, field semantics, labels and split provenance matter. Without the historical processed inputs and their verified extraction, bitwise reproduction of every published value is not established. No numeric table assertions or result fixtures are used in tests.
