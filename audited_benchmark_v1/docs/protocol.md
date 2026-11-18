# Episodic protocol

## Main five datasets

The portable class-rotation loader preserves the retained data-manager computation with explicit YAML settings in place of global configuration. The reference class is BENIGN/Benign, except the ISCX experiment's configured application reference; do not describe all ISCX classes as attacks. Target and validation attack classes are excluded from source training. Reference samples are split60/20/20 at fixed split_seed42. Each round fits its scaler using source meta-train only. This is sample-ID isolation, not certified session isolation.

Training plans use a training-seed-dependent generator,50 updates, up to5 eligible classes,10 preselected support candidates and15 queries per class; requested shot uses the support prefix. Classes with fewer than25 available samples cannot enter training episodes. Query/Support within each training episode are disjoint.

Held-out validation/test role pools are divided into disjoint Support/Query candidates. Validation plans follow the original training-seed-specific draws, so HPO does not share validation selection with test. With evaluation_plan=source, test plans also follow the corresponding training seed. With evaluation_plan=matched (default for portable comparisons), reference seed3407 draws reconstruct candidate unions independent of model outcomes; the shared evaluation seed generates100 episodes from those unions before any model is trained. These candidates are not the complete test partition. Across models and training seeds, reuse the exact same processed source order, dataset name, episode_seed and noise_seed, and verify saved IDs/input hashes. Changing the input record order changes the plan: seed equality alone is not pairing proof.

Test episodes are two-way, every class has15 queries and K support samples,K in1/3/5/10. Source model weights and buffers do not update during evaluation. Each Clean/corrupt pair uses the same clean Support prototypes. Embeddings are L2-normalized; prototypes are class means without a second normalization; the main matched/fusion classifier uses squared Euclidean distances and softmax temperature1. Closest-prototype ties go to the reference class. Threshold calibration uses attack_score>=threshold, with ties explicitly attack-positive.

Metrics: within-task episode mean, then task-equal mean, then training-seed mean/sampleSD(ddof1). Do not use episode SD as training-seed SD. Noise seeds and optimization restarts are different repeat units. A single fixed seed has no training-seed SD.

## Source-native binary deployment

prepare-binary is a separate explicit5-fold canonical-group split: fold0 test,fold1 validation,folds2-4 source training. Source support is reserved from training-role samples and excluded from gradient updates. The final fixed-update checkpoint is used by default. It is a portable re-training protocol, not a claim to recreate the original validation-selected single-capture deployment pilot.

## Recent IoT internal

Use the raw-packet-block split saved by the39D constructor. All five labels occur across train/validation/test; this is within-domain classification, NOT LOCO. Adjacent packet blocks can belong to one network session. Native-flow frozen transfer uses a different input construction and is not interchangeable.
