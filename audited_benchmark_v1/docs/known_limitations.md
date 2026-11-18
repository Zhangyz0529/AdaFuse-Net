# Known limitations

- This is a local source-release candidate, not an already published URL or a claim that independent researchers have reproduced results.
- No raw/processed data, trained weights, historical ID manifests, outputs or figures are distributed.
- Historical main77D preprocessing is not certified equivalent to NF57 extraction. Dimension agreement is insufficient. Some historical raw-to-PKL details, especially DoH29, remain unverified; explicit schema import is supported rather than invented reconstruction.
- The fixed-budget public main protocol and baseline objectives are explicit. They cannot retroactively resolve historical train/test contamination, missing seed records or unmatched episodes.
- Canonical-group splitting is stronger than sample splitting but is not proof of independent sessions/captures. IoT packet-block splitting is not flow/session isolation.
- ADEF expert mixing weights are unconstrained trainable global scalars initialized1; they are not input-dependent reliability estimates. SE coefficients,DST thresholds and independent sigmoid modality gates are input-dependent. Gates are not normalized shares.
- No open-set rejection or protocol-valid packet-level adversarial modification is claimed. Byte overwrites are synthetic input-space operators. The source pipeline does not prove robustness to semantic TLS/QUIC attacks.
- Native frozen transfer must reuse sourceScaler and feature contracts. Capture labels are not independently certified flow labels. Source calibration independence must be documented; a validation set used for checkpoint selection is not an independent calibration pool.
- No historical result table, figure or performance conclusion is embedded in this package. Generated test fixtures validate functionality only.
- The16 baseline definitions are local approximations. See THIRD_PARTY.md and baselines.md; model-label equality is not proof of official reproduction.
