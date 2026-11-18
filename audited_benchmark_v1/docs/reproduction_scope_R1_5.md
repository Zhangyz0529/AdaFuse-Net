# R1-5 reproduction scope — 2026-11-17

New audit scripts and factual inspection reports have been added locally. This does NOT establish a leakage-audited reproduction of the principal comparisons. The historical release ZIP is unchanged; no external upload has been performed.

Completed reports identify registered 512-byte PKLs, schema, labels and exact input duplicates. CSV tuple reuse is distinct from session identity and from actual cross-partition leakage. Loader reconstructions explicitly identify the snapshot, seed and protocol; they are not recovered historical manifests. Feature-only CSV links do not verify sequence provenance.

Still missing: authenticated input-to-source provenance, justified session/group definitions, feasible class-preserving partitions, three-way group/content/time audits, frozen episode pools, fresh principal model and baseline runs, and a full table-to-artifact chain. Training must not proceed using a PASS inferred from these inventory checks.

Validation may be used for declared model/threshold selection. Test must be excluded from these selection steps. Prior hyperparameter exposure must be disclosed.

The original reproduction package and results remain historical evidence. New results must be kept separate. Do not claim these new scripts reproduce historical table values or verify capture independence.
