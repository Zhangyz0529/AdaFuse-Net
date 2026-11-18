# AdaFuse-Net

PyTorch implementation and audited benchmark materials.

## Audited benchmark

See [benchmarks/r1_5/README.md](benchmarks/r1_5/README.md). The audited reevaluation covers USTC, ISCX and CIC-IDS with four support sizes and five training seeds. It is a new evaluation on reconstructed partitions, not recovery of historical splits or exact historical scores. Check the completeness table before comparing methods.

Repository files include source code, effective configurations, compact audit records, run metrics and artifact indexes. Processed inputs, full manifests and AdaFuse-Net checkpoints are separate Release assets. Original PCAP and baseline pretrained weights are not included. Pre-Contrast-Local is the original project implementation; baseline adaptations and limitations are documented.

The original source-only workflow is retained at [docs/legacy_source_release_README.md](docs/legacy_source_release_README.md). Its old validation and distribution scope do not describe the new artifacts.
