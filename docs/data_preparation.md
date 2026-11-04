# Data preparation

## Contracts

The five historical model inputs have dimensions77(CIC-IDS/CIC-DDoS),25(USTC/ISCX),29(CIC-DoH). Their dimension equality is not proof of semantic equivalence. The independent native deployment contract is NF57/IP512; the recent IoT internal contract is Stat39/Eth512. Field order is a model input contract, not a set.

### Trusted existing records

```bash
adafuse import-legacy --input data/raw/cic_ids2017/trusted_dataset.pkl --stat-dim 77 --output data/processed/cic_ids2017/dataset.pkl
```

This validates shape/finiteness and converts historical seq tokens without fabricating missing statistical fields. It does not recover an undocumented extraction history. DoH records with metadata embedded in stat_raw require an explicit schema conversion; the importer rejects them rather than silently slicing columns.

### CIC-IDS CSV plus PCAP

The included77-field list is the historical official TrafficLabelling column order after its seven identifying fields and before Label, with pandas duplicate-column suffixes retained. CSV must include the five-tuple, Timestamp, Flow Duration, Label. Match using canonical bidirectional tuples and the CSV time interval. Configure timestamp format and UTC offset explicitly; duration units in this adapter are microseconds. Ambiguous matches are excluded and counted, nonfinite CSV statistics are excluded and counted, and no padding of missing statistical columns occurs.

```bash
adafuse csv-pcap --csv data/raw/cic_ids2017/labelled_flows.csv --pcap data/raw/cic_ids2017/capture.pcap --features configs/datasets/cic_ids2017_features.json --output data/processed/cic_ids2017/dataset.pkl
```

This strict adapter is a documented portable procedure, NOT a certification of bitwise equivalence to an undocumented original dataset. Minute-resolution timestamps, timezone uncertainty, flow timeout differences and tuple reuse can prevent reliable matching. The sidecar records hashes, exclusions and matching parameters. MachineLearningCSV without identifying metadata is insufficient for this join. For CIC-DDoS, supply its actual ordered feature schema; do not reuse a77D header merely because the dimensions match. For DoH29, explicitly identify the29 training fields and metadata/time units before using a CSV adapter. A complete original raw-to-PKL generator for the historical DoH dataset is not certified by this release.

### USTC / ISCX / historical DDoS helpers

Portable versions of the retained raw extractors are in scripts/preprocessing. They preserve their original statistical calculations and byte construction but replace author-specific paths with arguments. These legacy helpers aggregate five-tuples within a capture and are not certified CICFlowMeter replacements or session-independent splitters.

```bash
python scripts/preprocessing/ustc.py --raw-dir data/raw/ustc_tfc2016 --output-dir data/processed/ustc_tfc2016
python scripts/preprocessing/iscx.py --raw-dir data/raw/iscx_vpn --output-dir data/processed/iscx_vpn
python scripts/preprocessing/cicddos.py --raw-dir data/raw/cic_ddos2019 --output-dir data/processed/cic_ddos2019
```

Check the generated filename against the input YAML. A mismatch is a configuration error, not an invitation to substitute incompatible features. The new strict matching adapter and the retained historical helper are distinct procedures.

## Preprocessing and splits

Source-only mean/std are fitted after the train partition is determined and saved with each round checkpoint. No target-only scaler fitting is allowed for frozen-source experiments. Sample-index splitting cannot certify session or capture independence. See protocol.md.

Ethernet construction anonymizes MAC/IP/transport identifiers before truncation; NF57 uses actual IP-layer bytes and masks checksums rather than reconstructing Ethernet. TLS record lengths and transport timestamps are not automatically zeroed by either implementation. No undocumented anti-shortcut prefix mask is silently applied.

`data/anti_shortcut.py` exposes an explicit optional experimental prefix-mask utility, separate from semantic header anonymization. The matched core entries intentionally do not apply the historical extra4-stat/54-sequence mask. If reconstructing a legacy masked protocol, apply it at the documented stage before Query corruption, record it in metadata, and do not merge those outputs with unmasked results. Header anonymization does not imply that every later byte replacement preserves masked values.
