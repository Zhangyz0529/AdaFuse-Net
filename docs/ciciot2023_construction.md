# CICIoT2023 Stat39/Eth512 internal subset

```bash
python -m adafuse.data.ciciot2023_builder --pcap-dir data/raw/ciciot2023/PCAP --output-dir data/processed/ciciot2023 --workers 2
adafuse prepare --config configs/experiments/ciciot2023_internal.yaml --output results/ciciot2023
adafuse train --config configs/experiments/ciciot2023_internal.yaml --output results/ciciot2023
```

Use only BenignTraffic.pcap,DDoS-SYN_Flood.pcap,DictionaryBruteForce.pcap,Mirai-udpplain.pcap,Recon-PortScan.pcap. The copied deterministic builder forms non-overlapping100-packet windows, resetting at raw10000-packet blocks. Windows may span five-tuples; they do not span labels/capture files or raw blocks. It records read packets, exclusions, tails, time regressions, deduplication and conflicting labels. Records are paired from the same packets, never positionally joined to official CSV rows.

The39-field list and definitions are in the source/produced manifest. IAT uses adjacent capture-order timestamps; statistics describe packets/windows, not native flows. Protocol/port indicators are heuristics; HTTPS uses port associations and does not identify TLS versions or QUIC. IP/MAC/ports are anonymized before the captured Ethernet frames are concatenated/truncated/padded to512; DPKT may recompute supported checksums. This differs from the native IP512 plugin, which explicitly masks checksums and excludes Ethernet.

The constructor assigns complete raw blocks to train/validation/test before training; the portable internal preparation consumes those indices and fits only the training scaler. All labels occur in each role; this is not attack-class-held-out LOCO. Different blocks from a single capture may contain one session. No TLS1.3/QUIC composition or full officialCSV feature equivalence is claimed. The directory name in a local dataset download is not evidence of a different scientific dataset identity.
