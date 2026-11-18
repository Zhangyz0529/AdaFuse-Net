# Separate NF57/IP512 offline deployment

Install requirements-deployment.txt. The shared NFStream frontend uses statistical_analysis=True,n_meters=1,idle_timeout=120s,active_timeout=1800s,accounting_mode=3,decode_tunnels=False,snapshot_length=1536,n_dissections=0. Native fields are listed in configs/datasets/nf57_source.yaml. Time fields are milliseconds, counts are packets/bytes at the configured accounting layer. These definitions differ from historical77D CSV features.

The plugin reads packet.ip_packet, excludes Ethernet, zeros IPv4/IPv6 addresses and available TCP/UDP ports/checksums, concatenates packet bytes in capture order, takes the first512 and right-zero-pads. Certified extraction is restricted to IPv4 TCP/UDP non-VLAN flows. Source and target must share this code/config; source standardization parameters are frozen.

```bash
adafuse nf-extract --pcap data/raw/nf_source/benign.pcap --label BENIGN --output data/processed/nf_source/benign.pkl
adafuse nf-extract --pcap data/raw/nf_source/attack.pcap --label ATTACK --output data/processed/nf_source/attack.pkl
adafuse merge-records --inputs data/processed/nf_source/benign.pkl data/processed/nf_source/attack.pkl --output data/processed/nf_source/dataset.pkl
adafuse prepare-binary --config configs/experiments/deployment_nf57_ip512.yaml --output results/nf_source
adafuse train --config configs/experiments/deployment_nf57_ip512.yaml --output results/nf_source
adafuse export-support --prepared results/nf_source/prepared --shot 10 --output data/processed/nf_source/support.npz
adafuse deploy --checkpoint results/nf_source/checkpoints/round0/shot10/model.pt --support data/processed/nf_source/support.npz --pcap data/raw/nf_source/held_out.pcap --output results/nf_source/pcap_run
```

The inference PCAP must be independently held out; export-support is a convenience, not a certification that an arbitrary PCAP is disjoint from it. Support statistics in the exported file are already source-scaled. Target live inputs receive the source mean/scale once.

Timing begins before NFStreamer construction and ends after all predictions. Model/support loading and prototype creation are outside this boundary. Every returned flow records iterator/input-assembly time, preprocessing/tensor/device time, T_encoder,T_decision,T_model. GPU timings synchronize before/after relevant stages. CPU execution uses no device transfer. The iterator stage can include IPC/prefetch waits and is NOT the isolated meter-worker CPU time. It is not labeled the historical meter-worker+assembly composite. Global wall time is independent; throughput=Ncompleted/walltime,not inverse average latency. No PCAP capture timestamp is subtracted from a monotonic clock. Flow-lifetime online latency is not measured.

This portable serial timing entry does not recreate historical worker-instrumented stage numbers. Raw PCAP parsing, flow assembly and statistics are performed together inside NFStream; they cannot be truthfully split into three independent costs without deeper instrumentation. Never add an unrelated NFStream benchmark to model-only timing and call it measured end-to-end time.

## Mixed Thursday source capture

Do not assign ATTACK to an entire mixed Thursday capture. `nf-extract-labelled` implements the retained weak tuple/time-overlap rule against official identifying CSVs and excludes conflicting/unmatched flows. It is not exact native-flow annotation. Timestamp/UTC assumptions are explicit arguments, not inferred from a filename. The recorded historical Thursday procedure used an explicit UTC offset and a12-hour afternoon adjustment for ambiguous minute timestamps; these must be justified for the user's files.

```bash
adafuse nf-extract-labelled --pcap data/raw/cic_ids2017/Thursday_slice.pcap --csv-dir data/raw/cic_ids2017/TrafficLabelling --utc-offset-seconds 10800 --afternoon-adjust --output data/processed/nf57_source/dataset.pkl
```

Use this output with the native source configuration. Capture-label extraction is appropriate only for files whose single-class capture labels are justified. Exporting/calibrating a threshold does not fix weak label correspondence.
