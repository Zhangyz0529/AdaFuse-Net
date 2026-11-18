# Dataset acquisition and placement

Obtain data from the original providers and comply with their current terms/citation instructions. No mirrors or dataset files are distributed here.

| Configuration | Original provider | Expected local location |
|---|---|---|
| CIC-IDS-2017 | https://www.unb.ca/cic/datasets/ids-2017.html | data/raw/cic_ids2017 |
| CIC-DDoS-2019 | https://www.unb.ca/cic/datasets/ddos-2019.html | data/raw/cic_ddos2019 |
| USTC-TFC2016 | https://github.com/yungshenglu/USTC-TFC2016 | data/raw/ustc_tfc2016 |
| ISCX-VPN | https://www.unb.ca/cic/datasets/vpn.html | data/raw/iscx_vpn |
| CIRA-CIC-DoHBrw2020 | https://www.unb.ca/cic/datasets/dohbrw-2020.html | data/raw/cic_doh |
| CICIoT2023 subset | https://www.unb.ca/cic/datasets/iotdataset-2023.html | data/raw/ciciot2023/PCAP |

The USTC repository is a distribution reference; consult the dataset's citation and redistribution terms. Each processed record contains stat_raw,seq_feature,label; arrays must have the configured statistical dimension and512 sequence positions. Trusted historical PKLs can be converted with import-legacy. Never execute/import an untrusted pickle. Official feature CSVs are precomputed statistics, not feature-extraction timing inputs.

See docs/data_preparation.md for schema and matching constraints. No arbitrary filling/truncation of statistical dimensions is permitted; length512 byte construction is an explicitly different operation.
