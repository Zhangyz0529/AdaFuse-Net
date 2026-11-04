# Dependencies and local baseline status

This distribution contains project implementation code, not downloaded author repositories. PyTorch, NumPy, SciPy, pandas, scikit-learn, Optuna, NFStream, DPKT, Scapy, PyYAML and tqdm are installed separately and retain their respective licenses. Datasets and weights are not redistributed and are not covered by this software license.

`src/adafuse/baselines` contains the project's local approximations/adapters under method labels, not certified reproductions or implementations endorsed by the cited method authors. In particular, the local NetMamba uses a pure-PyTorch state-space block, Pre-Contrast has no external pretraining stage, and GP-VAE uses the local encoder adapter. These distinctions must be retained in publications. Do not rename these files as official implementations. Their architecture statements are preserved; unsupported bibliographic claims in historical comments have been removed.

Historical baseline training objectives are not identical to the AdaFuse objective: local SupCon uses positive-count scaling; FC-Net has a relation-MSE training branch. No pretrained weights are included. See docs/baselines.md for scope.
