# Local baseline implementations

Sixteen local neural architectures are included in baselines/registry.py. They are project approximations, not certified official author implementations. The registry preserves class definitions while stripping unsupported historical bibliographic/marketing comments. FC-Net uses relation-MSE during training; the portable common evaluation reports nearest-prototype embedding classification, so it must not be described as official FC-Net evaluation. Other local baselines use the retained positive-count-scaled SupCon objective, not AdaFuse's valid-anchor normalized SupCon.

Use configs/experiments/local_baseline.yaml, set variant to the desired registry label and use one shared dataset/episode/noise definition. Default embedding128,Adam5e-4,temperature0.07,50updates,weight_decay0. The four matched baselines are GP-VAE,MetaTraffic,MFF,Trafficformer-Light. Do not use HPO for these while claiming the fixed-baseline protocol. Their tuning budget differs from AdaFuse Optuna.

Alternative fusion presets use the shared dual encoders and projection. Concat/FiLM/Cross-Attention replace ADEF and modality gating, retain SCL+Dec, disable Polar. Cross-Attention operates over16 channel-group tokens of width16 with4 heads, not temporal packet tokens. Residual-MLP and Standard-DST are also implemented. Input dimension is configurable, but the fusion alternatives require embed_dim256.

No external pretraining weights are distributed. This release does not include or claim official training recipes for all named papers. A RandomForest can be evaluated with the shared plan by concatenating statistical and sequence inputs and fitting on each support set; the rf-evaluate command implements this reference separately from the neural trainer registry. Its seed controls forest fitting, not independent pretrained source encoders.

```bash
adafuse rf-evaluate --prepared results/matched/prepared --dataset-config configs/datasets/cic_ids2017.yaml --seed 3407 --shots 1 10 --output results/matched/RandomForest
```
