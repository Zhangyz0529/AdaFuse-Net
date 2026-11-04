"""Generate one shared plan, then fixed-budget local-model matched evaluations."""
import argparse,copy,json
from pathlib import Path
from adafuse.utils.common import load_config,save
from adafuse.episodes.plans import prepare
from adafuse.training.trainer import train
from adafuse.evaluation.episodic import evaluate
from adafuse.baselines.registry import REGISTRY
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--dataset-config',default='configs/datasets/cic_ids2017.yaml');p.add_argument('--seed',type=int,default=3407);p.add_argument('--output',default='results/matched');p.add_argument('--device',default='cpu');a=p.parse_args()
cfg=load_config(a.config);cfg.update(training_seed=a.seed,output=a.output,shots=[1],device=a.device,evaluation_plan='matched');ds=load_config(a.dataset_config);prepared=prepare(ds,cfg);rounds=json.loads((prepared/'manifest.json').read_text())['rounds']
for model in ['AdaFuse-Net','GP-VAE','MetaTraffic','MFF','Trafficformer-Light']:
    c=copy.deepcopy(cfg);c.update(variant=model,output=str(Path(a.output)/model))
    if model in REGISTRY:c.update(embed_dim=128,lr=5e-4,weight_decay=0.,temperature=.07,seed_per_update=True)
    for rd in rounds:
        ck=train(ds,c,prepared,rd['round'],1)
        for op,rho in [('Clean',0.),('IBP',.3)]:evaluate(ck,prepared,operator=op,rho=rho,output=Path(c['output'])/'evaluation'/f"round{rd['round']}"/op)
