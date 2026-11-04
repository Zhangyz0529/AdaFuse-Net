"""Portable public CLI: paths are explicit and outputs are never distributed."""
import argparse,json,copy
from pathlib import Path
import torch
from ..utils.common import load_config,save

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','prepare-binary','train','search']:
        a=sub.add_parser(name);a.add_argument('--config',default='configs/default.yaml');a.add_argument('--dataset-config');a.add_argument('--output');a.add_argument('--device');a.add_argument('--seed',type=int);a.add_argument('--trials',type=int,default=200)
    for name in ['evaluate','robustness']:
        a=sub.add_parser(name);a.add_argument('--checkpoint',required=True);a.add_argument('--prepared',required=True);a.add_argument('--output',required=True);a.add_argument('--operator',default='Clean',choices=['Clean','INP','IBP','RTO','CTO','Gaussian','Joint']);a.add_argument('--rho',type=float,default=0.3);a.add_argument('--role',choices=['validation','test'],default='test')
    a=sub.add_parser('import-legacy');a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--stat-dim',type=int,required=True)
    a=sub.add_parser('csv-pcap');a.add_argument('--csv',required=True);a.add_argument('--pcap',required=True);a.add_argument('--features',required=True);a.add_argument('--output',required=True);a.add_argument('--time-format',default='%d/%m/%Y %H:%M:%S');a.add_argument('--utc-offset-seconds',type=int,default=0)
    a=sub.add_parser('nf-extract');a.add_argument('--pcap',required=True);a.add_argument('--label',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('nf-extract-labelled');a.add_argument('--pcap',required=True);a.add_argument('--csv-dir',required=True);a.add_argument('--output',required=True);a.add_argument('--utc-offset-seconds',type=int,default=0);a.add_argument('--afternoon-adjust',action='store_true')
    a=sub.add_parser('merge-records');a.add_argument('--inputs',nargs='+',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('export-support');a.add_argument('--prepared',required=True);a.add_argument('--round',type=int,default=0);a.add_argument('--shot',type=int,default=10);a.add_argument('--output',required=True)
    a=sub.add_parser('deploy');a.add_argument('--checkpoint',required=True);a.add_argument('--support',required=True);a.add_argument('--pcap',required=True);a.add_argument('--output',required=True);a.add_argument('--threshold',type=float,default=.5)
    a=sub.add_parser('frozen-pilot');a.add_argument('--checkpoint',required=True);a.add_argument('--source-calibration',required=True);a.add_argument('--target',required=True);a.add_argument('--normal-label',default='BENIGN');a.add_argument('--output',required=True);a.add_argument('--shot',type=int,default=10);a.add_argument('--repeats',type=int,default=100);a.add_argument('--queries-per-class',type=int,default=1500)
    a=sub.add_parser('calibrate');a.add_argument('--scores',required=True);a.add_argument('--test-scores');a.add_argument('--output',required=True)
    a=sub.add_parser('prevalence');a.add_argument('--scores',required=True);a.add_argument('--threshold',type=float,required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('rf-evaluate');a.add_argument('--prepared',required=True);a.add_argument('--dataset-config',required=True);a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=42);a.add_argument('--shots',nargs='+',type=int,default=[1,3,5,10])
    a=sub.add_parser('bootstrap');a.add_argument('--ours',required=True);a.add_argument('--baseline',required=True);a.add_argument('--output',required=True);a.add_argument('--repeats',type=int,default=10000)
    a=p.parse_args(argv);torch.set_num_threads(2)
    if a.command in ['prepare','prepare-binary','train','search']:
        cfg=load_config(a.config)
        if a.output:cfg['output']=a.output
        if a.device:cfg['device']=a.device
        if a.seed is not None:cfg['training_seed']=a.seed
        ds=load_config(a.dataset_config or cfg['dataset_config']);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True);save(root/'run_config.json',cfg);save(root/'dataset_config.json',ds)
        from ..episodes.plans import prepare,prepare_binary,prepare_internal
        prepared=root/'prepared'
        if a.command in ['prepare','prepare-binary']:
            (prepare_binary if a.command=='prepare-binary' else prepare_internal if ds.get('split_protocol')=='packet_block_internal' else prepare)(ds,cfg);return
        if not (prepared/'manifest.json').exists():raise FileNotFoundError('Run prepare or prepare-binary first')
        if a.command=='search':
            from ..training.search import search
            search(ds,cfg,prepared,a.trials);return
        from ..training.trainer import train
        rounds=json.loads((prepared/'manifest.json').read_text())['rounds']
        for rd in rounds:
            for shot in cfg['shots']:train(ds,cfg,prepared,rd['round'],shot)
    elif a.command in ['evaluate','robustness']:
        from ..evaluation.episodic import evaluate
        if a.command=='evaluate':evaluate(a.checkpoint,a.prepared,a.role,a.operator,a.rho,a.output)
        else:
            for op in ['Clean','INP','IBP','RTO','CTO','Gaussian','Joint']:
                for rho in ([0.] if op=='Clean' else [.1,.2,.3,.4,.5]):evaluate(a.checkpoint,a.prepared,a.role,op,rho,Path(a.output)/op/str(rho))
    elif a.command=='import-legacy':
        from ..data.preprocess import import_legacy
        import_legacy(a.input,a.output,a.stat_dim)
    elif a.command=='csv-pcap':
        from ..data.preprocess import csv_pcap
        csv_pcap(a.csv,a.pcap,a.features,a.output,a.time_format,a.utc_offset_seconds)
    elif a.command=='merge-records':
        import pickle
        records=[];contracts=[]
        for path in a.inputs:
            with Path(path).open('rb') as f:records.extend(pickle.load(f))
            side=Path(path).with_suffix('.json')
            if side.exists():contracts.append(json.loads(side.read_text()))
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('wb') as f:pickle.dump(records,f,protocol=4)
        if contracts and all(c.get('input_contract')=='NF57/IP512' for c in contracts):
            assert all(c['feature_order']==contracts[0]['feature_order'] for c in contracts);save(out.with_suffix('.json'),dict(input_contract='NF57/IP512',feature_order=contracts[0]['feature_order'],frontend_sha256=contracts[0].get('frontend_sha256'),nfstream_version=contracts[0].get('nfstream_version'),nfstream_config={k:v for k,v in contracts[0].get('nfstream_config',{}).items() if k!='source'}))
    elif a.command in ['nf-extract','nf-extract-labelled','export-support','deploy']:
        from ..deployment.pipeline import extract,extract_labelled,export_support,deploy
        if a.command=='nf-extract':extract(a.pcap,a.label,a.output)
        elif a.command=='nf-extract-labelled':extract_labelled(a.pcap,a.csv_dir,a.output,a.utc_offset_seconds,a.afternoon_adjust)
        elif a.command=='export-support':export_support(a.prepared,a.round,a.shot,a.output)
        else:deploy(a.checkpoint,a.support,a.pcap,a.output,a.threshold)
    elif a.command=='rf-evaluate':
        from ..evaluation.random_forest import evaluate_forest
        evaluate_forest(a.prepared,load_config(a.dataset_config),a.output,a.seed,a.shots)
    elif a.command=='frozen-pilot':
        from ..evaluation.frozen import frozen_pilot
        frozen_pilot(a.checkpoint,a.source_calibration,a.target,a.normal_label,a.output,a.shot,a.repeats,a.queries_per_class)
    else:
        import pandas as pd
        if a.command=='bootstrap':
            from ..evaluation.bootstrap import paired_bootstrap
            save(a.output,paired_bootstrap(pd.read_csv(a.ours),pd.read_csv(a.baseline),a.repeats))
        elif a.command=='prevalence':
            from ..evaluation.prevalence import sweep
            Path(a.output).parent.mkdir(parents=True,exist_ok=True);sweep(pd.read_csv(a.scores),a.threshold).to_csv(a.output,index=False)
        else:
            from ..evaluation.metrics import calibrate,binary_metrics
            f=pd.read_csv(a.scores);t=calibrate(f.label,f.score);r=dict(threshold=t,calibration=binary_metrics(f.label,f.score,t))
            if a.test_scores:
                test=pd.read_csv(a.test_scores);r['test']=binary_metrics(test.label,test.score,t)
            save(a.output,r)
if __name__=='__main__':main()
