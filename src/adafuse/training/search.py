"""Validation-only Optuna; test is evaluated only after selection."""
from pathlib import Path
import copy,json,pickle,fcntl
import optuna
import pandas as pd
from .trainer import train
from ..evaluation.episodic import evaluate
from ..utils.common import save

def search(dataset,cfg,prepared,trials=200):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True);lock=(root/'.study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sampler_file=root/'sampler.pkl';sampler=pickle.loads(sampler_file.read_bytes()) if sampler_file.exists() else optuna.samplers.TPESampler(seed=cfg['training_seed']);study=optuna.create_study(direction='maximize',sampler=sampler,pruner=optuna.pruners.NopPruner(),study_name='adafuse',storage='sqlite:///'+str((root/'study.db').resolve()),load_if_exists=True)
    rounds=json.loads((Path(prepared)/'manifest.json').read_text())['rounds']
    def objective(trial):
        c=copy.deepcopy(cfg);c.update(embed_dim=trial.suggest_categorical('embed_dim',[64,128,256]),lr=trial.suggest_float('lr',1e-5,1e-2,log=True),weight_decay=trial.suggest_float('weight_decay',1e-6,1e-3,log=True),temperature=trial.suggest_float('temperature',.05,.2),lambda_scl=trial.suggest_float('lambda_scl',.01,1.,log=True),lambda_regularizer=trial.suggest_float('lambda_regularizer',.001,.1,log=True),output=str(root/'trials'/f'trial{trial.number}'))
        scores=[]
        for rd in rounds:
            for k in c['shots']:
                ck=train(dataset,c,prepared,rd['round'],k);f=evaluate(ck,prepared,role='validation');scores.append(float(f.groupby('task').MacroF1.mean().mean()))
        return sum(scores)/len(scores)
    def persist(study,trial):
        tmp=sampler_file.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(study.sampler));tmp.replace(sampler_file)
    if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))>trials:raise ValueError('Completed trials already exceed requested budget; no silent selection truncation')
    # An interrupted trial is not COMPLETE and cannot count toward the budget.
    for t in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):study.tell(t.number,state=optuna.trial.TrialState.FAIL)
    while len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))<trials:study.optimize(objective,n_trials=1,callbacks=[persist],catch=(FloatingPointError,))
    best=study.best_trial;save(root/'SELECTED_BEFORE_TEST.json',dict(trial=best.number,value=best.value,params=best.params,target_complete_trials=trials))
    for ck in (root/'trials'/f'trial{best.number}'/'checkpoints').glob('round*/shot*/model.pt'):
        evaluate(ck,prepared,output=root/'selected_test'/ck.parent.parent.name/ck.parent.name)
