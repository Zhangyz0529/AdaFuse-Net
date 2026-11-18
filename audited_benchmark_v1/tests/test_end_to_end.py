"""Generated test fixtures are temporary and are never part of the release."""
import unittest,tempfile,pickle,json
from pathlib import Path
import numpy as np
import torch
from adafuse.episodes.plans import prepare
from adafuse.training.trainer import train
from adafuse.evaluation.episodic import evaluate

class PipelineTests(unittest.TestCase):
    def test_prepare_train_evaluate_relocated(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(3);rows=[]
            for label,n in [('BENIGN',300),('A',80),('B',80),('C',80)]:
                for i in range(n):rows.append(dict(label=label,stat_raw=rng.normal(size=7).astype('float32'),seq_feature=rng.integers(0,256,512,dtype='int16')))
            data=root/'data.pkl'
            with data.open('wb') as f:pickle.dump(rows,f)
            ds=dict(name='toy',input=str(data),stat_dim=7,normal_label='BENIGN',attack_classes=['A','B','C'],split_seed=42,normal_split_ratios=[.6,.2,.2],novel_classes_per_round=1,round_val_novel_classes=1)
            cfg=dict(output=str(root/'run'),training_seed=42,episode_seed=42,noise_seed=20261028,updates=2,episodes=2,shots=[1],embed_dim=32,device='cpu',lr=.0003,weight_decay=.0001,temperature=.1,lambda_scl=.1,lambda_regularizer=.01,variant='AdaFuse-Net',distance='squared_euclidean',training_noise='Clean')
            prepared=prepare(ds,cfg);ck=train(ds,cfg,prepared,0,1);a=evaluate(ck,prepared,output=root/'clean');b=evaluate(ck,prepared,operator='IBP',rho=.3,output=root/'noise')
            self.assertEqual(len(a),2);self.assertEqual(len(b),2);self.assertTrue(np.isfinite(a.MacroF1).all());self.assertTrue((root/'noise/predictions.jsonl.gz').exists())
if __name__=='__main__':unittest.main()
