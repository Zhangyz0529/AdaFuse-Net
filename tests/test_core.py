import unittest,tempfile,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adafuse.models.network import FactorialNet
from adafuse.losses.objectives import class_conditional_decoupling,supervised_contrastive_loss
from adafuse.perturbations.operators import corrupt_sequence
from adafuse.evaluation.metrics import calibrate,binary_metrics
from adafuse.evaluation.bootstrap import paired_bootstrap
from adafuse.utils.common import reset

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_fixed_length_operators(self):
        x=np.full((3,512),42,dtype='int16');ids=np.arange(3)
        for name in ['INP','IBP','RTO','CTO']:
            y=corrupt_sequence(x,ids,'test',name,.3);self.assertEqual(y.shape,x.shape);self.assertTrue(np.array_equal(y,corrupt_sequence(x,ids,'test',name,.3)))
            self.assertTrue(np.array_equal(x,corrupt_sequence(x,ids,'test',name,0)))
        y=corrupt_sequence(x,ids,'test','INP',.3);self.assertTrue(np.array_equal(y[:,:179],x[:,:179]));self.assertTrue(np.array_equal(y[:,332:],x[:,332:]))
        y=corrupt_sequence(x,ids,'test','CTO',.3);self.assertTrue((y[:,-153:]==255).all())
    def test_global_scalars_and_gate(self):
        reset(42);m=FactorialNet(True,True,True,embed_dim=32,stat_dim=25).eval();x=torch.randn(4,25);q=torch.randint(0,256,(4,512))
        with torch.inference_mode():a=m(x,q,True);b=m(x,q,True)
        self.assertTrue(torch.equal(a['z'],b['z']));self.assertEqual(m.fusion_block.w_se.numel(),1);self.assertEqual(float(m.fusion_block.w_se),1.)
        self.assertTrue(((a['w_seq']>=0)&(a['w_seq']<=1)).all());self.assertTrue(torch.allclose(a['z'].norm(dim=1),torch.ones(4),atol=1e-6))
    def test_loss_masks(self):
        h=torch.randn(4,8,requires_grad=True);same=torch.zeros(4,dtype=torch.long);self.assertEqual(float(class_conditional_decoupling(h,h,same)),0.)
        self.assertEqual(float(supervised_contrastive_loss(h,torch.arange(4),.1)),0.)
        v=class_conditional_decoupling(h,h,torch.tensor([0,0,1,1]));self.assertTrue(torch.isfinite(v));v.backward();self.assertTrue(torch.isfinite(h.grad).all())
    def test_threshold_ties(self):
        y=np.array([0,0,1,1]);s=np.array([.1,.7,.7,.9]);self.assertEqual(calibrate(y,s,1.),.7)
        self.assertTrue(binary_metrics(y,s,1.)['Precision_undefined'])
    def test_paired_bootstrap(self):
        rows=[]
        for t in ['a','b']:
            for e in range(5):
                for c,v in [('Clean',.5),('IBP30',.4)]:rows.append(dict(dataset='d',training_seed=42,shot=1,task=t,episode=e,condition=c,shared_episode_group=f'{t}/{e}',input_sha256=f'{t}/{e}/{c}',MacroF1=v))
        baseline=pd.DataFrame(rows);ours=baseline.copy();ours['MacroF1']+=.1;out=paired_bootstrap(ours,baseline,repeats=100)
        self.assertAlmostEqual(out[0]['estimate'],.1);self.assertAlmostEqual(out[2]['estimate'],0.)
        with self.assertRaises(ValueError):paired_bootstrap(pd.concat([ours,ours.iloc[:1]]),baseline,repeats=2)

if __name__=='__main__':unittest.main()
