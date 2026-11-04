import unittest
import torch
from adafuse.baselines.registry import Baseline
class BaselineTests(unittest.TestCase):
    def test_local_six_adapters_are_standalone(self):
        torch.set_num_threads(2);xs=torch.randn(4,77);seq=torch.randint(0,256,(4,512))
        for name in ['GP-VAE','MetaTraffic','MFF','Trafficformer-Light','Pre-Contrast','NetMamba']:
            model=Baseline(name,77,128).eval()
            with torch.inference_mode():z=model(xs,seq)
            self.assertEqual(tuple(z.shape),(4,128));self.assertTrue(torch.isfinite(z).all())
