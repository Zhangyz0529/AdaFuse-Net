import unittest,struct
from types import SimpleNamespace
import numpy as np
import torch
class NativeTests(unittest.TestCase):
    def test_ip512_header_and_masks(self):
        try:from adafuse.deployment.nfstream_frontend import AnonymousIPBytes,FEATURES
        except ModuleNotFoundError:self.skipTest('Optional NFStream not installed')
        raw=bytearray(48);raw[0]=0x45;raw[9]=6;raw[10:20]=bytes(range(10));raw[20:24]=b'ABCD';raw[36:38]=b'EF'
        flow=SimpleNamespace(udps=SimpleNamespace());plugin=AnonymousIPBytes();plugin.on_init(SimpleNamespace(ip_packet=bytes(raw)),flow);plugin.on_expire(flow)
        q=np.frombuffer(flow.udps.sequence,dtype=np.uint8);self.assertEqual(len(q),512);self.assertEqual(q[0],0x45);self.assertTrue((q[10:24]==0).all());self.assertTrue((q[36:38]==0).all());self.assertEqual(len(FEATURES),57)
    def test_ste_discrete_forward_and_gradient(self):
        from adafuse.perturbations.adaptive_pgd import discrete_byte_ste
        embedding=torch.nn.Embedding(257,3);x=torch.tensor([[2.4,4.2]],requires_grad=True);integer=x.round().long();expected=embedding(integer).detach()
        with discrete_byte_ste(embedding,x):y=embedding(integer);self.assertTrue(torch.equal(y.detach(),expected));y.sum().backward()
        self.assertIsNotNone(x.grad);self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(x.grad.abs().sum()>0)
