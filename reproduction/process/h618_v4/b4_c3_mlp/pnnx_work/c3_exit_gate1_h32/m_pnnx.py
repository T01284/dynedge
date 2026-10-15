# pnnx model stat
# model inputshape = [1,8,1,1]f32
# FLOPS = 706
# memory OPS = 492

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gate_net_0 = nn.Linear(bias=True, in_features=8, out_features=32)
        self.gate_net_1 = nn.ReLU()
        self.gate_net_2 = nn.Linear(bias=True, in_features=32, out_features=2)

        archive = zipfile.ZipFile('<repo>/results/h618_v4/b4_c3_mlp/pnnx_work/c3_exit_gate1_h32/m.pnnx.bin', 'r')
        self.gate_net_0.bias = self.load_pnnx_bin_as_parameter(archive, 'gate.net.0.bias', (32), 'float32')
        self.gate_net_0.weight = self.load_pnnx_bin_as_parameter(archive, 'gate.net.0.weight', (32,8), 'float32')
        self.gate_net_2.bias = self.load_pnnx_bin_as_parameter(archive, 'gate.net.2.bias', (2), 'float32')
        self.gate_net_2.weight = self.load_pnnx_bin_as_parameter(archive, 'gate.net.2.weight', (2,32), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = torch.flatten(v_0, end_dim=-1, start_dim=1)
        v_2 = self.gate_net_0(v_1)
        v_3 = self.gate_net_1(v_2)
        v_4 = self.gate_net_2(v_3)
        return v_4

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 8, 1, 1, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("<repo>/results/h618_v4/b4_c3_mlp/pnnx_work/c3_exit_gate1_h32/m_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 8, 1, 1, dtype=torch.float)

    torch.onnx.export(net, v_0, "<repo>/results/h618_v4/b4_c3_mlp/pnnx_work/c3_exit_gate1_h32/m_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 8, 1, 1, dtype=torch.float)

    import pnnx
    pnnx.export(net, "<repo>/results/h618_v4/b4_c3_mlp/pnnx_work/c3_exit_gate1_h32/m_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 8, 1, 1, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
