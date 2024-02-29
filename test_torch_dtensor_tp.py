import torch 
import os
import torch.nn as nn
from copy import deepcopy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, init_device_mesh

NUM_DEVICES = 4

class MLPModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, device=device)

    def forward(self, x):
        act1 = self.relu(self.net1(x))
        print("act1:", act1.shape)
        return self.net2(act1)

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()

mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
mpi_rank = int(os.environ.get('PMI_RANK', -1))
if mpi_world_size > 0:
    os.environ['RANK'] = str(mpi_rank)
    os.environ['WORLD_SIZE'] = str(mpi_world_size)
else:
    # set the default rank and world size to 0 and 1
    os.environ['RANK'] = str(os.environ.get('RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
os.environ['MASTER_PORT'] = '29500'  # your master port

device_type = "cpu"
# device_mesh = DeviceMesh(
#         device_type,
#         torch.arange(0, NUM_DEVICES),
#     )

device_mesh = init_device_mesh(device_type, mesh_shape=(2,))

inp_size = [8, 10]
# Ensure all tp ranks have same input.
torch.manual_seed(0)
inp = torch.rand(*inp_size, device=device_type)
model = MLPModule(device_type)
print("Original model:", model)
model_tp = deepcopy(model)

# Shard module and initialize optimizer.
parallelize_plan = {
    "net1": ColwiseParallel(),
    "net2": RowwiseParallel(),
}
model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)
print("After Tensor Parallel model_tp: ", model_tp)

with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):
    output = model(inp)
    output_tp = model_tp(inp)