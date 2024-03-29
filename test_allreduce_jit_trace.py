import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


#import oneccl_bindings_for_pytorch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-dev', type=str, default='cpu', help='Device type to use: cpu, xpu')
parser.add_argument('--dist_url', default='127.0.0.1', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_port', default='29800', type=str, help='url port used to set up distributed training')
parser.add_argument('--backend', default='gloo', type=str, help='backend used to set up distributed training')
parser.add_argument('--trace_mode', default=0, type=int, help='0 is for jit  and 1 is for torch.compile')
args = parser.parse_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        torch.distributed.all_reduce(input)
        return self.linear(input)


if __name__ == "__main__":

    dist.init_process_group(backend=args.backend)

    device = 'cpu'

    model = Model().to(device)

    input = torch.randn(4, 4).to(device)
    if args.trace_mode == 0:
        jit_model = torch.jit.trace(model, input)
        jit_model = torch.jit.freeze(jit_model)
        output = jit_model(input)
    else:
        model_compile = torch.compile(model)
        output = model_compile(input)
    print(output)
