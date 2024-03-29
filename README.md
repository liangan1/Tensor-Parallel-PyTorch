# Tensor-Parallel-PyTorch
## Launch multi-process with torchrun 
torchrun --nnodes=1 --nproc-per-node=2 test_torch_dtensor_tp.py

## Potential issue
- Perfomrance w/o no SHM based all-reduce
- MQA/GQA support
- odd ranks support. 

# All-reduce jit trace test case
## trace_mode 0 for jit 1 for torch.compile 
torchrun --nnodes=1 --nproc-per-node=2 test_allreduce_jit_trace.py --trace_mode 0 
