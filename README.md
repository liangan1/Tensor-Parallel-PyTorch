# Tensor-Parallel-PyTorch
## Launch multi-process with torchrun 
torchrun --nnodes=1 --nproc-per-node=2 test_torch_dtensor_tp.py

## Potential issue
- Perfomrance w/o no SHM based all-reduce
- MQA/GQA support
- odd ranks support. 

# All-reduce jit trace test case
## Help
- trace_mode 0 for jit, 1 for torch.compile, 2 for no jit/trace
- TORCH_COMPILE_DEBUG=1 to see the log of torch.compile 

torchrun --nnodes=1 --nproc-per-node=2 test_allreduce_jit_trace.py --trace_mode 0 

# run with deepspeed
deepspeed --bind_cores_to_rank test_allreduce_jit_trace.py --deepspeed --trace_mode 2
