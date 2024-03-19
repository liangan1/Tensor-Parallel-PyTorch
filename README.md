# Tensor-Parallel-PyTorch
## Launch multi-process with torchrun 
torchrun --nnodes=1 --nproc-per-node=2 test_torch_dtensor_tp.py

## Potential issue
- Perfomrance w/o no SHM based all-reduce
- MQA/GQA support
- odd ranks support. 
