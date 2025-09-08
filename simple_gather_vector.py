import torch
import argparse
import time

import triton
import triton.language as tl


def init_npu():    
        try:
            import torch_npu
            return torch_npu.npu.is_available()
        except ImportError:
            return False

npu_on = init_npu()
dev = 'npu' if npu_on else 'cuda'

print(f'NPU : {npu_on}')

@triton.jit
def simple_gather_kernel_for_vector(input, indices, output, BLOCK_SIZE: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    indices_ = tl.load(indices + offsets, mask=mask, other=0)
    data = tl.load(input + indices_, mask=mask, other=0)
    tl.store(output + offsets, data, mask=mask) 


def run(kern,VEC_SIZE=10000, BLOCK_SIZE=1024, dev='cuda'):
    
    input = torch.randn(VEC_SIZE, device='cpu', dtype=torch.float32)
    indices = torch.randint(0, VEC_SIZE, (VEC_SIZE,), device='cpu', dtype=torch.int64)
    output_ref = torch.empty_like(indices, device='cpu', dtype=torch.float32)
    torch.gather(input, 0, indices, out=output_ref)
   
    output = torch.empty_like(indices, device=dev, dtype=torch.float32)
    input = input.to(dev)
    indices = indices.to(dev)         
    
    N = indices.shape[0]
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kern[grid](input, indices, output, BLOCK_SIZE, N)
    output=output.to('cpu')

    assert torch.allclose(output, output_ref)
    print("Pass correctness test!")        



PARAM = { 'VEC_SIZE':10000, 'BLOCK_SIZE':1024 , 'dev': dev}
run(simple_gather_kernel_for_vector,**PARAM)  


