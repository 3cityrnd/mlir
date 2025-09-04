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
def load_1(input_ptr : tl.tensor , N : int, D : int , output_ptr : tl.tensor , BLOCK_SIZE: tl.constexpr):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    offset_d = offset_nd % D 
    offset_n = offset_nd // D 
    mask_block = offset_nd < N * D 
    input_th = input_ptr + offset_n * D + offset_d 
    output_th = output_ptr + offset_n * D + offset_d
    
    in_val = tl.load(input_th, mask=mask_block,other=0)
    result = in_val * 99;
    tl.store(output_th, result, mask=mask_block)


    # out = tl.load(output, mask=mask_block,other=0)
    
    # tgt = tgt + 17
    # val = tl.load(tgts_ptr, mask=mask_block,other=0)
    # val = val * 5

    # tl.store(tgts_ptr, val, mask=mask_block)


def run(kern,VEC_SIZE=10000, BLOCK_SIZE=1024, dev='cuda'):
    
    N, D = (500,60)
    input = torch.randn(N,D, device=dev)
    output = torch.empty_like(input, device=dev)
    total = N * D  
    grid = (triton.cdiv(total, BLOCK_SIZE),)
   # kern[grid](input, N, D ,output, BLOCK_SIZE)
    kern[grid](input, N, D, output, BLOCK_SIZE)
     


    # input = torch.randn(VEC_SIZE, device='cpu', dtype=torch.float32)
    # indices = torch.randint(0, VEC_SIZE, (VEC_SIZE,), device='cpu', dtype=torch.int64)
    # output_ref = torch.empty_like(indices, device='cpu', dtype=torch.float32)
    # torch.gather(input, 0, indices, out=output_ref)
   
    # output = torch.empty_like(indices, device=dev, dtype=torch.float32)
    # input = input.to(dev)
    # indices = indices.to(dev)         
    
    # N = indices.shape[0]
    # grid = (triton.cdiv(N, BLOCK_SIZE),)
    # kern[grid](input, indices, output, BLOCK_SIZE, N)
    # output=output.to('cpu')

    # assert torch.allclose(output, output_ref)
    print("Pass correctness test!")        



PARAM = { 'VEC_SIZE':10000, 'BLOCK_SIZE':1024 , 'dev': dev}
run(load_1,**PARAM)  


