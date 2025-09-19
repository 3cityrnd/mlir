import torch
import sys

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
    offset_d = offset_nd  % D
    offset_n = offset_nd // D
    mask_block = offset_nd < N * D
    input_th = input_ptr + offset_n * D + offset_d 
    output_th = output_ptr + offset_n * D + offset_d
    
    in_val = tl.load(input_th, mask=mask_block,other=0)
    result = in_val * 99
    tl.store(output_th, result, mask=mask_block)

def run(kern, BLOCK_SIZE, dev):
    N, D = (1024,32)
    input = torch.randn(N,D, device=dev)
    output = torch.empty_like(input, device=dev)
    total = N * D  
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    kern[grid](input, N, D, output, BLOCK_SIZE)

    input_cpu = input.to('cpu')
    input_cpu = input_cpu * 99

    assert torch.allclose(input_cpu, output.to('cpu'))
    print("Pass correctness test!")        


PARAM = { 'BLOCK_SIZE':1024 , 'dev': dev}
run(load_1,**PARAM)  
