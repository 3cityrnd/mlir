import torch
import triton
import triton.language as tl


@triton.jit
def test_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    index_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    index_block_start = pid * BLOCK_SIZE
    offsets = index_block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_size
    indicies = tl.load(index_ptr + offsets, mask=mask)

    elements = tl.load(input_ptr + indicies)
    tl.store(output_ptr + offsets, elements, mask=mask)


input = torch.rand(1024).to('npu:0')
indexes = torch.randint(0, 1024, (96, ), dtype=torch.int32).to('npu:0')
output = torch.empty(indexes.numel()).to('npu:0')
n_elements = output.numel()
block_size = 64
test_kernel[(2,)](input, indexes, output, n_elements, BLOCK_SIZE=block_size)
expected = torch.index_select(input, 0, indexes).to('cpu')
output_cpu = output.to('cpu')
print(output_cpu)
print(expected)
assert torch.equal(expected, output_cpu)

