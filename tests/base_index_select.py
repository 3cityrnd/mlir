import triton
import triton.language as tl
import torch
import torch_npu

def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def test_index_select(src_shape, dim, indice_shape, dtype):

    def torch_func(x0, dim, indices):
        res = torch.index_select(x0, dim, indices)
        return res
    
    @triton.jit
    def basic_index_select(in_ptr, indices_ptr, out_ptr, dim,
        g_stride: tl.constexpr, indice_length: tl.constexpr, 
        g_block : tl.constexpr, g_block_sub: tl.constexpr, other_block:tl.constexpr):
        g_begin=tl.program_id(0) * g_block
        for goffs in range(0, g_block, g_block_sub):
            g_idx=tl.arange(0, g_block_sub) + g_begin + goffs
            g_mask = g_idx < indice_length
            indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
            for other_offset in range(0, g_stride, other_block): 
                tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
                other_idx = tl.arange(0, other_block) + other_offset
                other_mask = other_idx < g_stride
                for i in range(0, g_block_sub):
                    gather_offset = tl.get_element(indices, (i,)) * g_stride
                    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, other_block), strides=(g_stride, 1))
                tl.store(out_ptr + g_idx[:,None] * g_stride + other_idx[None,:], tmp_buf, g_mask[:,None] & other_mask[None,:])

    def triton_func(x0, dim, indices, handle):
        sz = list(x0.shape)
        sz[dim]=len(indices)
        out = torch.empty(tuple(sz), dtype=x0.dtype, device=device)
        g_stride = x0.stride(dim)
        indice_length=indices.numel()
        num_vec_core=80
        g_block = (indice_length - 1) // num_vec_core + 1
        enable_multi_buffer=True
        available_ub_space = (128 * 1024) // (x0.element_size() * (2 if enable_multi_buffer else 1))
       
        if g_stride * 2 < available_ub_space:
            other_block = g_stride
            g_block_sub = available_ub_space // other_block
        else:
            other_block = available_ub_space
            g_block_sub = 1
        handle[num_vec_core, 1, 1](x0, indices, out, dim, g_stride = g_stride, indice_length=indice_length, 
        g_block = g_block, g_block_sub = g_block_sub, other_block = other_block)
        return out

    device = torch.device('npu:0')
    x0 = generate_tensor(shape=src_shape, dtype=dtype).to(device)
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32, device=device)

    torch_ref = torch_func(x0, dim, indices).to('cpu')
    triton_cal = triton_func(x0, dim, indices, basic_index_select).to('cpu')
    assert torch.allclose(torch_ref, triton_cal)
    print("Test done")

test_index_select((3200,512), 0, (1024*128,), 'float32')