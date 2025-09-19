import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def complex_load_kernel(
    in_ptr, out_ptr,ind_ptr,
    X, Y, Z,
    x_stride, y_stride, z_stride, 
    out_stride_x, out_stride_y, out_stride_z,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_Z: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    offset_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offset_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offset_z = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
    
    mask = (
        (offset_x[:, None, None] < X) & 
        (offset_y[None, :, None] < Y) & 
        (offset_z[None, None, :] < Z)
    )
   
    indices = tl.load(ind_ptr + 
                       offset_x[:, None, None] * x_stride +
                       offset_y[None, :, None] * y_stride +
                       offset_z[None, None, :] * z_stride,
                       mask=mask,
                       other=0)

    source_offsets = (
            offset_x[:, None, None] * x_stride +
            offset_y[None, :, None] * y_stride +
            indices * z_stride 
        )

    data = tl.load(
        in_ptr + source_offsets,
        mask=mask,
        other=0.0
    )
    
    out_offsets = (
        offset_x[:, None, None] * out_stride_x +
        offset_y[None, :, None] * out_stride_y +
        offset_z[None, None, :] * out_stride_z
    )
    
    tl.store(out_ptr + out_offsets, data, mask=mask)

def complex_load_torch(
    in_tensor, indices):
    in_expanded = in_tensor.unsqueeze(2)  # shape: (X, Y, 1, Z_in)
    indices_expanded = indices.unsqueeze(3)  # shape: (X, Y, Z_out, 1)
    
    out = torch.take_along_dim(in_expanded, indices_expanded, dim=3)
    return out.squeeze(3)  # shape: (X, Y, Z_out)

def test_complex_load():
    X, Y, Z = 32, 24, 16
    BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z = 16, 8, 8
    
    torch.manual_seed(42)
    input_tensor = torch.randn(X, Y, Z, dtype=torch.float32, device='npu')
    output_tensor = torch.zeros_like(input_tensor).to('npu')
    indicies = torch.randint(0, Z, (X, Y, Z)).to('npu')

    x_stride = input_tensor.stride(0)
    y_stride = input_tensor.stride(1)
    z_stride = input_tensor.stride(2)
    
    out_stride_x = output_tensor.stride(0)
    out_stride_y = output_tensor.stride(1)
    out_stride_z = output_tensor.stride(2)
    
    grid = (
        triton.cdiv(X, BLOCK_SIZE_X),
        triton.cdiv(Y, BLOCK_SIZE_Y),
        triton.cdiv(Z, BLOCK_SIZE_Z)
    )
    
    complex_load_kernel[grid](
        input_tensor, output_tensor,indicies,
        X, Y, Z,
        x_stride, y_stride, z_stride,
        out_stride_x, out_stride_y, out_stride_z,
        BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z
    )
    output_expected_tensor = complex_load_torch(input_tensor.to('cpu'), indicies.to('cpu'))
    
    if torch.allclose(output_expected_tensor, output_tensor.to('cpu')):
        print('PASS')
    else:
        print('Fail')
    print(f"\nTensor shape: {input_tensor.shape}")
    print(f"Block size: ({BLOCK_SIZE_X}, {BLOCK_SIZE_Y}, {BLOCK_SIZE_Z})")
    print(f"Grid size: {grid}")
    print(f"Strides - input: ({x_stride}, {y_stride}, {z_stride})")
    print(f"Strides - output: ({out_stride_x}, {out_stride_y}, {out_stride_z})")


if __name__ == "__main__":
    test_complex_load()
