import argparse
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule

# ============================================================================
# Default image paths (adjust if you want)
# ============================================================================
DEFAULT_INPUT  = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\FiltroLaplaciano\bird_9000x9000.jpg"
DEFAULT_OUTPUT = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\FiltroLaplaciano\bird_65x65.png"

# ============================================================================
# CUDA kernel source:
#   - laplacian3x3_u8_to_rgb  : classic 3x3 Laplacian (8 neighbors)
#   - conv_log_u8_to_f        : general NxN Laplacian of Gaussian (LoG) convolution
#   - f_abs_to_rgb_u8         : abs + clamp and write grayscale RGB
# All comments are ASCII to avoid encoding issues on Windows.
# ============================================================================
KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ size_t IDX(int x, int y, int w){
    return (size_t)y * w + x;
}

// --------------------------------------------------------------------------
// 3x3 Laplacian (8-neighbor) on grayscale input:
//   gray (uint8) -> rgb (uint8), using abs and clamp to [0,255].
// Kernel:
//   [-1 -1 -1]
//   [-1  8 -1]
//   [-1 -1 -1]
// --------------------------------------------------------------------------
__global__ void laplacian3x3_u8_to_rgb(const unsigned char* __restrict__ gray,
                                       unsigned char* __restrict__ rgb,
                                       int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int acc = 0;

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x    , 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), y, w)];
    acc +=  8 * (int)gray[IDX(x, y, w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), y, w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x    , 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];

    int v = acc >= 0 ? acc : -acc;  // abs
    if (v > 255) v = 255;

    unsigned char u = (unsigned char)v;
    size_t o = IDX(x, y, w) * 3;
    rgb[o + 0] = u;
    rgb[o + 1] = u;
    rgb[o + 2] = u;
}

// --------------------------------------------------------------------------
// General LoG convolution:
//   gray (uint8) * K (NxN float) -> out (float), border handling with clamp.
// --------------------------------------------------------------------------
__global__ void conv_log_u8_to_f(const unsigned char* __restrict__ gray,
                                 const float* __restrict__ K, int N,
                                 float* __restrict__ out,
                                 int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int ky = -r; ky <= r; ++ky){
        int yy = clampi(y + ky, 0, h - 1);
        int krow = (ky + r) * N;
        for (int kx = -r; kx <= r; ++kx){
            int xx = clampi(x + kx, 0, w - 1);
            acc += (float)gray[IDX(xx, yy, w)] * K[krow + (kx + r)];
        }
    }

    out[IDX(x, y, w)] = acc;
}

// --------------------------------------------------------------------------
// Convert float response to RGB uint8 using abs() and clamp to [0,255].
// Result is written as grayscale replicated on R,G,B.
// --------------------------------------------------------------------------
__global__ void f_abs_to_rgb_u8(const float* __restrict__ in,
                                unsigned char* __restrict__ out_rgb,
                                int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float v = in[IDX(x, y, w)];
    if (v < 0.f) v = -v;
    if (v > 255.f) v = 255.f;

    unsigned char u = (unsigned char)(v + 0.5f);
    size_t o = IDX(x, y, w) * 3;
    out_rgb[o + 0] = u;
    out_rgb[o + 1] = u;
    out_rgb[o + 2] = u;
}

} // extern "C"
"""


# ============================================================================
# Helper: convert RGB (uint8) to grayscale (uint8) in host, using same formula
#         as your C++ version (luma Rec.709 approx).
# ============================================================================
def rgb_to_gray_host(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: numpy array of shape (h, w, 3), dtype=uint8
    returns: flattened grayscale array of length w*h, dtype=uint8
    """
    # Coefficients: 0.2126 R + 0.7152 G + 0.0722 B
    gray = (0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2] + 0.5).astype(np.uint8)
    return gray.reshape(-1)


# ============================================================================
# Helper: create LoG kernel NxN (same logic as make_log_kernel in C++)
#   - sigma = N / 6
#   - LoG(x,y) = -((r^2 - 2*sigma^2)/sigma^4) * exp(-r^2 / (2*sigma^2))
#   - correction so that approximate sum(K) ~= 0
# ============================================================================
def make_log_kernel(N: int) -> np.ndarray:
    """
    Build an NxN Laplacian of Gaussian kernel as float32.
    """
    c = N // 2
    sigma = N / 6.0
    s2 = sigma * sigma
    s4 = s2 * s2

    K = np.empty((N, N), dtype=np.float32)
    s = 0.0

    for yy in range(-c, c + 1):
        for xx in range(-c, c + 1):
            r2 = float(xx * xx + yy * yy)
            val = -((r2 - 2.0 * s2) / s4) * np.exp(-r2 / (2.0 * s2))
            K[yy + c, xx + c] = val
            s += float(val)

    # Subtract the mean so that the kernel sum is close to zero
    corr = s / float(N * N)
    K -= np.float32(corr)
    return K.astype(np.float32)


def main():
    # ------------------------------------------------------------------------
    # Parse basic CLI options:
    #   --input, --output: image paths
    #   --blockX, --blockY: CUDA block dimensions
    # The choice between Laplacian 3x3 and LoG NxN will be made via user N.
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser("Laplacian / LoG filter in PyCUDA")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--blockX", type=int, default=5)
    parser.add_argument("--blockY", type=int, default=5)
    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Ask user for kernel size N:
    #   N = 3  -> classic Laplacian 3x3
    #   N > 3  and odd -> LoG NxN
    # ------------------------------------------------------------------------
    while True:
        try:
            userN = int(input("Ingrese N del kernel (3 = Laplaciano 3x3; impar >=5 = LoG NxN): "))
            if userN < 3:
                print("N debe ser >= 3.")
                continue
            if userN % 2 == 0:
                print("N debe ser impar. Se incrementa en 1.")
                userN += 1
            N = userN
            break
        except ValueError:
            print("Entrada invalida. Ingrese un entero, por ejemplo 3, 9, 21, 65.")

    useLoG = (N != 3)

    # ------------------------------------------------------------------------
    # Load image as RGB uint8 with Pillow and convert to grayscale on host
    # ------------------------------------------------------------------------
    img = Image.open(args.input).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    h, w, _ = rgb.shape

    gray = rgb_to_gray_host(rgb)  # flattened, length = w*h

    Npix = w * h
    bytes_gray = Npix
    bytes_rgb  = Npix * 3

    # ------------------------------------------------------------------------
    # Configure CUDA block/grid dimensions
    # ------------------------------------------------------------------------
    blockX = max(1, args.blockX)
    blockY = max(1, args.blockY)
    while blockX * blockY > 1024:
        blockY //= 2
        if blockY < 1:
            blockY = 1
            break

    block = (blockX, blockY, 1)
    grid  = ((w + blockX - 1) // blockX,
             (h + blockY - 1) // blockY,
             1)

    print(f"Imagen: {w}x{h}")
    print(f"N = {N}  modo = {'LoG NxN' if useLoG else 'Laplaciano 3x3'}")
    print(f"Block = {block}, Grid = {grid}\n")

    # ------------------------------------------------------------------------
    # Allocate GPU buffers:
    #   d_gray : grayscale input
    #   d_rgb  : RGB output
    #   d_tmpF : float buffer for LoG response (only if useLoG)
    #   d_K    : LoG kernel NxN in float32 (only if useLoG)
    # Also measure memory usage delta with cuda.mem_get_info.
    # ------------------------------------------------------------------------
    free0, _ = cuda.mem_get_info()

    d_gray = cuda.mem_alloc(bytes_gray)
    d_rgb  = cuda.mem_alloc(bytes_rgb)

    d_tmpF = None
    d_K    = None
    if useLoG:
        d_tmpF = cuda.mem_alloc(Npix * 4)      # float32 per pixel
        d_K    = cuda.mem_alloc(N * N * 4)     # float32 for NxN kernel

    free1, _ = cuda.mem_get_info()
    delta_bytes = free0 - free1

    # Copy grayscale image to device
    cuda.memcpy_htod(d_gray, gray)

    # If LoG is used, build the kernel on host and copy to GPU
    if useLoG:
        h_K = make_log_kernel(N)
        cuda.memcpy_htod(d_K, h_K)

    # ------------------------------------------------------------------------
    # Compile CUDA kernels with architecture flag based on current device
    # ------------------------------------------------------------------------
    dev = cuda.Device(0)
    cc_major, cc_minor = dev.compute_capability()
    arch_flag = f"-arch=sm_{cc_major}{cc_minor}"

    mod = DynamicSourceModule(KERNEL_SRC, options=[arch_flag])
    laplacian3x3 = mod.get_function("laplacian3x3_u8_to_rgb")
    conv_log     = mod.get_function("conv_log_u8_to_f")
    f_abs_to_rgb = mod.get_function("f_abs_to_rgb_u8")

    # ------------------------------------------------------------------------
    # Run only the GPU kernels and measure their time (no transfers included)
    # ------------------------------------------------------------------------
    start = cuda.Event()
    stop  = cuda.Event()
    start.record()

    if not useLoG:
        # Classic Laplacian 3x3
        laplacian3x3(d_gray, d_rgb,
                     np.int32(w), np.int32(h),
                     block=block, grid=grid)
    else:
        # LoG NxN: first convolution (gray * K -> tmpF), then abs+clamp to RGB
        conv_log(d_gray, d_K, np.int32(N), d_tmpF,
                 np.int32(w), np.int32(h),
                 block=block, grid=grid)

        f_abs_to_rgb(d_tmpF, d_rgb,
                     np.int32(w), np.int32(h),
                     block=block, grid=grid)

    stop.record()
    stop.synchronize()
    elapsed_s = stop.time_since(start) / 1000.0

    # ------------------------------------------------------------------------
    # Copy RGB result back to host and save as image
    # ------------------------------------------------------------------------
    h_rgb = np.empty(bytes_rgb, dtype=np.uint8)
    cuda.memcpy_dtoh(h_rgb, d_rgb)
    out_img = h_rgb.reshape(h, w, 3)
    Image.fromarray(out_img).save(args.output)

    # ------------------------------------------------------------------------
    # Compute theoretical memory usage (approximate) and print report
    # ------------------------------------------------------------------------
    if not useLoG:
        # gray + rgb
        bytes_theory = bytes_gray + bytes_rgb
    else:
        # gray + tmpF + K + rgb
        bytes_theory = (bytes_gray +
                        Npix * 4 +
                        N * N * 4 +
                        bytes_rgb)

    memMB_delta  = delta_bytes   / (1024.0 * 1024.0)
    memMB_theory = bytes_theory  / (1024.0 * 1024.0)

    print("\n================ PYCUDA – Laplaciano / LoG ================")
    print(f"Tiempo paralelo (solo kernels) (s): {elapsed_s:.6f}")
    print(f"Memoria usada (delta mem_get_info) MB: {memMB_delta:.2f}")
    print(f"Memoria buffers (teorica) MB:         {memMB_theory:.2f}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
