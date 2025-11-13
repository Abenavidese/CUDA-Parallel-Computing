# boxblur_pycuda.py
import os, sys, argparse, time
# --- Para Python 3.8+ en Windows: registra carpetas de CUDA (ajusta versión si hace falta) ---
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin")
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64")
except Exception:
    pass

import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit  # crea contexto
from pycuda.compiler import DynamicSourceModule  # NVRTC (no requiere cl.exe)

# ====== Parámetros por defecto (como tu C++) ======
DEFAULT_INPUT  = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Box_Blur\cuenca9000.jpg"
DEFAULT_OUTPUT = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Box_Blur\blox_blur_pycuda_output.jpg"
PASSES = 3  # fijo en código, como pidió tu inge

KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// Horizontal: srcRGB (uint8) -> tmpRGB (float) promediando en X (ventana N)
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ src,
                                  float* __restrict__ tmp,
                                  int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float R = 0.f, G = 0.f, B = 0.f;

    // acumula sobre X
    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        size_t s = ((size_t)y * w + xx) * 3;
        R += (float)src[s + 0];
        G += (float)src[s + 1];
        B += (float)src[s + 2];
    }

    size_t o = ((size_t)y * w + x) * 3;
    tmp[o + 0] = R;
    tmp[o + 1] = G;
    tmp[o + 2] = B;
}

// Vertical: tmpRGB (float) -> dstRGB (uint8) promediando en Y (ventana N)
__global__ void box_vert_f_to_u8(const float* __restrict__ tmp,
                                 unsigned char* __restrict__ dst,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float invN = 1.0f / (float)N;
    float R = 0.f, G = 0.f, B = 0.f;

    // acumula sobre Y
    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        size_t s = ((size_t)yy * w + x) * 3;
        R += tmp[s + 0];
        G += tmp[s + 1];
        B += tmp[s + 2];
    }

    size_t o = ((size_t)y * w + x) * 3;

    int r8 = (int)(R * invN * invN + 0.5f); // invN horizontal * invN vertical = invN^2
    int g8 = (int)(G * invN * invN + 0.5f);
    int b8 = (int)(B * invN * invN + 0.5f);

    r8 = r8 < 0 ? 0 : (r8 > 255 ? 255 : r8);
    g8 = g8 < 0 ? 0 : (g8 > 255 ? 255 : g8);
    b8 = b8 < 0 ? 0 : (b8 > 255 ? 255 : b8);

    dst[o + 0] = (unsigned char)r8;
    dst[o + 1] = (unsigned char)g8;
    dst[o + 2] = (unsigned char)b8;
}

} // extern "C"
"""

def main():
    parser = argparse.ArgumentParser(description="Box Blur separable PyCUDA (RGB), sin OpenCV.")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT, help="Ruta imagen de entrada")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ruta imagen de salida")
    parser.add_argument("--N",      type=int, default=65,           help="Tamaño ventana (impar >=3)")
    parser.add_argument("--blockX", type=int, default=32,           help="Bloques X")
    parser.add_argument("--blockY", type=int, default=32,           help="Bloques Y")
    args = parser.parse_args()

    N = max(3, args.N)
    if (N % 2) == 0:  # forzar impar
        N += 1
        print(f"[Aviso] N debe ser impar. Usando N={N}")

    blockX = max(1, args.blockX)
    blockY = max(1, args.blockY)

    # Cargar imagen con Pillow (RGB uint8)
    img = Image.open(args.input).convert("RGB")
    np_img = np.array(img, dtype=np.uint8)              # (h,w,3)
    h, w, _ = np_img.shape
    Npix = int(h) * int(w)
    bytesRGB = Npix * 3

    # Limitar hilos por bloque (<=1024)
    while blockX * blockY > 1024:
        blockY = max(1, blockY // 2)

    block = (int(blockX), int(blockY), 1)
    grid  = ((w + block[0] - 1) // block[0],
             (h + block[1] - 1) // block[1], 1)

    print(f"Imagen: {w}x{h}  N={N}  PASSES={PASSES}  block=({block[0]},{block[1]}) grid=({grid[0]},{grid[1]})")

    # Preparar buffers host contiguos
    h_in  = np_img.reshape(-1).astype(np.uint8).copy()  # 1D
    h_out = np.empty_like(h_in)

    # ===== Reservas GPU y delta memoria =====
    free0, total0 = cuda.mem_get_info()
    d_in  = cuda.mem_alloc(bytesRGB)
    d_out = cuda.mem_alloc(bytesRGB)
    d_tmp = cuda.mem_alloc(bytesRGB * 4)  # float por canal
    free1, total1 = cuda.mem_get_info()
    delta_bytes = free0 - free1  # “overhead real” incluido

    # H->D (fuera del tiempo medido)
    cuda.memcpy_htod(d_in, h_in)

    # ===== Compilar kernels con NVRTC, -arch según la GPU =====
    dev = cuda.Device(0)
    cc_major, cc_minor = dev.compute_capability()
    arch_flag = f"-arch=sm_{cc_major}{cc_minor}"
    mod = DynamicSourceModule(KERNEL_SRC, options=[arch_flag])

    box_horiz = mod.get_function("box_horiz_u8_to_f")
    box_vert  = mod.get_function("box_vert_f_to_u8")

    # ===== Medición SOLO procesamiento (kernels) =====
    start = cuda.Event(); stop = cuda.Event()
    start.record()

    for p in range(PASSES):
        box_horiz(d_in, d_tmp, np.int32(w), np.int32(h), np.int32(N),
                  block=block, grid=grid)
        box_vert(d_tmp, d_out, np.int32(w), np.int32(h), np.int32(N),
                 block=block, grid=grid)
        # swap in/out
        d_in, d_out = d_out, d_in

    stop.record(); stop.synchronize()
    ms = stop.time_since(start)
    seg = ms / 1000.0

    # D->H (fuera del tiempo medido)
    cuda.memcpy_dtoh(h_out, d_in)

    # Guardar salida
    out_img = h_out.reshape(h, w, 3)
    Image.fromarray(out_img, mode="RGB").save(args.output)

    # Reporte
    bytes_buffers = bytesRGB + bytesRGB + (bytesRGB * 4)  # in + out + tmp(float3)
    memMB_delta  = delta_bytes   / (1024.0 * 1024.0)
    memMB_theory = bytes_buffers / (1024.0 * 1024.0)

    print("\n================ PYCUDA – Box Blur (separable) ================")
    print(f"Tiempo Paralelo (s): {seg:.6f}")
    print(f"Memoria Usada (delta cuda.mem_get_info) MB: {memMB_delta:.2f}")
    print(f"Memoria Buffers (teórica) MB: {memMB_theory:.2f}")
    print(f"Salida: {args.output}")

if __name__ == "__main__":
    main()
