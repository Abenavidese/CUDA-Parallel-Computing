import argparse
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule

# ============================================================================
#  Rutas por defecto
# ============================================================================
DEFAULT_INPUT  = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\FiltroGaussiano\gauss.jpg"
DEFAULT_OUTPUT = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\FiltroGaussiano\gauss_251.png"

# ============================================================================
#  Kernels CUDA para Gaussiano separable (gris -> gris -> RGB gris)
# ============================================================================
KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// -------------------- u8 -> float --------------------
__global__ void u8_to_f(const unsigned char* __restrict__ in_u8,
                        float* __restrict__ out_f,
                        int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out_f[y * w + x] = (float)in_u8[y * w + x];
}

// -------------------- Gauss horizontal 1D --------------------
__global__ void gauss_horiz_f(const float* __restrict__ in,
                              float* __restrict__ tmp,
                              int w, int h,
                              const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;
    
    // Calcular el índice base de la fila para mejorar el acceso a memoria
    int row_offset = y * w;

    for(int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        acc += in[row_offset + xx] * k1d[i + r];
    }
    tmp[row_offset + x] = acc;
}

// -------------------- Gauss vertical 1D --------------------
__global__ void gauss_vert_f(const float* __restrict__ tmp,
                             float* __restrict__ out,
                             int w, int h,
                             const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for(int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        acc += tmp[yy * w + x] * k1d[j + r];
    }
    
    // Verificar bounds antes de escribir
    int out_idx = y * w + x;
    if (out_idx < w * h) {
        out[out_idx] = acc;
    }
}

// -------------------- float -> RGB u8 --------------------
__global__ void f_to_rgb_u8(const float* __restrict__ in_f,
                            unsigned char* __restrict__ out_rgb,
                            int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int in_idx = y * w + x;
    if (in_idx >= w * h) return;  // Verificación adicional de bounds
    
    float v = in_f[in_idx];
    
    // Clamp más robusto
    v = fmaxf(0.0f, fminf(255.0f, v));
    
    unsigned char u = (unsigned char)(v + 0.5f);
    
    // Cálculo seguro del índice RGB
    size_t o = ((size_t)y * (size_t)w + (size_t)x) * 3;
    if (o + 2 < (size_t)w * (size_t)h * 3) {
        out_rgb[o + 0] = u;
        out_rgb[o + 1] = u;
        out_rgb[o + 2] = u;
    }
}

} // extern "C"
"""

# ============================================================================
#  Host helpers
# ============================================================================

def rgb_to_gray_host(rgb: np.ndarray) -> np.ndarray:
    """Convierte RGB uint8 a gris uint8 usando la misma fórmula que en C."""
    gray = (0.2126 * rgb[:,:,0] +
            0.7152 * rgb[:,:,1] +
            0.0722 * rgb[:,:,2] + 0.5).astype(np.uint8)
    return gray.reshape(-1)


def make_gauss_1d(N: int) -> np.ndarray:
    """
    Construye un kernel Gaussiano 1D normalizado de tamaño N.
    Sigma se calcula automáticamente como sigma = N / 6.0.
    """
    r = N // 2
    sigma = N / 6.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    k = np.empty(N, dtype=np.float32)
    s = 0.0
    for i in range(-r, r+1):
        v = np.exp(-(i*i) * inv2s2)
        k[i + r] = v
        s += v
    k /= s
    return k.astype(np.float32)


# ============================================================================
#  MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser("Gaussian filter (separable) in PyCUDA")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--blockX", type=int, default=5)
    parser.add_argument("--blockY", type=int, default=5)
    args = parser.parse_args()

    # -------- Pedimos solo N (sigma se calcula con N/6) --------
    while True:
        try:
            N = int(input("Ingrese N del kernel (impar >=3, ej. 9, 21, 61, 121): "))
            if N < 3:
                print("N debe ser >= 3.")
                continue
            if N % 2 == 0:
                print("N debe ser impar. Se usa N+1.")
                N += 1
            break
        except ValueError:
            print("Valor inválido, ingrese un entero.")

    # Sigma efectivo que se usará internamente
    sigma_eff = N / 6.0

    # -------- Cargar imagen y pasar a gris --------
    img = Image.open(args.input).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape

    gray = rgb_to_gray_host(arr)
    Npix = w * h

    # -------- Kernel Gauss 1D en host --------
    k1d = make_gauss_1d(N)

    # -------- Configurar bloque / grilla (usar tamaños más conservadores) --------
    blockX = min(32, max(1, args.blockX))  # Limitar a 32 para mejor coalescencia
    blockY = min(16, max(1, args.blockY))  # Limitar a 16 para evitar problemas
    
    # Asegurar que el total de threads no exceda 512 (más conservador)
    while blockX * blockY > 512:
        if blockY > blockX:
            blockY //= 2
        else:
            blockX //= 2
        if blockY < 1:
            blockY = 1
        if blockX < 1:
            blockX = 1

    block = (blockX, blockY, 1)
    grid = ((w + blockX - 1) // blockX,
            (h + blockY - 1) // blockY,
            1)

    print(f"\nImagen: {w}x{h}")
    print(f"N = {N}, sigma = {sigma_eff:.3f} (auto N/6)")
    print(f"Block = {block}, Grid = {grid}")

    # -------- Reservas en GPU y delta de memoria --------
    free0, _ = cuda.mem_get_info()

    d_u8  = cuda.mem_alloc(Npix)       # gris u8
    d_in  = cuda.mem_alloc(Npix*4)     # float
    d_tmp = cuda.mem_alloc(Npix*4)
    d_out = cuda.mem_alloc(Npix*4)
    d_k1d = cuda.mem_alloc(N*4)        # kernel 1D
    d_rgb = cuda.mem_alloc(Npix*3)     # salida RGB

    free1, _ = cuda.mem_get_info()
    delta_bytes = free0 - free1

    cuda.memcpy_htod(d_u8, gray)
    cuda.memcpy_htod(d_k1d, k1d)

    # -------- Compilar kernels --------
    dev = cuda.Device(0)
    cc_major, cc_minor = dev.compute_capability()
    arch = f"-arch=sm_{cc_major}{cc_minor}"

    mod = DynamicSourceModule(KERNEL_SRC, options=[arch])
    u8_to_f        = mod.get_function("u8_to_f")
    gauss_horiz_f  = mod.get_function("gauss_horiz_f")
    gauss_vert_f   = mod.get_function("gauss_vert_f")
    f_to_rgb_u8    = mod.get_function("f_to_rgb_u8")

    # -------- Medir tiempo solo de kernels --------
    start = cuda.Event()
    stop  = cuda.Event()
    start.record()

    u8_to_f(d_u8, d_in,
            np.int32(w), np.int32(h),
            block=block, grid=grid)
    
    # Sincronizar después de cada kernel para evitar condiciones de carrera
    cuda.Context.synchronize()

    gauss_horiz_f(d_in, d_tmp,
                  np.int32(w), np.int32(h),
                  d_k1d, np.int32(N),
                  block=block, grid=grid)
    
    cuda.Context.synchronize()

    gauss_vert_f(d_tmp, d_out,
                 np.int32(w), np.int32(h),
                 d_k1d, np.int32(N),
                 block=block, grid=grid)
    
    cuda.Context.synchronize()

    f_to_rgb_u8(d_out, d_rgb,
                np.int32(w), np.int32(h),
                block=block, grid=grid)

    stop.record()
    stop.synchronize()
    elapsed_s = stop.time_since(start) / 1000.0

    # -------- Descargar resultado y guardar --------
    h_rgb = np.empty(Npix*3, dtype=np.uint8)
    cuda.memcpy_dtoh(h_rgb, d_rgb)
    Image.fromarray(h_rgb.reshape(h, w, 3)).save(args.output)

    # -------- Reporte de memoria --------
    bytes_theory = (
        Npix +          # d_u8
        Npix*4*3 +      # d_in, d_tmp, d_out
        N*4 +           # d_k1d
        Npix*3          # d_rgb
    )

    print("\n================ PYCUDA – Gaussian (separable) ================")
    print(f"Tiempo paralelo (s): {elapsed_s:.6f}")
    print(f"Memoria usada (delta) MB: {delta_bytes/(1024*1024):.2f}")
    print(f"Memoria teórica buffers MB: {bytes_theory/(1024*1024):.2f}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
