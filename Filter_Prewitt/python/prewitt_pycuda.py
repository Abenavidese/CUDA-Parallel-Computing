import argparse
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule

# =============================================================================
#  RUTAS POR DEFECTO (entrada y salida)
#  Puedes cambiarlas aquí directamente si no quieres pasar parámetros por CLI.
# =============================================================================
DEFAULT_INPUT  = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Prewitt\tajmahal6000.jpg"
DEFAULT_OUTPUT = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Prewitt\prewitt_pycuda_output.jpg"

# =============================================================================
#  Código CUDA del filtro de Prewitt separable para N general.
#  - blur3x3_u8_kernel      : pre-suavizado 3x3 en escala de grises
#  - box_vert_u8_to_f       : suma tipo "box filter" en dirección vertical
#  - prewitt_x_from_V       : gradiente gx a partir de las sumas verticales
#  - box_horiz_u8_to_f      : suma tipo "box filter" en horizontal
#  - prewitt_y_from_H       : gradiente gy a partir de las sumas horizontales
#  - combine_mag_to_rgb     : |gx| + |gy|, normalización y salida en RGB gris
# =============================================================================
KERNEL_SRC = r"""
extern "C" {

// Limita un entero v al rango [lo, hi].
__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// Índice lineal (y*w + x) para imagen 2D almacenada en 1D.
__device__ __forceinline__ size_t IDX(int x, int y, int w){
    return (size_t)y * w + x;
}

// ============================================================================
//  (1) Suavizado previo 3x3 en escala de grises
//     - in  : imagen gris uint8 de entrada
//     - out : imagen gris uint8 suavizada
// ============================================================================
__global__ void blur3x3_u8_kernel(const unsigned char* __restrict__ in,
                                  unsigned char* __restrict__ out,
                                  int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int acc = 0;
    // Ventana 3x3 centrada en (x,y) con bordes replicados
    for (int j = -1; j <= 1; ++j){
        int yy = clampi(y + j, 0, h - 1);
        for (int i = -1; i <= 1; ++i){
            int xx = clampi(x + i, 0, w - 1);
            acc += in[IDX(xx, yy, w)];
        }
    }
    out[IDX(x,y,w)] = (unsigned char)(acc / 9);
}

// ============================================================================
//  (2) Suma vertical tipo "box filter"
//     - gray : imagen gris uint8
//     - V    : salida float con la suma vertical en ventana de tamaño N
// ============================================================================
__global__ void box_vert_u8_to_f(const unsigned char* __restrict__ gray,
                                 float* __restrict__ V,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        acc += gray[IDX(x,yy,w)];
    }

    V[IDX(x,y,w)] = acc;
}

// ============================================================================
//  (3) Cálculo de gx a partir de V (suma vertical)
//     - Usa pesos sx = sign(i) en la dirección X
// ============================================================================
__global__ void prewitt_x_from_V(const float* __restrict__ V,
                                 float* __restrict__ gx,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        int sx = (i < 0 ? -1 : (i > 0 ? +1 : 0));  // -1, 0, +1
        acc += V[IDX(xx,y,w)] * (float)sx;
    }

    gx[IDX(x,y,w)] = acc;
}

// ============================================================================
//  (4) Suma horizontal tipo "box filter"
//     - gray : imagen gris uint8
//     - H    : salida float con la suma horizontal en ventana de tamaño N
// ============================================================================
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ gray,
                                  float* __restrict__ H,
                                  int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int i = -r; i <= r; ++i){
        int xx = clampi(x + i, 0, w - 1);
        acc += gray[IDX(xx,y,w)];
    }

    H[IDX(x,y,w)] = acc;
}

// ============================================================================
//  (5) Cálculo de gy a partir de H (suma horizontal)
//     - Usa pesos sy = sign(j) en la dirección Y
// ============================================================================
__global__ void prewitt_y_from_H(const float* __restrict__ H,
                                 float* __restrict__ gy,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int r = N / 2;
    float acc = 0.f;

    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        int sy = (j < 0 ? -1 : (j > 0 ? +1 : 0));  // -1, 0, +1
        acc += H[IDX(x,yy,w)] * (float)sy;
    }

    gy[IDX(x,y,w)] = acc;
}

// ============================================================================
//  (6) Combinación final |gx| + |gy| y normalización a [0,255]
//     - gain permite realzar el contraste de los bordes detectados
// ============================================================================
__global__ void combine_mag_to_rgb(const float* __restrict__ gx,
                                   const float* __restrict__ gy,
                                   unsigned char* __restrict__ rgb,
                                   int w, int h, int N, float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float mag = fabsf(gx[IDX(x,y,w)]) + fabsf(gy[IDX(x,y,w)]);
    float v = (mag * gain) / (float)(N * (long long)N);

    if (v < 0.f) v = 0.f;
    else if (v > 255.f) v = 255.f;

    unsigned char u = (unsigned char)(v + 0.5f);

    size_t o = (size_t)IDX(x,y,w) * 3;
    rgb[o+0] = u;
    rgb[o+1] = u;
    rgb[o+2] = u;
}

} // extern C
"""

def main():
    # =========================================================================
    #  Lectura de parámetros desde CLI
    #    - input/output: rutas de imagen (con valores por defecto)
    #    - gain       : factor de realce de la magnitud del gradiente
    #    - blockX/Y   : tamaño del bloque CUDA
    #    - preblur    : aplicar (1) blur3x3 antes del Prewitt (1=Sí, 0=No)
    #  El tamaño del kernel N se pedirá de forma interactiva.
    # =========================================================================
    parser = argparse.ArgumentParser("Prewitt separable PyCUDA")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--gain",   type=float, default=8.0)
    parser.add_argument("--blockX", type=int, default=32)
    parser.add_argument("--blockY", type=int, default=16)
    parser.add_argument("--preblur", type=int, default=1)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    #  Solicitar tamaño del kernel N por consola (impar >= 3).
    #  Esto ignora cualquier valor de N que se haya intentado pasar por CLI.
    # -------------------------------------------------------------------------
    while True:
        try:
            user_N = int(input("Ingrese el tamaño del kernel Prewitt (impar >= 3, ej. 9, 21, 65): "))
            if user_N < 3:
                print("  [!] N debe ser al menos 3.")
                continue
            if user_N % 2 == 0:
                print("  [!] N debe ser impar. Sumando 1 al valor ingresado.")
                user_N += 1
            N = user_N
            break
        except ValueError:
            print("  [!] Entrada no válida. Ingrese un entero, por ejemplo 9, 21, 65.")
    # -------------------------------------------------------------------------

    # ==== Carga de imagen en host y conversión a escala de grises ====
    img = Image.open(args.input).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # [h, w, 3]
    h, w, _ = arr.shape

    # Conversión a una sola banda (gris) usando luminancia estándar
    gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] + 0.5).astype(np.uint8)
    gray = gray.reshape(-1)  # vector 1D de tamaño w*h

    Npix = w * h
    bytesGray = Npix
    bytesRGB  = Npix * 3

    # ==== Configuración de bloque y grilla en CUDA ====
    blockX = max(1, args.blockX)
    blockY = max(1, args.blockY)
    # Aseguramos que no exceda 1024 hilos por bloque
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
    print(f"Kernel N: {N}")
    print(f"Bloque CUDA: {blockX}x{blockY}  -> hilos/bloque={blockX*blockY}")
    print(f"Grilla CUDA: {grid[0]}x{grid[1]} bloques\n")

    # ==== Reserva de memoria en GPU ====
    d_gray  = cuda.mem_alloc(bytesGray)
    d_gray2 = cuda.mem_alloc(bytesGray)       # buffer auxiliar para blur 3x3
    d_V  = cuda.mem_alloc(Npix * 4)          # float (4 bytes)
    d_H  = cuda.mem_alloc(Npix * 4)
    d_gx = cuda.mem_alloc(Npix * 4)
    d_gy = cuda.mem_alloc(Npix * 4)
    d_rgb = cuda.mem_alloc(bytesRGB)

    # Copiamos la imagen gris al dispositivo
    cuda.memcpy_htod(d_gray, gray)

    # ==== Compilación del código CUDA (NVRTC a través de PyCUDA) ====
    mod = DynamicSourceModule(KERNEL_SRC)
    blur3 = mod.get_function("blur3x3_u8_kernel")
    boxV  = mod.get_function("box_vert_u8_to_f")
    gxF   = mod.get_function("prewitt_x_from_V")
    boxH  = mod.get_function("box_horiz_u8_to_f")
    gyF   = mod.get_function("prewitt_y_from_H")
    comb  = mod.get_function("combine_mag_to_rgb")

    # ==== Pre-blur 3x3 opcional (igual que en tu versión en C/CUDA) ====
    if args.preblur:
        blur3(d_gray, d_gray2,
              np.int32(w), np.int32(h),
              block=block, grid=grid)
        # Intercambiamos buffers para que d_gray contenga la imagen suavizada
        d_gray, d_gray2 = d_gray2, d_gray

    # ==== Medición de tiempo SOLO de los kernels de Prewitt ====
    start = cuda.Event()
    stop  = cuda.Event()
    start.record()

    # gx a partir de suma vertical
    boxV(d_gray, d_V,
         np.int32(w), np.int32(h), np.int32(N),
         block=block, grid=grid)
    gxF(d_V, d_gx,
        np.int32(w), np.int32(h), np.int32(N),
        block=block, grid=grid)

    # gy a partir de suma horizontal
    boxH(d_gray, d_H,
         np.int32(w), np.int32(h), np.int32(N),
         block=block, grid=grid)
    gyF(d_H, d_gy,
        np.int32(w), np.int32(h), np.int32(N),
        block=block, grid=grid)

    # combinación |gx| + |gy| y normalización
    comb(d_gx, d_gy, d_rgb,
         np.int32(w), np.int32(h),
         np.int32(N), np.float32(args.gain),
         block=block, grid=grid)

    stop.record()
    stop.synchronize()
    elapsed_s = stop.time_since(start) / 1000.0

    # ==== Copia del resultado al host y guardado de imagen ====
    out = np.empty(bytesRGB, dtype=np.uint8)
    cuda.memcpy_dtoh(out, d_rgb)

    out_img = out.reshape(h, w, 3)
    Image.fromarray(out_img).save(args.output)

    print("========== PYCUDA – Prewitt ==========")
    print(f"Tiempo paralelo (solo kernels)  : {elapsed_s:.6f} s")
    print(f"Imagen de salida guardada en   : {args.output}")

if __name__ == "__main__":
    main()
