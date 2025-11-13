
import argparse
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule

# ============================================================================
#  Rutas de entrada y salida por defecto para las imágenes
#  (cámbialas aquí si quieres probar otras imágenes por defecto)
# ============================================================================
DEFAULT_INPUT  = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Box_Blur\cuenca9000.jpg"
DEFAULT_OUTPUT = r"C:\Users\EleXc\Desktop\part-w\CUDA-Parallel-Computing\Filter_Box_Blur\blox_blur_pycuda_output.jpg"

# Número de pasadas del box blur (igual que en la versión C++)
PASSES = 3

# ============================================================================
#  Fuente del kernel CUDA para box blur separable en imágenes RGB
#   - Primer kernel: box blur horizontal, src (uint8 RGB) -> tmp (float suma RGB)
#   - Segundo kernel: box blur vertical, tmp (float) -> dst (uint8 RGB, normalizado)
#   - El efecto global es equivalente a un filtro de caja 2D de tamaño N x N
# ============================================================================
KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

// --------------------------------------------------------------------------
// Paso horizontal: srcRGB (uint8) -> tmpRGB (float) acumulando a lo largo de X
//   Cada hilo procesa un píxel (x,y)
//   El resultado es la suma de N vecinos en X, almacenado en float
// --------------------------------------------------------------------------
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ src,
                                  float* __restrict__ tmp,
                                  int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float R = 0.f, G = 0.f, B = 0.f;

    // Acumular a lo largo de la dirección X (ventana horizontal)
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

// --------------------------------------------------------------------------
// Paso vertical: tmpRGB (suma float) -> dstRGB (uint8)
//   Cada hilo procesa un píxel (x,y)
//   Acumula N vecinos en Y y normaliza por N*N
// --------------------------------------------------------------------------
__global__ void box_vert_f_to_u8(const float* __restrict__ tmp,
                                 unsigned char* __restrict__ dst,
                                 int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float invN = 1.0f / (float)N;
    float R = 0.f, G = 0.f, B = 0.f;

    // Acumular a lo largo de la dirección Y (ventana vertical)
    for (int j = -r; j <= r; ++j){
        int yy = clampi(y + j, 0, h - 1);
        size_t s = ((size_t)yy * w + x) * 3;
        R += tmp[s + 0];
        G += tmp[s + 1];
        B += tmp[s + 2];
    }

    size_t o = ((size_t)y * w + x) * 3;

    // El factor total de normalización es invN * invN (horizontal * vertical)
    int r8 = (int)(R * invN * invN + 0.5f);
    int g8 = (int)(G * invN * invN + 0.5f);
    int b8 = (int)(B * invN * invN + 0.5f);

    // Limitar al rango válido [0,255]
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
    # -------------------------------------------------------------------------
    # Analizador de argumentos para parámetros opcionales de CLI:
    #   --input  : ruta de la imagen de entrada
    #   --output : ruta de la imagen de salida
    #   --blockX : dimensión del bloque en X
    #   --blockY : dimensión del bloque en Y
    # El tamaño del kernel N se solicita interactivamente al usuario.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Box Blur separable en PyCUDA (imagen RGB).")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT,  help="Ruta de la imagen de entrada")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ruta de la imagen de salida")
    parser.add_argument("--blockX", type=int, default=32, help="Tamaño del bloque en X")
    parser.add_argument("--blockY", type=int, default=32, help="Tamaño del bloque en Y")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Solicitar al usuario el tamaño del kernel N (debe ser impar y >= 3)
    # -------------------------------------------------------------------------
    while True:
        try:
            userN = int(input("Ingrese tamaño del kernel N (impar >= 3): "))
            if userN < 3:
                print("N debe ser >= 3.")
                continue
            if userN % 2 == 0:
                print("N debe ser impar. Se incrementa en 1.")
                userN += 1
            N = userN
            break
        except ValueError:
            print("Entrada invalida. Ingrese un entero (ej. 9, 21, 65).")

    # -------------------------------------------------------------------------
    # Cargar la imagen desde disco y convertir a RGB uint8
    # -------------------------------------------------------------------------
    img = Image.open(args.input).convert("RGB")
    np_img = np.array(img, dtype=np.uint8)  # forma: (h, w, 3)
    h, w, _ = np_img.shape

    # Aplanar la imagen a un array 1D (RGB empaquetado)
    h_in = np_img.reshape(-1).copy()
    h_out = np.empty_like(h_in)

    Npix = w * h
    bytesRGB = Npix * 3

    # -------------------------------------------------------------------------
    # Configurar tamaños de bloque y grid para CUDA
    # -------------------------------------------------------------------------
    blockX = max(1, args.blockX)
    blockY = max(1, args.blockY)

    # Forzar un máximo de 1024 hilos por bloque
    while blockX * blockY > 1024:
        blockY = max(1, blockY // 2)

    block = (blockX, blockY, 1)
    grid  = ((w + blockX - 1) // blockX,
             (h + blockY - 1) // blockY,
             1)

    print(f"Imagen: {w}x{h}")
    print(f"Kernel N = {N}")
    print(f"Block  = {block}")
    print(f"Grid   = {grid}")
    print(f"PASSES = {PASSES}\n")

    # -------------------------------------------------------------------------
    # Reservar memoria en la GPU para los buffers de entrada, salida e intermedios
    # d_in  : imagen de entrada (uint8 RGB)
    # d_out : imagen de salida (uint8 RGB)
    # d_tmp : buffer temporal en float RGB (sumas)
    # -------------------------------------------------------------------------
    free0, _ = cuda.mem_get_info()

    d_in  = cuda.mem_alloc(bytesRGB)
    d_out = cuda.mem_alloc(bytesRGB)
    d_tmp = cuda.mem_alloc(bytesRGB * 4)  # float = 4 bytes

    free1, _ = cuda.mem_get_info()
    delta_bytes = free0 - free1

    # Copiar la imagen del host al dispositivo
    cuda.memcpy_htod(d_in, h_in)

    # -------------------------------------------------------------------------
    # Compilar los kernels CUDA con una bandera de arquitectura que coincida con el dispositivo
    # -------------------------------------------------------------------------
    dev = cuda.Device(0)
    cc_major, cc_minor = dev.compute_capability()
    arch_flag = f"-arch=sm_{cc_major}{cc_minor}"

    mod = DynamicSourceModule(KERNEL_SRC, options=[arch_flag])
    box_horiz = mod.get_function("box_horiz_u8_to_f")
    box_vert  = mod.get_function("box_vert_f_to_u8")

    # -------------------------------------------------------------------------
    # Ejecutar solo el procesamiento en GPU (kernels) y medir su tiempo de ejecución
    # -------------------------------------------------------------------------
    start = cuda.Event()
    stop  = cuda.Event()
    start.record()

    for p in range(PASSES):
        # Desenfoque horizontal: uint8 -> float
        box_horiz(d_in, d_tmp,
                  np.int32(w), np.int32(h), np.int32(N),
                  block=block, grid=grid)

        # Desenfoque vertical: float -> uint8
        box_vert(d_tmp, d_out,
                 np.int32(w), np.int32(h), np.int32(N),
                 block=block, grid=grid)

        # Intercambiar los buffers de entrada y salida para la siguiente pasada
        d_in, d_out = d_out, d_in

    stop.record()
    stop.synchronize()
    elapsed_s = stop.time_since(start) / 1000.0

    # -------------------------------------------------------------------------
    # Copiar el resultado de vuelta al host y guardar la imagen de salida
    # -------------------------------------------------------------------------
    cuda.memcpy_dtoh(h_out, d_in)
    out_img = h_out.reshape(h, w, 3)
    Image.fromarray(out_img, mode="RGB").save(args.output)

    # -------------------------------------------------------------------------
    # Calcular e imprimir estadísticas de uso de memoria
    # -------------------------------------------------------------------------
    bytes_buffers = bytesRGB + bytesRGB + (bytesRGB * 4)  # in + out + tmp(float3)
    memMB_delta  = delta_bytes   / (1024.0 * 1024.0)
    memMB_theory = bytes_buffers / (1024.0 * 1024.0)

    print("================ PYCUDA – Box Blur (separable) ================")
    print(f"Tiempo paralelo (solo kernels) (s): {elapsed_s:.6f}")
    print(f"Memoria usada (delta cuda.mem_get_info) MB: {memMB_delta:.2f}")
    print(f"Memoria buffers (teorica) MB:             {memMB_theory:.2f}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
