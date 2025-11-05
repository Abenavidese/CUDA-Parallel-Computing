// prewitt_cuda.cu — Prewitt (N general) en CUDA con memoria y tiempo de kernels

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s @%s:%d -> %s\n", #x,__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(EXIT_FAILURE);} }while(0)

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ size_t IDX(int x, int y, int w) { return (size_t)y * w + x; }

// ===== Kernels =====

// RGB(A) -> GRAY (host lo hace; este es por si lo necesitas en device)
__global__ void rgb_to_gray(const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int w, int h, int nc)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t p = (size_t)y * w + x;
    int R = rgb[nc * p + 0], G = rgb[nc * p + 1], B = rgb[nc * p + 2];
    int y8 = (int)(0.299f * R + 0.587f * G + 0.114f * B + 0.5f);
    if (y8 < 0) y8 = 0; if (y8 > 255) y8 = 255;
    gray[p] = (unsigned char)y8;
}

// Pequeño pre-blur 3x3 (replicate borders) EN DEVICE
__global__ void blur3x3_u8_kernel(const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int acc = 0;
    for (int j = -1; j <= 1; ++j) {
        int yy = clampi(y + j, 0, h - 1);
        for (int i = -1; i <= 1; ++i) {
            int xx = clampi(x + i, 0, w - 1);
            acc += in[IDX(xx, yy, w)];
        }
    }
    out[IDX(x, y, w)] = (unsigned char)(acc / 9);
}

// 1) Suma vertical de caja: gray(u8) -> V(float)
__global__ void box_vert_u8_to_f(const unsigned char* __restrict__ gray,
    float* __restrict__ V,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int j = -r; j <= r; ++j) {
        int yy = clampi(y + j, 0, h - 1);
        acc += gray[IDX(x, yy, w)];
    }
    V[IDX(x, y, w)] = acc;
}

// 2) gx desde V con pesos sx = sign(i)
__global__ void prewitt_x_from_V(const float* __restrict__ V,
    float* __restrict__ gx,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int i = -r; i <= r; ++i) {
        int xx = clampi(x + i, 0, w - 1);
        int sx = (i < 0 ? -1 : (i > 0 ? +1 : 0));
        acc += V[IDX(xx, y, w)] * (float)sx;
    }
    gx[IDX(x, y, w)] = acc;
}

// 3) Suma horizontal de caja: gray(u8) -> H(float)
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ gray,
    float* __restrict__ H,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int i = -r; i <= r; ++i) {
        int xx = clampi(x + i, 0, w - 1);
        acc += gray[IDX(xx, y, w)];
    }
    H[IDX(x, y, w)] = acc;
}

// 4) gy desde H con pesos sy = sign(j)
__global__ void prewitt_y_from_H(const float* __restrict__ H,
    float* __restrict__ gy,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int j = -r; j <= r; ++j) {
        int yy = clampi(y + j, 0, h - 1);
        int sy = (j < 0 ? -1 : (j > 0 ? +1 : 0));
        acc += H[IDX(x, yy, w)] * (float)sy;
    }
    gy[IDX(x, y, w)] = acc;
}

// 5) Combinar |gx|+|gy|, gain y normalización por N*N -> RGB gris
__global__ void combine_mag_to_rgb(const float* __restrict__ gx,
    const float* __restrict__ gy,
    unsigned char* __restrict__ rgb,
    int w, int h, int N, float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float mag = fabsf(gx[IDX(x, y, w)]) + fabsf(gy[IDX(x, y, w)]);
    float v = (mag * gain) / (float)(N * (long long)N);
    if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
    unsigned char u = (unsigned char)(v + 0.5f);
    size_t o = (size_t)IDX(x, y, w) * 3;
    rgb[o + 0] = u; rgb[o + 1] = u; rgb[o + 2] = u;
}

// ===== Host helpers =====
static unsigned char* to_gray_host(const unsigned char* img, int w, int h, int nc) {
    size_t N = (size_t)w * h;
    unsigned char* g = (unsigned char*)malloc(N);
    if (!g) return nullptr;
    for (size_t p = 0; p < N; ++p) {
        int R = img[nc * p + 0], G = img[nc * p + 1], B = img[nc * p + 2];
        int y8 = (int)(0.299 * R + 0.587 * G + 0.114 * B + 0.5);
        if (y8 < 0) y8 = 0; if (y8 > 255) y8 = 255;
        g[p] = (unsigned char)y8;
    }
    return g;
}

static void stretch_host_u8_inplace(unsigned char* g, int w, int h) {
    size_t N = (size_t)w * h;
    int mx = 0; for (size_t i = 0; i < N; ++i) if (g[i] > mx) mx = g[i];
    if (mx == 0) return;
    for (size_t i = 0; i < N; ++i) {
        int ng = (int)std::lround((g[i] * 255.0) / mx);
        if (ng > 255) ng = 255; g[i] = (unsigned char)ng;
    }
}

int main() {
    // ====== EDITA AQUÍ (estilo VS) ======
    const char* INPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/prewitt/tajmahal9000.jpg";
    const char* OUTPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/prewitt/cuda/65x65/32x16.png";
    int   N_KERNEL = 65;       // impar >=3 (9,21,61)
    float GAIN = 8.0f;     // brillo de bordes
    int   PRE_BLUR_3x3 = 1;        // 1=on, 0=off (como en tu C)
    int   CONTRAST_STRETCH = 1;        // 1=on (lo haré en host)
    int   blockX = 32;       
    int   blockY = 16;       // (recom: 16x16, 32x8, 32x16)
    // ====================================

    if (N_KERNEL < 3 || (N_KERNEL % 2) == 0) { N_KERNEL = std::max(3, N_KERNEL | 1); }

    // Carga imagen
    int w, h, nc;
    unsigned char* img = stbi_load(INPUT_PATH, &w, &h, &nc, 0);
    if (!img) { fprintf(stderr, "No se pudo cargar %s\n", INPUT_PATH); return 1; }
    if (nc < 3) { fprintf(stderr, "Se requiere RGB(A)\n"); stbi_image_free(img); return 1; }

    // Gris en host (rápido y sencillo)
    unsigned char* h_gray = to_gray_host(img, w, h, nc);
    stbi_image_free(img);
    if (!h_gray) { fprintf(stderr, "Sin memoria para gris\n"); return 1; }

    // Reservas y medición de memoria
    size_t free0, total0, free1, total1;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

    unsigned char* d_gray = nullptr, * d_gray2 = nullptr, * d_rgb = nullptr;
    float* d_V = nullptr, * d_H = nullptr, * d_gx = nullptr, * d_gy = nullptr;

    size_t Npix = (size_t)w * h, bytesGray = Npix, bytesRGB = Npix * 3;
    CUDA_CHECK(cudaMalloc(&d_gray, bytesGray));
    CUDA_CHECK(cudaMalloc(&d_gray2, bytesGray)); // para pre-blur (ping-pong pequeño)
    CUDA_CHECK(cudaMalloc(&d_V, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb, bytesRGB));

    CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    size_t deltaBytes = free0 - free1;

    // Subir gris
    CUDA_CHECK(cudaMemcpy(d_gray, h_gray, bytesGray, cudaMemcpyHostToDevice));

    // Configurar bloques y grilla
    if (blockX > 1024) blockX = 1024;
    if (blockY > 1024) blockY = 1024;
    while ((long long)blockX * blockY > 1024) blockY = std::max(1, blockY / 2);

    dim3 block(blockX, blockY);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    printf("Imagen: %dx%d  N=%d  gain=%.2f  block=(%d,%d) grid=(%u,%u)\n",
        w, h, N_KERNEL, GAIN, block.x, block.y, grid.x, grid.y);

    // PRE-BLUR 3x3 (se hace ANTES del cronómetro para empatar con tu C)
    if (PRE_BLUR_3x3) {
        blur3x3_u8_kernel << <grid, block >> > (d_gray, d_gray2, w, h);
        CUDA_CHECK(cudaGetLastError());
        // Swap -> d_gray queda “filtrada”
        unsigned char* tmp = d_gray; d_gray = d_gray2; d_gray2 = tmp;
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ===== Tiempo SOLO procesamiento (kernels Prewitt) =====
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // gx vía V (vertical box, luego pesos sx en X)
    box_vert_u8_to_f << <grid, block >> > (d_gray, d_V, w, h, N_KERNEL);          CUDA_CHECK(cudaGetLastError());
    prewitt_x_from_V << <grid, block >> > (d_V, d_gx, w, h, N_KERNEL);             CUDA_CHECK(cudaGetLastError());
    // gy vía H (horizontal box, luego pesos sy en Y)
    box_horiz_u8_to_f << <grid, block >> > (d_gray, d_H, w, h, N_KERNEL);          CUDA_CHECK(cudaGetLastError());
    prewitt_y_from_H << <grid, block >> > (d_H, d_gy, w, h, N_KERNEL);             CUDA_CHECK(cudaGetLastError());
    // combinar a RGB gris
    combine_mag_to_rgb << <grid, block >> > (d_gx, d_gy, d_rgb, w, h, N_KERNEL, GAIN); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seg = ms / 1000.0;

    // Descargar (no cuenta en tiempo de procesamiento)
    unsigned char* h_rgb = (unsigned char*)malloc(bytesRGB);
    CUDA_CHECK(cudaMemcpy(h_rgb, d_rgb, bytesRGB, cudaMemcpyDeviceToHost));

    // Contrast stretch en host (fuera del cronómetro para mantener “solo kernels”)
    if (CONTRAST_STRETCH) {
        // aplicar a canal gris y volver a RGB
        unsigned char* h_mag = (unsigned char*)malloc(bytesGray);
        for (size_t p = 0; p < Npix; ++p) h_mag[p] = h_rgb[3 * p]; // cualquier canal
        stretch_host_u8_inplace(h_mag, w, h);
        for (size_t p = 0; p < Npix; ++p) {
            unsigned char u = h_mag[p];
            h_rgb[3 * p + 0] = u; h_rgb[3 * p + 1] = u; h_rgb[3 * p + 2] = u;
        }
        free(h_mag);
    }

    // Guardar
    stbi_write_png(OUTPUT_PATH, w, h, 3, h_rgb, w * 3);

    // Reporte de memoria
    // Teórico aproximado: gray + gray2 + V + H + gx + gy + rgb
    size_t bytes_theory = bytesGray + bytesGray + Npix * sizeof(float) * 4 + bytesRGB;
    double memMB_delta = (double)deltaBytes / (1024.0 * 1024.0);
    double memMB_theory = (double)bytes_theory / (1024.0 * 1024.0);

    printf("\n================ CUDA – Prewitt (N general, separable) ================\n");
    printf("Tiempo Paralelo (s): %.6f\n", seg);
    printf("Memoria Usada (delta cudaMemGetInfo) MB: %.2f\n", memMB_delta);
    printf("Memoria Buffers (teórica) MB: %.2f\n", memMB_theory);
    printf("Salida: %s\n", OUTPUT_PATH);

    // Limpieza
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_gray); cudaFree(d_gray2);
    cudaFree(d_V); cudaFree(d_H); cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_rgb);
    free(h_gray); free(h_rgb);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
