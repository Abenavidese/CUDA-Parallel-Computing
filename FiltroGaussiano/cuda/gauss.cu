// gauss_cuda.cu — Gauss separable en CUDA (gris->RGB gris), tiempo y memoria GPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s @%s:%d -> %s\n", #x,__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(EXIT_FAILURE);} }while(0)

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ===== Kernels =====
// gris uint8 -> float (0..255)
__global__ void u8_to_f(const unsigned char* __restrict__ in_u8,
    float* __restrict__ out_f, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    out_f[y * w + x] = (float)in_u8[y * w + x];
}

// Horizontal: gauss 1D (float -> float)
__global__ void gauss_horiz_f(const float* __restrict__ in,
    float* __restrict__ tmp, int w, int h,
    const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int i = -r; i <= r; ++i) {
        int xx = clampi(x + i, 0, w - 1);
        acc += in[y * w + xx] * k1d[i + r];
    }
    tmp[y * w + x] = acc;
}

// Vertical: gauss 1D (float -> float)
__global__ void gauss_vert_f(const float* __restrict__ tmp,
    float* __restrict__ out, int w, int h,
    const float* __restrict__ k1d, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int r = N / 2;
    float acc = 0.f;
    for (int j = -r; j <= r; ++j) {
        int yy = clampi(y + j, 0, h - 1);
        acc += tmp[yy * w + x] * k1d[j + r];
    }
    out[y * w + x] = acc;
}

// float -> RGB uint8 (replicando gris a 3 canales)
__global__ void f_to_rgb_u8(const float* __restrict__ in_f,
    unsigned char* __restrict__ out_rgb,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v = in_f[y * w + x];
    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
    unsigned char u = (unsigned char)(v + 0.5f);
    size_t o = ((size_t)y * w + x) * 3;
    out_rgb[o + 0] = u; out_rgb[o + 1] = u; out_rgb[o + 2] = u;
}

// ===== Host helpers =====
static float* make_gauss_1d(int N, float sigma) {
    int r = N / 2;
    float* k = (float*)malloc(N * sizeof(float));
    if (!k) return nullptr;
    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) {
        float v = expf(-(i * i) * inv2s2);
        k[i + r] = v;
        sum += v;
    }
    for (int i = 0; i < N; ++i) k[i] /= sum;
    return k;
}

static inline unsigned char to_gray_rgb(unsigned char r, unsigned char g, unsigned char b) {
    // misma ponderación “luminancia” que usas en CPU (sRGB)
    float v = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
    return (unsigned char)(v + 0.5f);
}

int main() {
    // ====== EDITA AQUÍ (o usa Propiedades>Depuración>Argumentos si prefieres) ======
    const char* INPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/gaussiano/7.jpg";
    const char* OUTPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/gaussiano/cuda/61x61/32x16.png";
    int N = 61;   // 9, 21, 65 (impar >=3)
    float sigma = -1.f; // si <0 => N/6.0
    int blockX = 32;   // cambia aquí para VS
    int blockY = 16;   // 16x16, 32x8, 32x16 recomendados
    // =================================================================================

    if (N < 3 || (N % 2) == 0) { N = std::max(3, N | 1); printf("[Aviso] N impar, usando %d\n", N); }
    if (sigma <= 0.f) sigma = N / 6.0f;

    // Cargar RGB y convertir a gris en host (para mantener exactitud visual como tu CPU)
    int w, h, nc;
    unsigned char* img = stbi_load(INPUT_PATH, &w, &h, &nc, 0);
    if (!img) { fprintf(stderr, "No se pudo cargar %s\n", INPUT_PATH); return 1; }
    if (nc < 3) { fprintf(stderr, "Se espera RGB(A)\n"); stbi_image_free(img); return 1; }

    size_t Npix = (size_t)w * h;
    unsigned char* h_gray = (unsigned char*)malloc(Npix);
    if (!h_gray) { fprintf(stderr, "Sin memoria h_gray\n"); stbi_image_free(img); return 1; }
    for (size_t p = 0; p < Npix; ++p) {
        unsigned char r = img[nc * p + 0];
        unsigned char g = img[nc * p + 1];
        unsigned char b = img[nc * p + 2];
        h_gray[p] = to_gray_rgb(r, g, b);
    }
    stbi_image_free(img);

    // Kernel Gauss 1D en host
    float* h_k1d = make_gauss_1d(N, sigma);
    if (!h_k1d) { fprintf(stderr, "No se pudo crear kernel 1D\n"); free(h_gray); return 1; }

    // ===== Reservas en device y medición de memoria =====
    size_t free0, total0, free1, total1;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

    unsigned char* d_u8 = nullptr, * d_rgb = nullptr;
    float* d_in = nullptr, * d_tmp = nullptr, * d_out = nullptr, * d_k1d = nullptr;

    CUDA_CHECK(cudaMalloc(&d_u8, Npix * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_in, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, Npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k1d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb, Npix * 3 * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    size_t deltaBytes = free0 - free1;

    // H->D (no cronometra)
    CUDA_CHECK(cudaMemcpy(d_u8, h_gray, Npix * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k1d, h_k1d, N * sizeof(float), cudaMemcpyHostToDevice));

    // Bloques / grilla
    // Validar bloque (máx 1024 hilos por bloque)
    if (blockX > 1024) blockX = 1024;
    if (blockY > 1024) blockY = 1024;
    while ((long long)blockX * blockY > 1024) blockY = std::max(1, blockY / 2);

    dim3 block(blockX, blockY);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    printf("Imagen: %dx%d  N=%d  sigma=%.3f  block=(%d,%d) grid=(%u,%u)\n",
        w, h, N, sigma, block.x, block.y, grid.x, grid.y);

    // ===== Tiempo SOLO procesamiento (kernels) =====
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    u8_to_f << <grid, block >> > (d_u8, d_in, w, h);               CUDA_CHECK(cudaGetLastError());
    gauss_horiz_f << <grid, block >> > (d_in, d_tmp, w, h, d_k1d, N);  CUDA_CHECK(cudaGetLastError());
    gauss_vert_f << <grid, block >> > (d_tmp, d_out, w, h, d_k1d, N);  CUDA_CHECK(cudaGetLastError());
    f_to_rgb_u8 << <grid, block >> > (d_out, d_rgb, w, h);         CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seg = ms / 1000.0;

    // D->H (no cronometra)
    unsigned char* h_rgb = (unsigned char*)malloc(Npix * 3);
    CUDA_CHECK(cudaMemcpy(h_rgb, d_rgb, Npix * 3, cudaMemcpyDeviceToHost));

    // Guardar
    stbi_write_png(OUTPUT_PATH, w, h, 3, h_rgb, w * 3);

    // Reporte
    // Teórico por buffers principales: d_u8 + d_in + d_tmp + d_out + d_k1d + d_rgb
    size_t bytes_theory = (Npix * sizeof(unsigned char)) + (Npix * sizeof(float)) * 3 +
        (N * sizeof(float)) + (Npix * 3 * sizeof(unsigned char));
    double memMB_delta = (double)deltaBytes / (1024.0 * 1024.0);
    double memMB_theory = (double)bytes_theory / (1024.0 * 1024.0);

    printf("\n================ CUDA – Gauss (separable) ================\n");
    printf("Tiempo Paralelo (s): %.6f\n", seg);
    printf("Memoria Usada (delta cudaMemGetInfo) MB: %.2f\n", memMB_delta);
    printf("Memoria Buffers (teórica) MB: %.2f\n", memMB_theory);
    printf("Salida: %s\n", OUTPUT_PATH);

    // Limpieza
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_u8); cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
    cudaFree(d_k1d); cudaFree(d_rgb);
    free(h_gray); free(h_k1d); free(h_rgb);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
