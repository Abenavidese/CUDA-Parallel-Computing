// laplaciano_cuda.cu — Laplaciano 3x3 y LoG NxN en CUDA (tiempo y memoria GPU)

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
__device__ __forceinline__ size_t IDX(int x, int y, int w) { return (size_t)y * w + x; }

// ===== KERNELS =====

// 3x3 Laplacian clásico (8 vecinos), entrada gris u8 -> salida RGB u8 (abs y clamp)
__global__ void laplacian3x3_u8_to_rgb(const unsigned char* __restrict__ gray,
    unsigned char* __restrict__ rgb,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Pesos 8-vecinos: [-1 -1 -1; -1 8 -1; -1 -1 -1]
    int acc = 0;
    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x, 0, w - 1), clampi(y - 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y - 1, 0, h - 1), w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), y, w)];
    acc += +8 * (int)gray[IDX(x, y, w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), y, w)];

    acc += -(int)gray[IDX(clampi(x - 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x, 0, w - 1), clampi(y + 1, 0, h - 1), w)];
    acc += -(int)gray[IDX(clampi(x + 1, 0, w - 1), clampi(y + 1, 0, h - 1), w)];

    int v = abs(acc);
    if (v > 255) v = 255;
    unsigned char u = (unsigned char)v;
    size_t o = IDX(x, y, w) * 3;
    rgb[o + 0] = u; rgb[o + 1] = u; rgb[o + 2] = u;
}

// Convolución general LoG: gris u8 * K(NxN float) -> float (acumula), clamp por bordes
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
    for (int ky = -r; ky <= r; ++ky) {
        int yy = clampi(y + ky, 0, h - 1);
        int krow = (ky + r) * N;
        for (int kx = -r; kx <= r; ++kx) {
            int xx = clampi(x + kx, 0, w - 1);
            acc += (float)gray[IDX(xx, yy, w)] * K[krow + (kx + r)];
        }
    }
    out[IDX(x, y, w)] = acc; // puede ser negativo
}

// float -> RGB u8 con abs + clamp
__global__ void f_abs_to_rgb_u8(const float* __restrict__ in,
    unsigned char* __restrict__ out_rgb,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v = fabsf(in[IDX(x, y, w)]);
    if (v > 255.f) v = 255.f;
    unsigned char u = (unsigned char)(v + 0.5f);
    size_t o = IDX(x, y, w) * 3;
    out_rgb[o + 0] = u; out_rgb[o + 1] = u; out_rgb[o + 2] = u;
}

// ===== HOST HELPERS =====
static unsigned char* to_gray_host(const unsigned char* img, int w, int h, int nc) {
    size_t N = (size_t)w * h;
    unsigned char* g = (unsigned char*)malloc(N);
    if (!g) return nullptr;
    for (size_t p = 0; p < N; ++p) {
        int R = img[nc * p + 0], G = img[nc * p + 1], B = img[nc * p + 2];
        int y8 = (int)(0.2126 * R + 0.7152 * G + 0.0722 * B + 0.5);
        if (y8 < 0) y8 = 0; if (y8 > 255) y8 = 255;
        g[p] = (unsigned char)y8;
    }
    return g;
}

// LoG NxN (igual a tu CPU): sigma=N/6, corrección para suma≈0
static float* make_log_kernel(int N) {
    int c = N / 2;
    float sigma = N / 6.0f;
    float s2 = sigma * sigma;
    float s4 = s2 * s2;
    float* k = (float*)malloc(N * sizeof(float) * N);
    if (!k) return nullptr;

    double sum = 0.0;
    for (int y = -c; y <= c; ++y) {
        for (int x = -c; x <= c; ++x) {
            float r2 = (float)(x * x + y * y);
            float val = -((r2 - 2.f * s2) / s4) * expf(-r2 / (2.f * s2));
            k[(y + c) * N + (x + c)] = val;
            sum += val;
        }
    }
    float corr = (float)(sum / (double)(N * N));
    for (int i = 0; i < N * N; ++i) k[i] -= corr;
    return k;
}

int main() {
    // ====== EDITA AQUÍ (Visual Studio) ======
    const char* INPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/laplaciano/bird_9000x9000.jpg";
    const char* OUTPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/laplaciano/cuda/65x65/32x16.png";
    int  N = 65;         // 3 -> Laplaciano clásico; >3 (impar) -> LoG NxN (9,21,61)
    int  blockX = 32;    
    int  blockY = 16;    // 16x16, 32x8, 32x16 recomendados
    // =========================================

    if (N < 3) N = 3;
    if ((N % 2) == 0) { ++N; printf("[Aviso] N debe ser impar. Usando N=%d\n", N); }

    // Carga RGB y pasa a gris en host (igual que en tus otros filtros)
    int w, h, nc;
    unsigned char* img = stbi_load(INPUT_PATH, &w, &h, &nc, 0);
    if (!img) { fprintf(stderr, "No se pudo cargar %s\n", INPUT_PATH); return 1; }
    if (nc < 3) { fprintf(stderr, "Se requiere RGB(A)\n"); stbi_image_free(img); return 1; }

    size_t Npix = (size_t)w * h;
    unsigned char* h_gray = to_gray_host(img, w, h, nc);
    stbi_image_free(img);
    if (!h_gray) { fprintf(stderr, "Sin memoria para gris\n"); return 1; }

    // ===== Memoria en device y delta =====
    size_t free0, total0, free1, total1;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

    unsigned char* d_gray = nullptr, * d_rgb = nullptr;
    float* d_tmpF = nullptr, * d_K = nullptr; // d_tmpF solo para LoG
    CUDA_CHECK(cudaMalloc(&d_gray, Npix * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_rgb, Npix * 3 * sizeof(unsigned char)));

    bool useLoG = (N != 3);
    if (useLoG) {
        CUDA_CHECK(cudaMalloc(&d_tmpF, Npix * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, N * N * sizeof(float)));
    }

    CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    size_t deltaBytes = free0 - free1;

    // Copias H->D (no cuentan en tiempo)
    CUDA_CHECK(cudaMemcpy(d_gray, h_gray, Npix * sizeof(unsigned char), cudaMemcpyHostToDevice));
    float* h_K = nullptr;
    if (useLoG) {
        h_K = make_log_kernel(N);
        if (!h_K) { fprintf(stderr, "No se pudo crear kernel LoG\n"); cudaFree(d_gray); cudaFree(d_rgb); if (d_tmpF)cudaFree(d_tmpF); if (d_K)cudaFree(d_K); free(h_gray); return 1; }
        CUDA_CHECK(cudaMemcpy(d_K, h_K, N * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Bloques / grilla
    if (blockX > 1024) blockX = 1024;
    if (blockY > 1024) blockY = 1024;
    while ((long long)blockX * blockY > 1024) blockY = std::max(1, blockY / 2);

    dim3 block(blockX, blockY);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    printf("Imagen: %dx%d  N=%d  modo=%s  block=(%d,%d) grid=(%u,%u)\n",
        w, h, N, useLoG ? "LoG" : "Laplaciano3x3", block.x, block.y, grid.x, grid.y);

    // ===== Tiempo SOLO procesamiento GPU =====
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    if (!useLoG) {
        laplacian3x3_u8_to_rgb << <grid, block >> > (d_gray, d_rgb, w, h);
        CUDA_CHECK(cudaGetLastError());
    }
    else {
        conv_log_u8_to_f << <grid, block >> > (d_gray, d_K, N, d_tmpF, w, h);
        CUDA_CHECK(cudaGetLastError());
        f_abs_to_rgb_u8 << <grid, block >> > (d_tmpF, d_rgb, w, h);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seg = ms / 1000.0;

    // Descargar (no cuenta en tiempo)
    unsigned char* h_rgb = (unsigned char*)malloc(Npix * 3);
    CUDA_CHECK(cudaMemcpy(h_rgb, d_rgb, Npix * 3, cudaMemcpyDeviceToHost));

    // Guardar
    stbi_write_png(OUTPUT_PATH, w, h, 3, h_rgb, w * 3);

    // Reporte memoria
    size_t bytes_theory = 0;
    if (!useLoG) {
        // gray + rgb
        bytes_theory = Npix + Npix * 3;
    }
    else {
        // gray + tmpF + K + rgb
        bytes_theory = Npix + Npix * sizeof(float) + N * N * sizeof(float) + Npix * 3;
    }
    double memMB_delta = (double)deltaBytes / (1024.0 * 1024.0);
    double memMB_theory = (double)bytes_theory / (1024.0 * 1024.0);

    printf("\n================ CUDA – Laplaciano =================\n");
    printf("Tiempo Paralelo (s): %.6f\n", seg);
    printf("Memoria Usada (delta cudaMemGetInfo) MB: %.2f\n", memMB_delta);
    printf("Memoria Buffers (teórica) MB: %.2f\n", memMB_theory);
    printf("Salida: %s\n", OUTPUT_PATH);

    // Limpieza
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_gray); cudaFree(d_rgb);
    if (d_tmpF) cudaFree(d_tmpF);
    if (d_K)    cudaFree(d_K);
    free(h_rgb); free(h_gray); free(h_K);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
