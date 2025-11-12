// boxblur_cuda.cu
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

// ----------------- Helpers -----------------
#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", \
                #stmt, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ----------------- Kernels (separable box) -----------------
// Horizontal: srcRGB (uint8) -> tmpRGB (float) promediando en X
__global__ void box_horiz_u8_to_f(const unsigned char* __restrict__ src,
    float* __restrict__ tmp,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float invN = 1.0f / N;

    // Acumular por canal en ventana horizontal
    float R = 0.f, G = 0.f, B = 0.f;
    for (int i = -r; i <= r; ++i) {
        int xx = clampi(x + i, 0, w - 1);
        size_t s = ((size_t)y * w + xx) * 3;
        R += src[s + 0];
        G += src[s + 1];
        B += src[s + 2];
    }

    size_t o = ((size_t)y * w + x) * 3;
    tmp[o + 0] = R * invN;
    tmp[o + 1] = G * invN;
    tmp[o + 2] = B * invN;
}

// Vertical: tmpRGB (float) -> dstRGB (uint8) promediando en Y
__global__ void box_vert_f_to_u8(const float* __restrict__ tmp,
    unsigned char* __restrict__ dst,
    int w, int h, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    if (x >= w || y >= h) return;

    int r = N / 2;
    float invN = 1.0f / N;

    float R = 0.f, G = 0.f, B = 0.f;
    for (int j = -r; j <= r; ++j) {
        int yy = clampi(y + j, 0, h - 1);
        size_t s = ((size_t)yy * w + x) * 3;
        R += tmp[s + 0];
        G += tmp[s + 1];
        B += tmp[s + 2];
    }

    size_t o = ((size_t)y * w + x) * 3;
    int r8 = (int)(R * invN + 0.5f);
    int g8 = (int)(G * invN + 0.5f);
    int b8 = (int)(B * invN + 0.5f);
    r8 = r8 < 0 ? 0 : (r8 > 255 ? 255 : r8);
    g8 = g8 < 0 ? 0 : (g8 > 255 ? 255 : g8);
    b8 = b8 < 0 ? 0 : (b8 > 255 ? 255 : b8);
    dst[o + 0] = (unsigned char)r8;
    dst[o + 1] = (unsigned char)g8;
    dst[o + 2] = (unsigned char)b8;
}

// ----------------- Main -----------------
int main(int argc, char** argv) {
    // ===== Parámetros =====
    const char* INPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/box_blur/cuenca9000.jpg";   // cámbialo si quieres
    const char* OUTPUT_PATH = "C:/Users/EleXc/OneDrive/Desktop/pruebas_filtros/box_blur/32x16.png"; // salida
    int N = 65;        // 9, 21, 65 (impar >=3)
    int PASSES = 3;    // igual que tu C secuencial
    int blockX = 32;   // permitir variar bloques por CLI
    int blockY = 16;

    // Uso: exe [N] [PASSES] [blockX] [blockY]
    if (argc >= 2) N = std::max(3, atoi(argv[1]));
    if (argc >= 3) PASSES = std::max(1, atoi(argv[2]));
    if (argc >= 4) blockX = std::max(1, atoi(argv[3]));
    if (argc >= 5) blockY = std::max(1, atoi(argv[4]));
    if ((N % 2) == 0) { ++N; printf("[Aviso] N debe ser impar. Usando N=%d\n", N); }
    if (N < 3) { N = 3; }

    // Validar bloque según HW (máx 1024 hilos por bloque)
    if (blockX > 1024) blockX = 1024;
    if (blockY > 1024) blockY = 1024;
    while ((long long)blockX * blockY > 1024) {
        blockY = std::max(1, blockY / 2);
    }

    // ===== Cargar imagen (host) =====
    int w, h, nc;
    unsigned char* img = stbi_load(INPUT_PATH, &w, &h, &nc, 0);
    if (!img) {
        fprintf(stderr, "No se pudo cargar %s\n", INPUT_PATH);
        return 1;
    }
    if (nc < 3) {
        fprintf(stderr, "Se espera imagen con >=3 canales\n");
        stbi_image_free(img);
        return 1;
    }

    size_t Npix = (size_t)w * h;
    size_t bytesRGB = Npix * 3;

    // Preparar buffer RGB contiguo (ignorando alfa si existe)
    unsigned char* h_in = (unsigned char*)malloc(bytesRGB);
    unsigned char* h_out = (unsigned char*)malloc(bytesRGB);
    if (!h_in || !h_out) {
        fprintf(stderr, "Sin memoria host\n");
        stbi_image_free(img);
        free(h_in); free(h_out);
        return 1;
    }
    for (size_t p = 0; p < Npix; ++p) {
        h_in[3 * p + 0] = img[nc * p + 0];
        h_in[3 * p + 1] = img[nc * p + 1];
        h_in[3 * p + 2] = img[nc * p + 2];
    }
    stbi_image_free(img);

    // ===== Reservas GPU y medir memoria usada (delta) =====
    size_t free0, total0, free1, total1;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

    unsigned char* d_in = nullptr, * d_out = nullptr;
    float* d_tmp = nullptr; // tmp intermedio en float
    CUDA_CHECK(cudaMalloc(&d_in, bytesRGB));
    CUDA_CHECK(cudaMalloc(&d_out, bytesRGB));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytesRGB * sizeof(float) / sizeof(unsigned char)));

    CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    size_t deltaBytes = free0 - free1; // overhead real del driver + tus buffers

    // Copia H->D (no se cronometra porque solo pides tiempo de procesamiento)
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytesRGB, cudaMemcpyHostToDevice));

    // ===== Configurar bloques y grilla =====
    dim3 block(blockX, blockY);
    dim3 grid((w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    printf("Imagen: %dx%d  N=%d  PASSES=%d  block=(%d,%d) grid=(%u,%u)\n",
        w, h, N, PASSES, block.x, block.y, grid.x, grid.y);

    // ===== Medición SOLO procesamiento (kernels) =====
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // Pases ping-pong en device: d_in -> d_out por cada pasada (separable)
    // Cada pase: horiz (u8->f tmp) + vert (f tmp -> u8)
    for (int p = 0; p < PASSES; ++p) {
        box_horiz_u8_to_f << <grid, block >> > (d_in, d_tmp, w, h, N);
        CUDA_CHECK(cudaGetLastError());
        box_vert_f_to_u8 << <grid, block >> > (d_tmp, d_out, w, h, N);
        CUDA_CHECK(cudaGetLastError());
        // swap para siguiente pasada
        unsigned char* tmp = d_in; d_in = d_out; d_out = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seg = ms / 1000.0;

    // ===== Copiar resultado (no cuenta en tiempo de procesamiento) =====
    CUDA_CHECK(cudaMemcpy(h_out, d_in /*último en 'in'*/, bytesRGB, cudaMemcpyDeviceToHost));

    // ===== Guardar salida =====
    if (strstr(OUTPUT_PATH, ".jpg") || strstr(OUTPUT_PATH, ".jpeg"))
        stbi_write_jpg(OUTPUT_PATH, w, h, 3, h_out, 95);
    else
        stbi_write_png(OUTPUT_PATH, w, h, 3, h_out, w * 3);

    // ===== Reporte =====
    double memMB_delta = (double)deltaBytes / (1024.0 * 1024.0);
    // Teórica solo por buffers (aprox): d_in + d_out + d_tmp(float3)
    size_t bytes_buffers = bytesRGB /*in*/ + bytesRGB /*out*/ + (bytesRGB * sizeof(float));
    double memMB_theory = (double)bytes_buffers / (1024.0 * 1024.0);

    printf("\n================ CUDA – Box Blur (separable) ================\n");
    printf("Tiempo Paralelo (s): %.6f\n", seg);
    printf("Memoria Usada (delta cudaMemGetInfo) MB: %.2f\n", memMB_delta);
    printf("Memoria Buffers (teórica) MB: %.2f\n", memMB_theory);
    printf("Salida: %s\n", OUTPUT_PATH);

    // ===== Limpieza =====
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    free(h_in); free(h_out);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
