// === gauss_cpu.c — Filtro Gaussiano CPU sin hilos, salida RGB como Java ===
// Compilar: gcc -O3 -march=native -std=c11 gauss_cpu.c -o gauss_cpu -lm
// Ejecutar : ./gauss_cpu

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

static inline double now_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static int ensure_dir_exists(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        fprintf(stderr, "Existe pero no es carpeta: %s\n", path);
        return -1;
    }
    if (mkdir(path, 0755) == 0) return 0;
    if (errno == EEXIST) return 0;
    perror("mkdir");
    return -1;
}

// === KERNEL GAUSSIANO NxN (normalizado) ===
double* crear_kernel_gaussiano(int size, double sigma) {
    int c = size / 2;
    double *k = (double*)malloc((size_t)size * (size_t)size * sizeof(double));
    if (!k) return NULL;

    double sigma2 = sigma * sigma;
    double twoSigma2 = 2.0 * sigma2;
    double PI = acos(-1.0);
    double norm = 1.0 / (2.0 * PI * sigma2);

    double sum = 0.0;
    for (int y = -c; y <= c; y++) {
        for (int x = -c; x <= c; x++) {
            double val = norm * exp(-(x * x + y * y) / twoSigma2);
            k[(y + c) * size + (x + c)] = val;
            sum += val;
        }
    }
    if (sum != 0.0) {
        for (int i = 0; i < size * size; i++) k[i] /= sum;
    }
    return k;
}

int main(void) {
    // === RUTAS ===
    const char* RUTA_IN  = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/images.jpg";
    const char* DIR_OUT  = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/";

    // === ENTRADA ===
    int maskSize;
    printf("Tamaño de máscara (impar, >=3): ");
    if (scanf("%d", &maskSize) != 1) {
        fprintf(stderr, "Entrada inválida.\n");
        return 1;
    }
    if (maskSize < 3 || (maskSize % 2) == 0) {
        fprintf(stderr, "Tamaño de máscara inválido. Debe ser impar y >= 3.\n");
        return 1;
    }

    // === CARGA DE IMAGEN ===
    int w, h, comp_in;
    unsigned char *img = stbi_load(RUTA_IN, &w, &h, &comp_in, 3);
    if (!img) {
        fprintf(stderr, "No se pudo cargar la imagen: %s\n", RUTA_IN);
        return 1;
    }
    printf("Imagen: %dx%d (entrada reportada %d canales; forzado a 3)\n", w, h, comp_in);

    // === BUFFERS ===
    unsigned char *gray = (unsigned char*)malloc((size_t)w * (size_t)h);
    // salida RGB (3 canales) para igualar Java TYPE_INT_RGB
    unsigned char *out_rgb  = (unsigned char*)malloc((size_t)w * (size_t)h * 3);
    if (!gray || !out_rgb) {
        fprintf(stderr, "Sin memoria.\n");
        free(gray); free(out_rgb); stbi_image_free(img);
        return 1;
    }

    // === RGB -> GRIS (misma luminancia que Java) ===
    // WR=0.2126 WG=0.7152 WB=0.0722
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            int r = img[idx + 0], g = img[idx + 1], b = img[idx + 2];
            int lum = (int)llround(0.2126*r + 0.7152*g + 0.0722*b);
            if (lum < 0) lum = 0; else if (lum > 255) lum = 255;
            gray[y * w + x] = (unsigned char)lum;
        }
    }

    // === KERNEL GAUSSIANO (sigma como en Java) ===
    double sigma = maskSize / 6.0;
    double *kernel = crear_kernel_gaussiano(maskSize, sigma);
    if (!kernel) {
        fprintf(stderr, "No se pudo crear el kernel gaussiano.\n");
        free(out_rgb); free(gray); stbi_image_free(img);
        return 1;
    }

    // === CONVOLUCIÓN SERIAL (SIN HILOS) ===
    int c = maskSize / 2;
    printf("Aplicando Gaussiano %dx%d (sigma=%.3f) en CPU sin hilos...\n", maskSize, maskSize, sigma);

    double t0 = now_seconds();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double acc = 0.0;
            for (int ky = -c; ky <= c; ky++) {
                int yy = clampi(y + ky, 0, h - 1);
                int krow = (ky + c) * maskSize;
                for (int kx = -c; kx <= c; kx++) {
                    int xx = clampi(x + kx, 0, w - 1);
                    double wgt = kernel[krow + (kx + c)];
                    acc += (double)gray[yy * w + xx] * wgt;
                }
            }
            int val = (int)llround(acc);
            if (val < 0) val = 0; else if (val > 255) val = 255;

            int oidx = (y * w + x) * 3;
            out_rgb[oidx + 0] = (unsigned char)val;
            out_rgb[oidx + 1] = (unsigned char)val;
            out_rgb[oidx + 2] = (unsigned char)val;
        }
    }

    double t1 = now_seconds();
    printf("Tiempo de convolución: %.3f s\n", (t1 - t0));

    // === SALIDA EN DISCO  ===
    if (ensure_dir_exists(DIR_OUT) != 0) {
        fprintf(stderr, "No se pudo asegurar la carpeta de salida: %s\n", DIR_OUT);
        free(kernel); free(out_rgb); free(gray); stbi_image_free(img);
        return 1;
    }

    char nombre[256];
    snprintf(nombre, sizeof(nombre),
             "resultado_gaussiano_sin_hilos_C_%dx%d_sigma%.3f.png",
             maskSize, maskSize, sigma);

    char ruta_out[1024];
    snprintf(ruta_out, sizeof(ruta_out), "%s%s", DIR_OUT, nombre);

    // escribir como 3 canales (RGB)
    if (!stbi_write_png(ruta_out, w, h, 3, out_rgb, w * 3)) {
        fprintf(stderr, "No se pudo guardar la imagen de salida: %s\n", ruta_out);
        free(kernel); free(out_rgb); free(gray); stbi_image_free(img);
        return 1;
    }
    printf("Guardado en: %s\n", ruta_out);

    // === LIMPIEZA ===
    free(kernel);
    free(out_rgb);
    free(gray);
    stbi_image_free(img);
    return 0;
}
