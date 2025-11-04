

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

// === UTILIDADES ===
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

static double mem_rss_mb(void) {
    long resident_pages = 0;
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        long size_pages = 0;
        if (fscanf(f, "%ld %ld", &size_pages, &resident_pages) != 2) resident_pages = 0;
        fclose(f);
    }
    long page_sz = sysconf(_SC_PAGESIZE);
    if (page_sz <= 0) page_sz = 4096;
    return (double)resident_pages * (double)page_sz / (1024.0 * 1024.0);
}

/* === KERNELS === */

// Laplaciano 3x3 (8 vecinos)
static void kernel_laplaciano3x3(double K[3][3]) {
    static const double base[3][3] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    memcpy(K, base, sizeof(base));
}

// LoG NxN (N impar >=3), sigma ≈ N/6, suma ~ 0
static double* kernel_log(int size) {
    int c = size / 2;
    double sigma = size / 6.0;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;
    double *k = (double*)malloc((size_t)size * (size_t)size * sizeof(double));
    if (!k) return NULL;

    double sum = 0.0;
    for (int y = -c; y <= c; y++) {
        for (int x = -c; x <= c; x++) {
            double r2 = (double)(x*x + y*y);
            double val = -((r2 - 2.0 * sigma2) / sigma4) * exp(-r2 / (2.0 * sigma2));
            k[(y + c) * size + (x + c)] = val;
            sum += val;
        }
    }
    // Corrección para que la suma sea ≈ 0 (igual que en Java)
    double corr = sum / (size * size);
    for (int i = 0; i < size * size; i++) k[i] -= corr;
    return k;
}

int main(void) {
    // === RUTAS Y ENTRADA ===
    const char* RUTA_IN = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/images.jpg";
    const char* DIR_OUT = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/";

    int N;
    printf("Tamaño de máscara (impar, >=3): ");
    if (scanf("%d", &N) != 1) {
        fprintf(stderr, "Entrada inválida.\n");
        return 1;
    }
    if (N < 3 || (N % 2) == 0) {
        fprintf(stderr, "Tamaño de máscara inválido. Debe ser impar y >= 3.\n");
        return 1;
    }

    int w, h, comp_in;
    unsigned char *img = stbi_load(RUTA_IN, &w, &h, &comp_in, 3);
    if (!img) {
        fprintf(stderr, "No se pudo cargar la imagen: %s\n", RUTA_IN);
        return 1;
    }
    printf("Imagen: %dx%d (entrada reportada %d canales; forzado a 3)\n", w, h, comp_in);

    // === BUFFERS ===
    unsigned char *gray    = (unsigned char*)malloc((size_t)w * (size_t)h);
    unsigned char *out_rgb = (unsigned char*)malloc((size_t)w * (size_t)h * 3);
    if (!gray || !out_rgb) {
        fprintf(stderr, "Sin memoria.\n");
        free(gray); free(out_rgb); stbi_image_free(img);
        return 1;
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            int r = img[idx + 0], g = img[idx + 1], b = img[idx + 2];
            int lum = (int)llround(0.2126*r + 0.7152*g + 0.0722*b);
            if (lum < 0) lum = 0; else if (lum > 255) lum = 255;
            gray[y * w + x] = (unsigned char)lum;
        }
    }

    // === SELECCIÓN DE KERNEL ===
    double *Kdyn = NULL;
    double K3[3][3];
    int c = N / 2;
    int usaLoG = (N != 3);

    if (!usaLoG) {
        kernel_laplaciano3x3(K3);
        printf("Usando Laplaciano clásico 3x3 (8 vecinos).\n");
    } else {
        Kdyn = kernel_log(N);
        if (!Kdyn) {
            fprintf(stderr, "No se pudo crear el kernel LoG.\n");
            free(out_rgb); free(gray); stbi_image_free(img);
            return 1;
        }
        printf("Usando LoG %dx%d (sigma≈%.3f).\n", N, N, N/6.0);
    }

    // === CONVOLUCIÓN SERIAL (1 hilo) ===
    printf("Convolucionando...\n");
    double mem_before = mem_rss_mb();
    double t0 = now_seconds();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double acc = 0.0;
            if (!usaLoG) {
                for (int ky = -1; ky <= 1; ky++) {
                    int yy = clampi(y + ky, 0, h - 1);
                    for (int kx = -1; kx <= 1; kx++) {
                        int xx = clampi(x + kx, 0, w - 1);
                        double wgt = K3[ky + 1][kx + 1];
                        acc += (double)gray[yy * w + xx] * wgt;
                    }
                }
            } else {
                for (int ky = -c; ky <= c; ky++) {
                    int yy = clampi(y + ky, 0, h - 1);
                    int krow = (ky + c) * N;
                    for (int kx = -c; kx <= c; kx++) {
                        int xx = clampi(x + kx, 0, w - 1);
                        double wgt = Kdyn[krow + (kx + c)];
                        acc += (double)gray[yy * w + xx] * wgt;
                    }
                }
            }
            int val = (int)fabs(acc);
            if (val > 255) val = 255;

            int oidx = (y * w + x) * 3;
            out_rgb[oidx + 0] = (unsigned char)val;
            out_rgb[oidx + 1] = (unsigned char)val;
            out_rgb[oidx + 2] = (unsigned char)val;
        }
    }

    double t1 = now_seconds();
    double mem_after = mem_rss_mb();
    printf("Tiempo de convolución: %.3f s\n", (t1 - t0));
    printf("Memoria utilizada total: %.2f MB\n", mem_after);
    printf("Memoria durante la convolución (delta): %.2f MB\n", fmax(0.0, mem_after - mem_before));

    // === SALIDA (PNG RGB, como Java) ===
    if (ensure_dir_exists(DIR_OUT) != 0) {
        fprintf(stderr, "No se pudo asegurar la carpeta de salida: %s\n", DIR_OUT);
        free(Kdyn); free(out_rgb); free(gray); stbi_image_free(img);
        return 1;
    }

    char nombre[256];
    if (!usaLoG) {
        snprintf(nombre, sizeof(nombre), "resultado_LAP_%dx%d_1hilos.png", N, N);
    } else {
        snprintf(nombre, sizeof(nombre), "resultado_LoG_%dx%d_1hilos.png", N, N); 
    }

    char ruta_out[1024];
    snprintf(ruta_out, sizeof(ruta_out), "%s%s", DIR_OUT, nombre);

    if (!stbi_write_png(ruta_out, w, h, 3, out_rgb, w * 3)) {
        fprintf(stderr, "No se pudo guardar la imagen de salida: %s\n", ruta_out);
        free(Kdyn); free(out_rgb); free(gray); stbi_image_free(img);
        return 1;
    }
    printf("Guardado en: %s\n", ruta_out);

    // === LIMPIEZA ===
    free(Kdyn);
    free(out_rgb);
    free(gray);
    stbi_image_free(img);
    return 0;
}
