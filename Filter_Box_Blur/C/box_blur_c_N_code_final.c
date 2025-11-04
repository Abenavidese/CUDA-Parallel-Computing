#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/resource.h>
#endif

// === EDITA AQUÍ ===
static const char* INPUT_PATH  = "cuenca9000.jpg"; // Ruta a tu imagen
static const char* OUTPUT_PATH = "out_box_c.png";            // Salida
static const int   N_KERNEL    = 21;  // 9, 21 o 65 (una N por corrida; impar >=3)
static const int   PASSES      = 3;   // 2–3 para notar más el cambio en N=9/21
static const char* CSV_PATH    = "tiempos_box.csv"; // se agrega una fila por corrida

static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

// Timer (ms). En POSIX usa clock_gettime; si no, usa clock() como fallback.
static double now_ms(){
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

// Memoria pico (MB) en Unix/macOS; en Windows retorna 0.0 (opcional)
static double get_mem_mb(){
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    if(getrusage(RUSAGE_SELF, &ru)==0){
    #ifdef __APPLE__
        return ru.ru_maxrss / (1024.0*1024.0); // bytes -> MB
    #else
        return ru.ru_maxrss / 1024.0;          // KB -> MB
    #endif
    }
#endif
    return 0.0;
}

// Un paso de Box Blur: src -> dst (RGB), borde: replicate (clamp)
static void box_blur_into(const unsigned char* src, unsigned char* dst,
                          int w, int h, int nc_src, int N){
    int r = N/2;
    float norm = 1.0f / (N * (float)N);
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            float accR=0, accG=0, accB=0;
            for(int j=-r; j<=r; ++j){
                int yy = clampi(y+j, 0, h-1);
                for(int i=-r; i<=r; ++i){
                    int xx = clampi(x+i, 0, w-1);
                    size_t idx = ((size_t)yy*w + xx) * nc_src;
                    accR += src[idx+0];
                    accG += src[idx+1];
                    accB += src[idx+2];
                }
            }
            int R = (int)(accR*norm + 0.5f);
            int G = (int)(accG*norm + 0.5f);
            int B = (int)(accB*norm + 0.5f);
            if(R<0)R=0; if(R>255)R=255;
            if(G<0)G=0; if(G>255)G=255;
            if(B<0)B=0; if(B>255)B=255;
            size_t oidx = ((size_t)y*w + x) * 3;
            dst[oidx+0] = (unsigned char)R;
            dst[oidx+1] = (unsigned char)G;
            dst[oidx+2] = (unsigned char)B;
        }
    }
}

int main(){
    if(N_KERNEL < 3 || (N_KERNEL % 2) == 0){
        fprintf(stderr, "N_KERNEL debe ser impar y >=3\n");
        return 1;
    }

    int w,h,nc;
    unsigned char* img = stbi_load(INPUT_PATH, &w, &h, &nc, 0);
    if(!img){
        fprintf(stderr, "No se pudo cargar %s\n", INPUT_PATH);
        return 1;
    }
    if(nc < 3){
        fprintf(stderr, "Se espera imagen con al menos 3 canales (RGB/A)\n");
        stbi_image_free(img);
        return 1;
    }

    size_t Npix = (size_t)w * h;
    unsigned char* A = (unsigned char*)malloc(Npix * 3);
    unsigned char* B = (unsigned char*)malloc(Npix * 3);
    if(!A || !B){
        fprintf(stderr, "Sin memoria\n");
        stbi_image_free(img);
        free(A); free(B);
        return 1;
    }

    // Copiar RGB desde la fuente (ignorando alfa si existe)
    for(size_t p=0; p<Npix; ++p){
        A[3*p+0] = img[nc*p+0];
        A[3*p+1] = img[nc*p+1];
        A[3*p+2] = img[nc*p+2];
    }

    // Multi-paso (ping-pong A<->B)
    double t0 = now_ms();
    unsigned char* in  = A;
    unsigned char* out = B;
    for(int pass=0; pass<PASSES; ++pass){
        box_blur_into(in, out, w, h, 3, N_KERNEL);
        unsigned char* tmp = in; in = out; out = tmp;
    }
    double t1 = now_ms();

    // El resultado quedó en 'in'
    if(strstr(OUTPUT_PATH, ".jpg") || strstr(OUTPUT_PATH, ".jpeg"))
        stbi_write_jpg(OUTPUT_PATH, w, h, 3, in, 95);
    else
        stbi_write_png(OUTPUT_PATH, w, h, 3, in, w*3);

    double memMB = get_mem_mb();

    // Consola
    printf("================ C Secuencial – Box Blur (multi-paso) ================\n");
    printf("Imagen: %dx%d\tN=%d\tPases=%d\n", w, h, N_KERNEL, PASSES);
    printf("Tiempo total: %.3f ms\tMemoria usada: %.2f MB\n", (t1 - t0), memMB);
    printf("Salida: %s\n", OUTPUT_PATH);

    // CSV (una fila por corrida)
    FILE* f = fopen(CSV_PATH, "a");
    if(f){
        // Si quieres asegurar cabecera, descomenta y maneja si no existe.
        // fprintf(f, "filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl,passes\n");
        fprintf(f, "BoxBlur,%dx%d,%d,%d,%d,%.3f,%.2f,Cseq,%d\n",
                N_KERNEL, N_KERNEL, w, h, 1, (t1 - t0), memMB, PASSES);
        fclose(f);
    }

    stbi_image_free(img);
    free(A); free(B);
    return 0;
}
