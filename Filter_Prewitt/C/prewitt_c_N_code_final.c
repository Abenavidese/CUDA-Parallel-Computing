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
static const char* INPUT_PATH   = "tajmahal6000.jpg";
static const char* OUTPUT_PATH  = "out_prewitt_c.png";
static const int   N_KERNEL     = 61;       // impar >=3
static const double GAIN        = 8.0;      // brillo de bordes
static const int   PRE_BLUR_3x3 = 1;        // 1=on, 0=off
static const int   CONTRAST_STRETCH = 1;    // 1=on, 0=off
static const char* CSV_PATH     = "tiempos_prewitt.csv";

static inline int clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
static inline size_t IDX(int x,int y,int w){ return (size_t)y*w + x; }

static double now_ms(){
#if defined(CLOCK_MONOTONIC)
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
#else
    return (double)clock()*1000.0/(double)CLOCKS_PER_SEC;
#endif
}
static double get_mem_mb(){
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru; if(getrusage(RUSAGE_SELF,&ru)==0){
    #ifdef __APPLE__
        return ru.ru_maxrss/(1024.0*1024.0);
    #else
        return ru.ru_maxrss/1024.0;
    #endif
    }
#endif
    return 0.0;
}

static unsigned char* to_gray(const unsigned char* img,int w,int h,int nc){
    size_t N=(size_t)w*h;
    unsigned char* g=(unsigned char*)malloc(N);
    if(!g) return NULL;
    for(size_t p=0;p<N;p++){
        int R=img[nc*p+0], G=img[nc*p+1], B=img[nc*p+2];
        int y = (int)(0.299*R + 0.587*G + 0.114*B + 0.5);
        if(y<0)y=0; if(y>255)y=255; g[p]=(unsigned char)y;
    }
    return g;
}
static void blur3x3_u8(const unsigned char* in, unsigned char* out, int w, int h){
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            int acc=0;
            for(int j=-1;j<=1;j++){
                int yy=clampi(y+j,0,h-1);
                for(int i=-1;i<=1;i++){
                    int xx=clampi(x+i,0,w-1);
                    acc += in[IDX(xx,yy,w)];
                }
            }
            out[IDX(x,y,w)] = (unsigned char)(acc/9);
        }
    }
}
static void prewitt_into_u8(const unsigned char* gray, unsigned char* out, int w, int h, int N, double gain){
    int r=N/2;
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            long gx=0, gy=0;
            for(int j=-r;j<=r;j++){
                int yy=clampi(y+j,0,h-1);
                for(int i=-r;i<=r;i++){
                    int xx=clampi(x+i,0,w-1);
                    int g=gray[IDX(xx,yy,w)];
                    int sx = (i<0?-1:(i>0?+1:0));
                    int sy = (j<0?-1:(j>0?+1:0));
                    gx += sx*g; gy += sy*g;
                }
            }
            long mag = labs(gx) + labs(gy);
            long val = (long)((mag * gain) / (N*(double)N));
            if(val>255) val=255; if(val<0) val=0;
            out[IDX(x,y,w)] = (unsigned char)val;
        }
    }
}
static void stretch_inplace_u8(unsigned char* g, int w, int h){
    int max=0; size_t N=(size_t)w*h;
    for(size_t i=0;i<N;i++){ if(g[i]>max) max=g[i]; }
    if(max==0) return;
    for(size_t i=0;i<N;i++){
        int ng = (int)( (g[i]*255.0)/max + 0.5 );
        if(ng>255) ng=255;
        g[i]=(unsigned char)ng;
    }
}

int main(){
    if(N_KERNEL<3 || (N_KERNEL%2)==0){ fprintf(stderr,"N debe ser impar >=3\n"); return 1; }

    int w,h,nc;
    unsigned char* img = stbi_load(INPUT_PATH,&w,&h,&nc,0);
    if(!img){ fprintf(stderr,"No se pudo cargar %s\n", INPUT_PATH); return 1; }
    if(nc<3){ fprintf(stderr,"Se requiere RGB(A)\n"); stbi_image_free(img); return 1; }

    unsigned char* gray = to_gray(img,w,h,nc);
    if(!gray){ fprintf(stderr,"Sin memoria\n"); stbi_image_free(img); return 1; }

    if(PRE_BLUR_3x3){
        unsigned char* g2=(unsigned char*)malloc((size_t)w*h);
        blur3x3_u8(gray,g2,w,h);
        free(gray); gray=g2;
    }

    unsigned char* mag  = (unsigned char*)malloc((size_t)w*h);
    if(!mag){ fprintf(stderr,"Sin memoria\n"); stbi_image_free(img); free(gray); return 1; }

    double t0=now_ms();
    prewitt_into_u8(gray,mag,w,h,N_KERNEL,GAIN);
    if(CONTRAST_STRETCH) stretch_inplace_u8(mag,w,h);
    double t1=now_ms();

    // Guardar como RGB gris
    unsigned char* rgb = (unsigned char*)malloc((size_t)w*h*3);
    for(size_t p=0;p<(size_t)w*h;p++){ rgb[3*p+0]=mag[p]; rgb[3*p+1]=mag[p]; rgb[3*p+2]=mag[p]; }
    if(strstr(OUTPUT_PATH,".jpg")||strstr(OUTPUT_PATH,".jpeg")) stbi_write_jpg(OUTPUT_PATH,w,h,3,rgb,95);
    else stbi_write_png(OUTPUT_PATH,w,h,3,rgb,w*3);

    double memMB=get_mem_mb();

    printf("================ C Secuencial – Prewitt (N general) ================\n");
    printf("Imagen: %dx%d\tN=%d\n", w,h,N_KERNEL);
    printf("Tiempo: %.3f ms\tMemoria usada: %.2f MB\n", (t1-t0), memMB);
    printf("Salida: %s\n", OUTPUT_PATH);

    FILE* f=fopen(CSV_PATH,"a");
    if(f){
        fprintf(f,"Prewitt,%dx%d,%d,%d,%d,%.3f,%.2f,Cseq\n", N_KERNEL,N_KERNEL,w,h,1,(t1-t0),memMB);
        fclose(f);
    }

    stbi_image_free(img); free(gray); free(mag); free(rgb);
    return 0;
}
