import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class FiltroGaussiano {

    // Rutas 
    static final String INPUT_PATH    = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/3.jpg";
    static final String OUTPUT_PREFIX = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/resultado_gauss_par";
    static final String DIFF_PREFIX   = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/diff_gauss_par_vs_seq";
    static final String CSV_PATH      = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/tiempos_gauss.csv";

    static final int[]  N_HILOS  = {1, 2, 4, 8};
    static final int    N_KERNEL = 60;                
    static final boolean VERIFY_VISUAL = true;

    // Luminancia
    static final double WR = 0.2126, WG = 0.7152, WB = 0.0722;

    static int clamp(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }

    static BufferedImage loadImage(String path) throws IOException { return ImageIO.read(new File(path)); }

    static void saveImage(BufferedImage img, String path) throws IOException { ImageIO.write(img, "png", new File(path)); }

    static int[][] toGrayLuminance(BufferedImage src){
        int h = src.getHeight(), w = src.getWidth();
        int[][] g = new int[h][w];
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                int rgb = src.getRGB(x,y);
                int r = (rgb>>16)&0xFF, gch=(rgb>>8)&0xFF, b=rgb&0xFF;
                g[y][x] = (int)Math.round(WR*r + WG*gch + WB*b);
            }
        }
        return g;
    }

    static double[][] crearKernelGaussiano(int size){
        int N = (size & 1)==0 ? size+1 : size;
        double sigma = Math.max(0.1, (N - 1) / 6.0);
        double[][] k = new double[N][N];
        int c = N/2;
        double sigma2 = sigma*sigma, twoSigma2 = 2.0*sigma2;
        double norm = 1.0/(2.0*Math.PI*sigma2);
        double sum = 0.0;
        for(int y=-c;y<=c;y++){
            for(int x=-c;x<=c;x++){
                double val = norm * Math.exp(-(x*x + y*y)/twoSigma2);
                k[y+c][x+c] = val;
                sum += val;
            }
        }
        for(int y=0;y<N;y++) for(int x=0;x<N;x++) k[y][x] /= sum;
        return k;
    }

    static void gaussSequential(int[][] src, int[][] dst, int w, int h, double[][] K){
        int kh = K.length, kw = K[0].length, oy = kh/2, ox = kw/2;
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                double acc = 0.0;
                for(int ky=-oy; ky<=oy; ky++){
                    int yy = clamp(y+ky,0,h-1);
                    for(int kx=-ox; kx<=ox; kx++){
                        int xx = clamp(x+kx,0,w-1);
                        acc += src[yy][xx] * K[ky+oy][kx+ox];
                    }
                }
                dst[y][x] = clamp((int)Math.round(acc), 0, 255);
            }
        }
    }

    static void gaussRows(int ys, int ye, int w, int h, int[][] src, int[][] dst, double[][] K){
        int kh = K.length, kw = K[0].length, oy = kh/2, ox = kw/2;
        for(int y=ys;y<ye;y++){
            for(int x=0;x<w;x++){
                double acc = 0.0;
                for(int ky=-oy; ky<=oy; ky++){
                    int yy = clamp(y+ky,0,h-1);
                    for(int kx=-ox; kx<=ox; kx++){
                        int xx = clamp(x+kx,0,w-1);
                        acc += src[yy][xx] * K[ky+oy][kx+ox];
                    }
                }
                dst[y][x] = clamp((int)Math.round(acc), 0, 255);
            }
        }
    }

    static int diffAndSave(int[] parPix, int[] refPix, int w, int h, String outPath) throws IOException {
        final int BLACK=0x000000, RED=0xFF0000;
        int mismatches = 0;
        int[] D = new int[w*h];
        for(int i=0;i<w*h;i++){
            if(parPix[i]==refPix[i]) D[i]=BLACK; else { D[i]=RED; mismatches++; }
        }
        BufferedImage diff = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        diff.setRGB(0,0,w,h,D,0,w);
        ImageIO.write(diff, "png", new File(outPath));
        return mismatches;
    }

    public static void main(String[] args) throws Exception {
        BufferedImage src = loadImage(INPUT_PATH);
        if (src == null) return;
        int w = src.getWidth(), h = src.getHeight();

        int[][] A = toGrayLuminance(src);
        int[][] B = new int[h][w];
        double[][] KERNEL = crearKernelGaussiano(N_KERNEL);

        // Referencia secuencial 
        int[][] ref = new int[h][w];
        gaussSequential(A, ref, w, h, KERNEL);
        int[] refPix = new int[w*h];
        for(int y=0;y<h;y++) for(int x=0;x<w;x++){ int v=ref[y][x]; refPix[y*w+x]=(v<<16)|(v<<8)|v; }

        // CSV
        File csv = new File(CSV_PATH);
        boolean existed = csv.exists();
        try(PrintWriter pw = new PrintWriter(new FileWriter(csv, true))){
            if(!existed) pw.println("filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl,pases");
        }

        System.out.printf("Imagen: %dx%d  N=%d  Pases=%d%n", w, h, N_KERNEL, 1);
        System.out.println("Hilos\tTiempo(ms)\tSpeedup\tEficiencia(%)\tMem(MB)");

        double t1_ms = -1.0;

        for(int th : N_HILOS){
            int[][] X = A;
            int[][] Y = B;

            System.gc();
            long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long t0 = System.nanoTime();

            final int[][] Xin  = X;
            final int[][] Yout = Y;
            final double[][] K = KERNEL;

            ExecutorService pool = Executors.newFixedThreadPool(th);
            List<Future<?>> futures = new ArrayList<>();
            int rowsPerTask = Math.max(1, h / (th * 2));
            for(int y0=0; y0<h; y0+=rowsPerTask){
                final int ys = y0;
                final int ye = Math.min(h, y0 + rowsPerTask);
                futures.add(pool.submit(() -> gaussRows(ys, ye, w, h, Xin, Yout, K)));
            }
            for(Future<?> f : futures) f.get();
            pool.shutdown();

            long t1 = System.nanoTime();
            long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            double ms = (t1 - t0)/1e6;
            double memMB = Math.max(0, (memAfter - memBefore)/(1024.0*1024.0));
            if(th==1) t1_ms = ms;
            double speedup = (t1_ms>0)? (t1_ms/ms) : 1.0;
            double eficiencia = 100.0 * speedup / th;
            System.out.printf(Locale.US, "%d\t%.3f\t%.2fx\t%.2f\t%.2f%n", th, ms, speedup, eficiencia, memMB);

            int[] outPix = new int[w*h];
            for(int y=0;y<h;y++) for(int x=0;x<w;x++){ int v=Y[y][x]; outPix[y*w+x]=(v<<16)|(v<<8)|v; }
            BufferedImage outImg = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
            outImg.setRGB(0,0,w,h,outPix,0,w);
            String outPath = String.format("%s_N%02d_T%02d.png", OUTPUT_PREFIX, N_KERNEL, th);
            saveImage(outImg, outPath);

            try(PrintWriter pw = new PrintWriter(new FileWriter(csv, true))){
                pw.printf(Locale.US, "Gauss,%dx%d,%d,%d,%d,%.3f,%.2f,JavaPar,%d%n",
                        N_KERNEL, N_KERNEL, w, h, th, ms, memMB, 1);
            }

            if(VERIFY_VISUAL){
                String diffPath = String.format("%s_N%02d_T%02d.png", DIFF_PREFIX, N_KERNEL, th);
                int mismatches = diffAndSave(outPix, refPix, w, h, diffPath);
                if(mismatches==0) System.out.println("[Verificación] T="+th+" idéntico. "+diffPath);
                else {
                    double pct = (mismatches/(double)(w*h))*100.0;
                    System.out.printf(Locale.US, "[Verificación] T=%d Diferencias: %,d (%.6f%%). %s%n",
                            th, mismatches, pct, diffPath);
                }
            }
        }
    }
}
