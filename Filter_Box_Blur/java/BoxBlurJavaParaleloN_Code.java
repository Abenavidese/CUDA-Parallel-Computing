import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.Locale;

public class BoxBlurJavaParaleloN_Code {
    // === EDITA AQUÍ ===
    static final String INPUT_PATH    = "cuenca9000.jpg";
    static final String OUTPUT_PREFIX = "out_box_par"; // salida paralela por hilo
    static final int    N_KERNEL      = 21;            // 9, 21 o 65 (impar >=3)
    static final int    PASSES        = 3;             // 2–3 para notar más con 9/21
    static final int[]  N_HILOS       = {1,4,8,16};
    static final String CSV_PATH      = "tiempos_box.csv";

    // Verificación visual (siempre contra referencia generada aquí mismo)
    static final boolean VERIFY_VISUAL = true;
    static final String  DIFF_PREFIX   = "diff_par_vs_seq"; // prefijo mapas de diferencias

    private static int clamp(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }

    private static BufferedImage loadImage(String path) throws IOException { return ImageIO.read(new File(path)); }
    private static void saveImage(BufferedImage img, String path) throws IOException {
        String fmt = path.toLowerCase().endsWith(".jpg") || path.toLowerCase().endsWith(".jpeg") ? "jpg" : "png";
        ImageIO.write(img, fmt, new File(path));
    }

    // Paso de blur (idéntico al secuencial) src[] -> dst[]
    private static void boxBlurInto(int[] srcPix, int[] dstPix, int w, int h, int k){
        int r = k/2; float norm = 1.0f/(k*(float)k);
        for(int y=0; y<h; y++){
            for(int x=0; x<w; x++){
                float accR=0, accG=0, accB=0;
                for(int j=-r; j<=r; j++){
                    int yy = clamp(y+j, 0, h-1);
                    int base = yy*w;
                    for(int i=-r; i<=r; i++){
                        int xx = clamp(x+i, 0, w-1);
                        int rgb = srcPix[base+xx];
                        accR += (rgb>>16)&0xFF;
                        accG += (rgb>>8) &0xFF;
                        accB +=  rgb     &0xFF;
                    }
                }
                int R = Math.round(accR*norm);
                int G = Math.round(accG*norm);
                int B = Math.round(accB*norm);
                if(R<0)R=0; if(R>255)R=255;
                if(G<0)G=0; if(G>255)G=255;
                if(B<0)B=0; if(B>255)B=255;
                dstPix[y*w + x] = (R<<16)|(G<<8)|B;
            }
        }
    }

    // Referencia secuencial (multi-paso ping-pong) -> int[] final
    private static int[] sequentialReference(final int[] srcPix, int w, int h, int k, int passes){
        int[] A = Arrays.copyOf(srcPix, srcPix.length);
        int[] B = new int[w*h];
        int[] in = A, out = B;
        for(int p=0; p<passes; p++){
            boxBlurInto(in, out, w, h, k);
            int[] tmp = in; in = out; out = tmp;
        }
        return in; // resultado final
    }

    // Paralelo por filas (un paso) usando in/out provistos
    private static void boxBlurRows(int yStart, int yEnd, int w, int h, int r, float norm,
                                    final int[] srcPix, final int[] dstPix){
        for(int y=yStart; y<yEnd; y++){
            for(int x=0; x<w; x++){
                float accR=0, accG=0, accB=0;
                for(int j=-r; j<=r; j++){
                    int yy = clamp(y+j, 0, h-1);
                    int base = yy*w;
                    for(int i=-r; i<=r; i++){
                        int xx = clamp(x+i, 0, w-1);
                        int rgb = srcPix[base+xx];
                        accR += (rgb>>16)&0xFF;
                        accG += (rgb>>8) &0xFF;
                        accB +=  rgb     &0xFF;
                    }
                }
                int R = Math.round(accR*norm);
                int G = Math.round(accG*norm);
                int B = Math.round(accB*norm);
                if(R<0)R=0; if(R>255)R=255;
                if(G<0)G=0; if(G>255)G=255;
                if(B<0)B=0; if(B>255)B=255;
                dstPix[y*w + x] = (R<<16)|(G<<8)|B;
            }
        }
    }

    private static double toMB(long bytes){ return bytes / (1024.0 * 1024.0); }

    private static void appendCsv(File csv, String line) throws Exception {
        boolean exists = csv.exists();
        try (PrintWriter pw = new PrintWriter(new FileWriter(csv, true))) {
            if(!exists) pw.println("filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl,passes");
            pw.println(line);
        }
    }

    static class Res {
        int hilos; double ms, speedup, eficiencia, memMB;
        Res(int h, double t, double s, double e, double m){hilos=h;ms=t;speedup=s;eficiencia=e;memMB=m;}
    }

    // Genera mapa de diferencias y cuenta píxeles distintos
    private static int diffAndSave(int[] parPix, int[] refPix, int w, int h, String outPath) throws IOException {
        final int BLACK = 0x000000, RED = 0xFF0000;
        int mismatches = 0;
        int[] D = new int[w*h];
        for(int i=0;i<w*h;i++){
            if(parPix[i]==refPix[i]) D[i]=BLACK;
            else { D[i]=RED; mismatches++; }
        }
        BufferedImage diff = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        diff.setRGB(0,0,w,h,D,0,w);
        ImageIO.write(diff, outPath.toLowerCase().endsWith(".jpg")?"jpg":"png", new File(outPath));
        return mismatches;
    }

    public static void main(String[] args) throws Exception {
        BufferedImage src = loadImage(INPUT_PATH);
        int w = src.getWidth(), h = src.getHeight();
        final int[] srcOrig = src.getRGB(0,0,w,h,null,0,w);

        int r = N_KERNEL/2;
        float norm = 1.0f/(N_KERNEL*(float)N_KERNEL);

        // ===== Referencia secuencial generada aquí (NO se mide en speedup) =====
        System.out.println("[Ref] Generando referencia secuencial interna (no afecta tiempos del paralelo)...");
        final int[] refSeq = sequentialReference(srcOrig, w, h, N_KERNEL, PASSES);

        System.out.println("================ Java Paralelo – Box Blur (multi-paso) ================");
        System.out.printf(Locale.US, "Imagen: %dx%d\tN=%d\tPases=%d%n", w, h, N_KERNEL, PASSES);
        System.out.println("Hilos\tTiempo(ms)\tSpeedup\tEficiencia(%)\tMem(MB)");

        double t1_ms = -1.0; // baseline 1 hilo (del paralelo)
        File csv = new File(CSV_PATH);
        List<Res> resultados = new ArrayList<>();

        for(int th : N_HILOS){
            int[] A = Arrays.copyOf(srcOrig, srcOrig.length);
            int[] B = new int[w*h];
            int[] in  = A;
            int[] out = B;

            System.gc();
            long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long t0 = System.nanoTime();

            for(int p=0; p<PASSES; p++){
                final int[] inLocal  = in;
                final int[] outLocal = out;

                ExecutorService pool = Executors.newFixedThreadPool(th);
                List<Future<?>> futures = new ArrayList<>();
                int rowsPerTask = Math.max(1, h / (th * 2));
                for(int y=0; y<h; y+=rowsPerTask){
                    final int ys = y;
                    final int ye = Math.min(h, y + rowsPerTask);
                    futures.add(pool.submit(() -> boxBlurRows(ys, ye, w, h, r, norm, inLocal, outLocal)));
                }
                for(Future<?> f : futures) f.get();
                pool.shutdown();

                // ping-pong
                int[] tmp = in; in = out; out = tmp;
            }

            long t1 = System.nanoTime();
            long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

            // ---- Resultado paralelo (in) -> imagen y verificación ----
            String outPath = String.format("%s_N%02d_P%02d_T%02d.png", OUTPUT_PREFIX, N_KERNEL, PASSES, th);
            BufferedImage outImg = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
            outImg.setRGB(0,0,w,h, in, 0, w);
            saveImage(outImg, outPath);

            double ms = (t1 - t0)/1e6;
            double memMB = toMB(Math.max(0L, memAfter - memBefore));
            if(th == 1) t1_ms = ms;
            double speedup    = (t1_ms > 0) ? (t1_ms / ms) : 1.0;
            double eficiencia = 100.0 * speedup / th;

            // Consola
            System.out.printf(Locale.US, "%d\t%,.3f\t%.2fx\t%.2f\t%,.2f%n", th, ms, speedup, eficiencia, memMB);

            // CSV
            appendCsv(csv, String.format(Locale.US,
                "BoxBlur,%dx%d,%d,%d,%d,%.3f,%.2f,JavaPar,%d",
                N_KERNEL,N_KERNEL,w,h,th,ms,memMB,PASSES
            ));

            // Verificación visual: comparar arrays directamente y guardar diff
            if(VERIFY_VISUAL){
                String diffPath = String.format("%s_N%02d_P%02d_T%02d.png", DIFF_PREFIX, N_KERNEL, PASSES, th);
                int mismatches = diffAndSave(in, refSeq, w, h, diffPath);
                if(mismatches==0){
                    System.out.println("[Verificación] ✅ T="+th+" -> Paralelo BIT-A-BIT idéntico a la referencia. Mapa: "+diffPath);
                }else{
                    double pct = (mismatches / (double)(w*h)) * 100.0;
                    System.out.printf(Locale.US, "[Verificación] ⚠ T=%d -> Diferencias: %,d (%.6f%%). Mapa: %s%n",
                            th, mismatches, pct, diffPath);
                }
            }

            resultados.add(new Res(th, ms, speedup, eficiencia, memMB));
        }

        // Resumen
        Res best = resultados.stream().max(Comparator.comparingDouble(rz -> rz.speedup)).orElse(null);
        System.out.println("-----------------------------------------------------------------------");
        System.out.println("Resumen:");
        for(Res r0 : resultados){
            System.out.printf(Locale.US, "T=%2d  Tiempo=%,.3f ms  Speedup=%.2fx  Efic=%.2f%%  Mem=%,.2f MB%n",
                    r0.hilos, r0.ms, r0.speedup, r0.eficiencia, r0.memMB);
        }
        if(best != null){
            System.out.printf(Locale.US, "Mejor speedup: %.2fx con %d hilos%n", best.speedup, best.hilos);
        }

        System.out.println("Salidas paralelas: " + OUTPUT_PREFIX + "_N**_P**_T**.png");
        if(VERIFY_VISUAL) System.out.println("Mapas de diferencias: " + DIFF_PREFIX + "_N**_P**_T**.png");
        System.out.println("CSV: " + new File(CSV_PATH).getAbsolutePath());
    }
}
