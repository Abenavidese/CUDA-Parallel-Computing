import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class FiltroLaplaciano {

    // === CONFIGURACIÓN (mantiene tus rutas) ===
    static final String INPUT_PATH    = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/1.jpg";
    static final String OUTPUT_PREFIX = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/resultado_laplace_par";
    static final String DIFF_PREFIX   = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/diff_laplace_par";
    static final String CSV_PATH      = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/tiempos_laplace.csv";

    static final int[] N_HILOS = {1, 2, 4, 8}; // número de hilos a probar
    static final int MASK_SIZE = 23;            // tamaño de máscara (puedes cambiarlo)
    static final boolean USE_8_NEIGHBORS = true;
    static final boolean VERIFY_VISUAL = true;

    static final double WR = 0.2126, WG = 0.7152, WB = 0.0722;

    // ===============================================================
    // UTILIDADES
    // ===============================================================
    private static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    private static BufferedImage loadImage(String path) throws IOException {
        return ImageIO.read(new File(path));
    }

    private static void saveImage(BufferedImage img, String path) throws IOException {
        ImageIO.write(img, "png", new File(path));
    }

    private static double[][] kernelLaplaciano3x3(boolean ochoVecinos) {
        if (ochoVecinos) {
            return new double[][]{
                {-1, -1, -1},
                {-1,  8, -1},
                {-1, -1, -1}
            };
        } else {
            return new double[][]{
                { 0, -1,  0},
                {-1,  4, -1},
                { 0, -1,  0}
            };
        }
    }

    // LoG para máscaras grandes
    private static double[][] kernelLoG(int size) {
        double sigma = size / 6.0;
        double sigma2 = sigma * sigma, sigma4 = sigma2 * sigma2;
        int c = size / 2;
        double[][] k = new double[size][size];
        double sum = 0.0;
        for (int y = -c; y <= c; y++) {
            for (int x = -c; x <= c; x++) {
                double r2 = x * x + y * y;
                double val = -((r2 - 2.0 * sigma2) / sigma4) * Math.exp(-r2 / (2.0 * sigma2));
                k[y + c][x + c] = val;
                sum += val;
            }
        }
        double corr = sum / (size * size);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                k[y][x] -= corr;
        return k;
    }

    private static int[][] toGrayLuminance(BufferedImage src) {
        int h = src.getHeight(), w = src.getWidth();
        int[][] dst = new int[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
                dst[y][x] = (int)Math.round(WR * r + WG * g + WB * b);
            }
        }
        return dst;
    }

    // ===============================================================
    // CONVOLUCIÓN SECUENCIAL (REFERENCIA)
    // ===============================================================
    private static int[] laplaceSequential(int[][] gray, int w, int h, double[][] kernel) {
        int kh = kernel.length, kw = kernel[0].length;
        int oy = kh / 2, ox = kw / 2;
        int[] out = new int[w * h];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double sum = 0;
                for (int ky = -oy; ky <= oy; ky++) {
                    int yy = clamp(y + ky, 0, h - 1);
                    for (int kx = -ox; kx <= ox; kx++) {
                        int xx = clamp(x + kx, 0, w - 1);
                        sum += gray[yy][xx] * kernel[ky + oy][kx + ox];
                    }
                }
                int val = (int)Math.min(255, Math.abs(sum));
                out[y * w + x] = (val << 16) | (val << 8) | val;
            }
        }
        return out;
    }

    // ===============================================================
    // CONVOLUCIÓN PARALELA POR FILAS
    // ===============================================================
    private static void laplaceRows(int yStart, int yEnd, int w, int h, int[][] gray, double[][] kernel, int[] dst) {
        int kh = kernel.length, kw = kernel[0].length;
        int oy = kh / 2, ox = kw / 2;

        for (int y = yStart; y < yEnd; y++) {
            for (int x = 0; x < w; x++) {
                double sum = 0;
                for (int ky = -oy; ky <= oy; ky++) {
                    int yy = clamp(y + ky, 0, h - 1);
                    for (int kx = -ox; kx <= ox; kx++) {
                        int xx = clamp(x + kx, 0, w - 1);
                        sum += gray[yy][xx] * kernel[ky + oy][kx + ox];
                    }
                }
                int val = (int)Math.min(255, Math.abs(sum));
                dst[y * w + x] = (val << 16) | (val << 8) | val;
            }
        }
    }

    // ===============================================================
    // MAPA DE DIFERENCIAS (visual)
    // ===============================================================
    private static int diffAndSave(int[] parPix, int[] refPix, int w, int h, String outPath) throws IOException {
        final int BLACK = 0x000000, RED = 0xFF0000;
        int mismatches = 0;
        int[] D = new int[w * h];
        for (int i = 0; i < w * h; i++) {
            if (parPix[i] == refPix[i]) D[i] = BLACK;
            else { D[i] = RED; mismatches++; }
        }
        BufferedImage diff = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        diff.setRGB(0, 0, w, h, D, 0, w);
        ImageIO.write(diff, "png", new File(outPath));
        return mismatches;
    }

    // ===============================================================
    // MAIN
    // ===============================================================
    public static void main(String[] args) throws Exception {

        BufferedImage src = loadImage(INPUT_PATH);
        int w = src.getWidth(), h = src.getHeight();
        System.out.printf("Imagen cargada: %dx%d%n", w, h);

        double[][] kernel = (MASK_SIZE == 3)
                ? kernelLaplaciano3x3(USE_8_NEIGHBORS)
                : kernelLoG(MASK_SIZE);

        int[][] gray = toGrayLuminance(src);

        // === Referencia secuencial ===
        System.out.println("[Ref] Generando referencia secuencial...");
        int[] refSeq = laplaceSequential(gray, w, h, kernel);

        // === CSV ===
        File csv = new File(CSV_PATH);
        boolean exists = csv.exists();
        try (PrintWriter pw = new PrintWriter(new FileWriter(csv, true))) {
            if (!exists)
                pw.println("filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl");
        }

        System.out.println("================ Java Paralelo – Laplaciano ================");
        System.out.println("Hilos\tTiempo(ms)\tSpeedup\tEficiencia(%)\tMem(MB)");

        double t1_ms = -1.0;

        for (int th : N_HILOS) {
            System.gc();
            long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long t0 = System.nanoTime();

            int[] dst = new int[w * h];
            ExecutorService pool = Executors.newFixedThreadPool(th);
            int rowsPerTask = Math.max(1, h / (th * 2));
            List<Future<?>> futures = new ArrayList<>();

            for (int y = 0; y < h; y += rowsPerTask) {
                final int ys = y;
                final int ye = Math.min(h, y + rowsPerTask);
                futures.add(pool.submit(() -> laplaceRows(ys, ye, w, h, gray, kernel, dst)));
            }
            for (Future<?> f : futures) f.get();
            pool.shutdown();

            long t1 = System.nanoTime();
            long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            double ms = (t1 - t0) / 1e6;
            double memMB = (memAfter - memBefore) / (1024.0 * 1024.0);
            if (memMB < 0) memMB = 0;

            if (th == 1) t1_ms = ms;
            double speedup = (t1_ms > 0) ? (t1_ms / ms) : 1.0;
            double eficiencia = 100.0 * speedup / th;

            System.out.printf(Locale.US, "%d\t%.3f\t%.2fx\t%.2f\t%.2f%n", th, ms, speedup, eficiencia, memMB);

            // Guardar imagen
            BufferedImage outImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            outImg.setRGB(0, 0, w, h, dst, 0, w);
            String outPath = String.format("%s_N%02d_T%02d.png", OUTPUT_PREFIX, MASK_SIZE, th);
            saveImage(outImg, outPath);

            // CSV
            try (PrintWriter pw = new PrintWriter(new FileWriter(csv, true))) {
                pw.printf(Locale.US, "Laplaciano,%dx%d,%d,%d,%d,%.3f,%.2f,JavaPar%n",
                        MASK_SIZE, MASK_SIZE, w, h, th, ms, memMB);
            }

            // Comparación visual
            if (VERIFY_VISUAL) {
                String diffPath = String.format("%s_N%02d_T%02d.png", DIFF_PREFIX, MASK_SIZE, th);
                int mismatches = diffAndSave(dst, refSeq, w, h, diffPath);
                if (mismatches == 0)
                    System.out.println("[Verificación]  T=" + th + " -> Idéntico bit a bit. Mapa: " + diffPath);
                else {
                    double pct = (mismatches / (double)(w * h)) * 100.0;
                    System.out.printf(Locale.US, "[Verificación] ⚠ T=%d -> Diferencias: %,d (%.6f%%). Mapa: %s%n",
                            th, mismatches, pct, diffPath);
                }
            }
        }

        System.out.println("-------------------------------------------------------------");
        System.out.println("CSV: " + CSV_PATH);
        System.out.println("Imágenes: " + OUTPUT_PREFIX + "_N**_T**.png");
        System.out.println("Mapas diff: " + DIFF_PREFIX + "_N**_T**.png");
    }
}
