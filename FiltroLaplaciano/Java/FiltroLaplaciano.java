import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;
import javax.imageio.ImageIO;

public class FiltroLaplaciano {

    static int numHilos;
    static int maskSize;
    static BufferedImage imagenOriginal;
    static BufferedImage imagenResultado;
    static int[][] gray;
    static double[][] KERNEL;
    static boolean useLoG;
    static final boolean USE_8_NEIGHBORS = true;

    // Luminancia estándar (idéntica en ambas versiones)
    static final double WR = 0.2126, WG = 0.7152, WB = 0.0722;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        try {
            // === CARGA ===
            String ruta = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/images.jpg";
            File f = new File(ruta);
            if (!f.exists()) { System.out.println("No existe la imagen: " + ruta); return; }
            imagenOriginal = ImageIO.read(f);
            if (imagenOriginal == null) { System.out.println("Formato no soportado."); return; }
            final int width = imagenOriginal.getWidth();
            final int height = imagenOriginal.getHeight();
            System.out.println("Imagen cargada: " + width + "x" + height);

            // === PARÁMETROS ===
            int maxHilos = Runtime.getRuntime().availableProcessors();
            System.out.print("Ingrese número de hilos (1.." + maxHilos + "): ");
            numHilos = sc.nextInt();
            if (numHilos < 1 || numHilos > maxHilos) { System.out.println("Número de hilos inválido."); return; }

            System.out.print("Ingrese tamaño de máscara (impar, >=3): ");
            maskSize = sc.nextInt();
            if (maskSize < 3 || (maskSize % 2) == 0) { System.out.println("Tamaño de máscara inválido."); return; }

            // === KERNEL ===
            if (maskSize == 3) {
                KERNEL = kernelLaplaciano3x3(USE_8_NEIGHBORS);
                useLoG = false;
                System.out.println("Usando Laplaciano clásico 3x3 (8 vecinos).");
            } else {
                KERNEL = kernelLoG(maskSize);
                useLoG = true;
                System.out.println("Usando LoG " + maskSize + "x" + maskSize + ".");
            }

            // === GRIS ===
            imagenResultado = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            gray = new int[height][width];
            toGrayLuminance(imagenOriginal, gray);

            // === CONVOLUCIÓN MULTIHILO ===
            long start = System.nanoTime();
            Thread[] hilos = new Thread[numHilos];
            int filasPorHilo = height / numHilos;

            for (int i = 0; i < numHilos; i++) {
                int yIni = i * filasPorHilo;
                int yFin = (i == numHilos - 1) ? height : yIni + filasPorHilo;
                hilos[i] = new Thread(new Worker(yIni, yFin));
                hilos[i].start();
            }
            for (Thread t : hilos) t.join();
            double segundos = (System.nanoTime() - start) / 1_000_000_000.0;

            // === STATS ===
            Runtime rt = Runtime.getRuntime();
            rt.gc();
            double memMB = (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0);
            System.out.printf("Convolución finalizada en %.3f s | Memoria: %.2f MB%n", segundos, memMB);

            // === GUARDAR (PNG) ===
            String dir = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/";
            new File(dir).mkdirs();
            String modo = useLoG ? "LoG" : "LAP";
            String nombre = String.format("resultado_%s_%dx%d_%dhilos.png", modo, maskSize, maskSize, numHilos);
            File out = new File(dir + nombre);
            ImageIO.write(imagenResultado, "png", out);
            System.out.println("Guardado en: " + out.getAbsolutePath());

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            sc.close();
        }
    }

    static class Worker implements Runnable {
        final int yIni, yFin;
        Worker(int yIni, int yFin) { this.yIni = yIni; this.yFin = yFin; }

        @Override public void run() {
            final int h = imagenOriginal.getHeight();
            final int w = imagenOriginal.getWidth();
            final int kH = KERNEL.length, kW = KERNEL[0].length;
            final int oy = kH / 2, ox = kW / 2;

            for (int y = yIni; y < yFin; y++) {
                for (int x = 0; x < w; x++) {
                    double sum = 0.0;
                    for (int ky = -oy; ky <= oy; ky++) {
                        int yy = clamp(y + ky, 0, h - 1);
                        int kyIdx = ky + oy;
                        for (int kx = -ox; kx <= ox; kx++) {
                            int xx = clamp(x + kx, 0, w - 1);
                            int kxIdx = kx + ox;
                            sum += gray[yy][xx] * KERNEL[kyIdx][kxIdx];
                        }
                    }
                    int val = (int)Math.min(255.0, Math.abs(sum));
                    int rgb = (val << 16) | (val << 8) | val;
                    imagenResultado.setRGB(x, y, rgb);
                }
            }
        }
    }

    static void toGrayLuminance(BufferedImage src, int[][] dst) {
        final int h = src.getHeight(), w = src.getWidth();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
                int lum = (int)Math.round(WR*r + WG*g + WB*b);
                dst[y][x] = lum;
            }
        }
    }

    static double[][] kernelLaplaciano3x3(boolean ochoVecinos) {
        if (ochoVecinos) {
            return new double[][] {
                {-1, -1, -1},
                {-1,  8, -1},
                {-1, -1, -1}
            };
        } else {
            return new double[][] {
                { 0, -1,  0},
                {-1,  4, -1},
                { 0, -1,  0}
            };
        }
    }

    static double[][] kernelLoG(int size) {
        double sigma = size / 6.0;
        double sigma2 = sigma * sigma, sigma4 = sigma2 * sigma2;
        int c = size / 2;
        double[][] k = new double[size][size];

        double sum = 0.0;
        for (int y = -c; y <= c; y++) {
            for (int x = -c; x <= c; x++) {
                double r2 = x*x + y*y;
                double val = -((r2 - 2.0*sigma2) / sigma4) * Math.exp(-r2 / (2.0*sigma2));
                k[y + c][x + c] = val;
                sum += val;
            }
        }
        // Corrección para suma ~0 
        double corr = sum / (size * size);
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) k[y][x] -= corr;
        }
        return k;
    }

    static int clamp(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }
}
