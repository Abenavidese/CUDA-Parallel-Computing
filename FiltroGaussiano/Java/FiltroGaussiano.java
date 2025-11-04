import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;
import javax.imageio.ImageIO;

public class FiltroGaussiano {

    static int numHilos;
    static int maskSize;
    static double sigma;
    static BufferedImage imagenOriginal;
    static BufferedImage imagenResultado;
    static int[][] gray;
    static double[][] KERNEL;

    // === CONSTANTES DE LUMINANCIA (IGUALES EN AMBAS VERSIONES) ===
    static final double WR = 0.2126;
    static final double WG = 0.7152;
    static final double WB = 0.0722;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        try {
            System.out.println("Iniciando FiltroGaussiano (multihilo)");

            // === CARGA DE IMAGEN ===
            String ruta = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/images.jpg";
            File f = new File(ruta);
            if (!f.exists()) {
                System.out.println("No existe la imagen: " + ruta);
                return;
            }
            imagenOriginal = ImageIO.read(f);
            if (imagenOriginal == null) {
                System.out.println("Formato de imagen no soportado o archivo corrupto.");
                return;
            }
            final int width = imagenOriginal.getWidth();
            final int height = imagenOriginal.getHeight();
            System.out.println("Imagen cargada: " + width + "x" + height);

            // === PARÁMETROS ===
            int maxHilos = Runtime.getRuntime().availableProcessors();
            System.out.println("Ingrese número de hilos (1.." + maxHilos + "): ");
            numHilos = sc.nextInt();
            if (numHilos < 1 || numHilos > maxHilos) {
                System.out.println("Número de hilos inválido.");
                return;
            }

            System.out.println("Ingrese tamaño de máscara (impar, >=3): ");
            maskSize = sc.nextInt();
            if (maskSize < 3 || (maskSize % 2) == 0) {
                System.out.println("Tamaño de máscara inválido.");
                return;
            }

            // === KERNEL GAUSSIANO ===
            sigma = maskSize / 6.0;
            KERNEL = crearKernelGaussiano(maskSize, sigma);
            System.out.printf("Kernel gaussiano %dx%d creado (sigma=%.3f)%n", maskSize, maskSize, sigma);

            // === ESCALA DE GRISES (MISMA FÓRMULA QUE EN CPU) ===
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
                hilos[i] = new Thread(new WorkerGaussiano(yIni, yFin, i + 1));
                hilos[i].start();
            }
            for (Thread t : hilos) t.join();

            long end = System.nanoTime();
            double segundos = (end - start) / 1_000_000_000.0;

            // === MEMORIA (opcional, solo diagnóstico) ===
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long memoriaUsada = runtime.totalMemory() - runtime.freeMemory();
            double memoriaMB = memoriaUsada / (1024.0 * 1024.0);

            System.out.printf("Tiempo total de convolución: %.3f s%n", segundos);
            System.out.printf("Memoria utilizada: %.2f MB%n", memoriaMB);

            // === GUARDAR RESULTADO EN PNG  ===
            String carpetaResultados = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/";
            File carpeta = new File(carpetaResultados);
            if (!carpeta.exists()) carpeta.mkdir();

            String nombre = String.format("resultado_gaussiano_%dx%d_%dhilos_sigma%.3f.png",
                    maskSize, maskSize, numHilos, sigma).replace(',', '.');
            File out = new File(carpetaResultados + nombre);
            ImageIO.write(imagenResultado, "png", out);
            System.out.println("Imagen guardada en: " + out.getAbsolutePath());

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            sc.close();
            System.out.println("Programa finalizado.");
        }
    }

    // === TRABAJO DE CADA HILO  ===
    static class WorkerGaussiano implements Runnable {
        final int yIni, yFin, id;

        WorkerGaussiano(int yIni, int yFin, int id) {
            this.yIni = yIni;
            this.yFin = yFin;
            this.id = id;
        }

        @Override
        public void run() {
            try {
                final int h = imagenOriginal.getHeight();
                final int w = imagenOriginal.getWidth();
                final int kH = KERNEL.length;
                final int kW = KERNEL[0].length;
                final int oy = kH / 2;
                final int ox = kW / 2;

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
                        int val = (int) Math.round(sum);
                        val = clamp(val, 0, 255);
                        int rgb = (val << 16) | (val << 8) | val;
                        imagenResultado.setRGB(x, y, rgb);
                    }
                }
            } catch (Exception e) {
                System.out.println("Error en hilo " + id + ": " + e.getMessage());
            }
        }
    }

    // === ESCALA DE GRISES CONSISTENTE ===
    static void toGrayLuminance(BufferedImage src, int[][] dstGray) {
        final int h = src.getHeight();
        final int w = src.getWidth();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = (rgb) & 0xFF;
                int lum = (int) Math.round(WR * r + WG * g + WB * b);
                dstGray[y][x] = lum;
            }
        }
    }

    // === KERNEL GAUSSIANO (MISMO EN AMBAS VERSIONES) ===
    static double[][] crearKernelGaussiano(int size, double sigma) {
        double[][] k = new double[size][size];
        int c = size / 2;
        double sigma2 = sigma * sigma;
        double twoSigma2 = 2.0 * sigma2;
        double norm = 1.0 / (2.0 * Math.PI * sigma2);
        double sum = 0.0;

        for (int y = -c; y <= c; y++) {
            for (int x = -c; x <= c; x++) {
                double val = norm * Math.exp(-(x * x + y * y) / twoSigma2);
                k[y + c][x + c] = val;
                sum += val;
            }
        }
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                k[y][x] /= sum; // normaliza a suma 1.0
            }
        }
        return k;
    }

    static int clamp(int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi) ? hi : v;
    }
}
