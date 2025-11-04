import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class BoxBlurJavaSeqN_Code {

    static final String INPUT_PATH = "cuenca9000.jpg"; // ruta
    static final String OUTPUT_PATH = "out_box_seq.png";         // salida
    static final int N_KERNEL = 21;   // 9,21,65 (una N por corrida)
    static final int PASSES   = 3;    // 1..3  (para notar más el cambio con N=9/21)
    static final String CSV_PATH = "tiempos_box.csv"; // opcional

    private static int clamp(int v, int lo, int hi){
        return v < lo ? lo : (v > hi ? hi : v);
    }

    private static BufferedImage loadImage(String path) throws IOException {
        return ImageIO.read(new File(path));
    }

    private static void saveImage(BufferedImage img, String path) throws IOException {
        String fmt = path.toLowerCase().endsWith(".jpg") || path.toLowerCase().endsWith(".jpeg") ? "jpg" : "png";
        ImageIO.write(img, fmt, new File(path));
    }

    private static void boxBlurInto(int[] srcPix, int[] dstPix, int w, int h, int k){
        int r = k/2; float norm = 1.0f/(k*(float)k);
        for(int y=0; y<h; y++){
            for(int x=0; x<w; x++){
                float accR=0, accG=0, accB=0;
                for(int j=-r; j<=r; j++){
                    int yy = clamp(y+j, 0, h-1);
                    int base = yy * w;
                    for(int i=-r; i<=r; i++){
                        int xx = clamp(x+i, 0, w-1);
                        int rgb = srcPix[base + xx];
                        accR += (rgb>>16)&0xFF;
                        accG += (rgb>>8) &0xFF;
                        accB +=  rgb      &0xFF;
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

    public static void main(String[] args) throws Exception {
        BufferedImage src = loadImage(INPUT_PATH);
        int W = src.getWidth(), H = src.getHeight();
        int[] A = src.getRGB(0,0,W,H,null,0,W);
        int[] B = new int[W*H];

        System.gc();
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long t0 = System.nanoTime();

        int[] in=A, out=B;
        for(int p=0; p<PASSES; p++){
            boxBlurInto(in, out, W, H, N_KERNEL);
            int[] tmp = in; in = out; out = tmp; // ping-pong
        }
        int[] finalPix = (PASSES%2==0)? A : B; // si PASSES par, el resultado quedó en A (por último swap)
        if(PASSES%2==0){
            // copiar A->finalPix para no romper el original
            finalPix = new int[W*H];
            System.arraycopy(A, 0, finalPix, 0, W*H);
        }

        BufferedImage outImg = new BufferedImage(W,H,BufferedImage.TYPE_INT_RGB);
        outImg.setRGB(0,0,W,H, (PASSES%2==0)? A : B, 0, W);

        long t1 = System.nanoTime();
        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        saveImage(outImg, OUTPUT_PATH);

        double ms = (t1 - t0)/1e6;
        double memMB = toMB(Math.max(0L, memAfter - memBefore));

        System.out.println("================ Java Secuencial – Box Blur (multi‑paso) ================");
        System.out.printf("Imagen: %dx%d	N=%d	Pases=%d", W, H, N_KERNEL, PASSES);
        System.out.printf("Tiempo total: %,.3f ms	Memoria usada: %,.2f MB", ms, memMB);
        System.out.println("Salida: " + OUTPUT_PATH);

        appendCsv(new File(CSV_PATH), String.format(
            "BoxBlur,%dx%d,%d,%d,%d,%.3f,%.2f,JavaSeq,%d",
            N_KERNEL,N_KERNEL,W,H,1,ms,memMB,PASSES
        ));
    }
}


