package Filter_Prewitt.java;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Locale;

public class PrewittJavaSeqN_Code {

    static final String INPUT_PATH  = "Filter_Prewitt/tajmahal6000.jpg";
    static final String OUTPUT_PATH = "out_prewitt_seq.png";
    static final int    N_KERNEL    = 61;                // impar >=3
    static final String CSV_PATH    = "tiempos_prewitt.csv";

    static final boolean PRE_BLUR_3x3      = true;       // desenfoque previo leve
    static final double  GAIN              = 8.0;        // sube/baja brillo de bordes
    static final boolean CONTRAST_STRETCH  = true;       // reescala 0..max → 0..255

    private static int clampCoord(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
    private static int idx(int x,int y,int w){ return y*w + x; }

    private static BufferedImage load(String p) throws IOException { return ImageIO.read(new File(p)); }
    private static void save(BufferedImage img, String p) throws IOException {
        String fmt = p.toLowerCase().endsWith(".jpg")||p.toLowerCase().endsWith(".jpeg")?"jpg":"png";
        ImageIO.write(img, fmt, new File(p));
    }

    private static int[] rgbToArray(BufferedImage src){
        int w=src.getWidth(), h=src.getHeight();
        return src.getRGB(0,0,w,h,null,0,w);
    }
    private static int[] toGrayInt(BufferedImage src){
        int w=src.getWidth(), h=src.getHeight();
        int[] rgb = rgbToArray(src);
        int[] g   = new int[w*h];
        for(int i=0;i<w*h;i++){
            int c=rgb[i], R=(c>>16)&0xFF, G=(c>>8)&0xFF, B=c&0xFF;
            g[i]=(int)Math.round(0.299*R + 0.587*G + 0.114*B);
        }
        return g;
    }

    private static void blur3x3(int[] src, int[] dst, int w, int h){
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                int acc=0;
                for(int j=-1;j<=1;j++){
                    int yy=clampCoord(y+j,0,h-1);
                    for(int i=-1;i<=1;i++){
                        int xx=clampCoord(x+i,0,w-1);
                        acc += src[idx(xx,yy,w)];
                    }
                }
                dst[idx(x,y,w)] = acc/9;
            }
        }
    }

    private static void prewittInto(int[] gray, int[] dstRgb, int w, int h, int N){
        int r=N/2;
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                long gx=0, gy=0;
                for(int j=-r;j<=r;j++){
                    int yy=clampCoord(y+j,0,h-1);
                    for(int i=-r;i<=r;i++){
                        int xx=clampCoord(x+i,0,w-1);
                        int g = gray[idx(xx,yy,w)];
                        int sx = (i<0?-1:(i>0?+1:0));
                        int sy = (j<0?-1:(j>0?+1:0));
                        gx += sx*g; gy += sy*g;
                    }
                }
                long mag = Math.abs(gx)+Math.abs(gy);
                int val = (int)Math.round((mag * GAIN) / (N*(double)N));
                if(val>255) val=255; if(val<0) val=0;
                dstRgb[idx(x,y,w)] = (val<<16)|(val<<8)|val;
            }
        }
    }

    private static void stretchInPlaceGrayRGB(int[] img){
        int max=0; for(int c: img){ int g=c&0xFF; if(g>max) max=g; }
        if(max<=0) return;
        for(int i=0;i<img.length;i++){
            int g = img[i] & 0xFF;
            int ng = (int)Math.round(g*255.0/max);
            img[i] = (ng<<16)|(ng<<8)|ng;
        }
    }

    private static double toMB(long bytes){ return bytes/(1024.0*1024.0); }
    private static void appendCsv(File csv, String line) throws Exception{
        boolean exists=csv.exists();
        try(PrintWriter pw=new PrintWriter(new FileWriter(csv,true))){
            if(!exists) pw.println("filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl");
            pw.println(line);
        }
    }

    public static void main(String[] args) throws Exception{
        BufferedImage src = load(INPUT_PATH);
        int W=src.getWidth(), H=src.getHeight();

        int[] gray = toGrayInt(src);
        if(PRE_BLUR_3x3){
            int[] tmp = new int[W*H];
            blur3x3(gray,tmp,W,H);
            gray = tmp;
        }
        int[] outGrayRgb = new int[W*H];

        System.gc();
        long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long t0 = System.nanoTime();

        prewittInto(gray, outGrayRgb, W, H, N_KERNEL);
        if(CONTRAST_STRETCH) stretchInPlaceGrayRGB(outGrayRgb);

        long t1 = System.nanoTime();
        long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        BufferedImage outImg = new BufferedImage(W,H,BufferedImage.TYPE_INT_RGB);
        outImg.setRGB(0,0,W,H,outGrayRgb,0,W);
        save(outImg, OUTPUT_PATH);

        double ms=(t1-t0)/1e6;
        double memMB=toMB(Math.max(0L,memAfter-memBefore));

        System.out.println("================ Java Secuencial – Prewitt (N general) ================");
        System.out.printf(Locale.US,"Imagen: %dx%d\tN=%d%n", W,H,N_KERNEL);
        System.out.printf(Locale.US,"Tiempo: %,.3f ms\tMemoria usada: %,.2f MB%n", ms, memMB);
        System.out.println("Salida: " + OUTPUT_PATH);

        appendCsv(new File(CSV_PATH), String.format(Locale.US,
                "Prewitt,%dx%d,%d,%d,%d,%.3f,%.2f,JavaSeq",
                N_KERNEL,N_KERNEL,W,H,1,ms,memMB));
    }
}
