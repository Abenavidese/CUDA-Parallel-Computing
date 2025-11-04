package Filter_Prewitt.java;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.Locale;

public class PrewittJavaParaleloN_Code {

    static final String INPUT_PATH    = "Filter_Prewitt/tajmahal6000.jpg";
    static final String OUTPUT_PREFIX = "out_prewitt_par";
    static final String DIFF_PREFIX   = "diff_prewitt_par_vs_seq";
    static final String CSV_PATH      = "tiempos_prewitt.csv";
    static final int    N_KERNEL      = 61;               // impar >=3
    static final int[]  N_HILOS       = {1,4,8,16};

    static final boolean PRE_BLUR_3x3     = true;
    static final double  GAIN             = 8.0;
    static final boolean CONTRAST_STRETCH = true;

    // Verificación
    static final boolean VERIFY_VISUAL    = true;

    private static int clampCoord(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
    private static int idx(int x,int y,int w){ return y*w + x; }
    private static BufferedImage load(String p) throws IOException { return ImageIO.read(new File(p)); }
    private static void save(BufferedImage img, String p) throws IOException {
        String fmt = p.toLowerCase().endsWith(".jpg")||p.toLowerCase().endsWith(".jpeg")?"jpg":"png";
        ImageIO.write(img, fmt, new File(p));
    }
    private static double toMB(long b){ return b/(1024.0*1024.0); }

    private static int[] rgbToArray(BufferedImage src){
        int w=src.getWidth(), h=src.getHeight();
        return src.getRGB(0,0,w,h,null,0,w);
    }
    private static int[] toGrayInt(BufferedImage src){
        int[] rgb = rgbToArray(src);
        int w=src.getWidth(), h=src.getHeight();
        int[] g = new int[w*h];
        for(int i=0;i<w*h;i++){
            int c=rgb[i], R=(c>>16)&0xFF, G=(c>>8)&0xFF, B=c&0xFF;
            g[i]=(int)Math.round(0.299*R+0.587*G+0.114*B);
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
                        acc+=src[idx(xx,yy,w)];
                    }
                }
                dst[idx(x,y,w)] = acc/9;
            }
        }
    }

    private static void prewittSeqInto(int[] gray, int[] dstRgb, int w, int h, int N, double gain){
        int r=N/2;
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                long gx=0, gy=0;
                for(int j=-r;j<=r;j++){
                    int yy=clampCoord(y+j,0,h-1);
                    for(int i=-r;i<=r;i++){
                        int xx=clampCoord(x+i,0,w-1);
                        int g=gray[idx(xx,yy,w)];
                        int sx=(i<0?-1:(i>0?+1:0));
                        int sy=(j<0?-1:(j<0?0:+1)); // (no importa el centro)
                        sy = (j<0?-1:(j>0?+1:0));
                        gx+=sx*g; gy+=sy*g;
                    }
                }
                long mag = Math.abs(gx)+Math.abs(gy);
                int val = (int)Math.round((mag * gain) / (N*(double)N));
                if(val>255) val=255; if(val<0) val=0;
                dstRgb[idx(x,y,w)] = (val<<16)|(val<<8)|val;
            }
        }
    }
    private static void prewittRows(int ys,int ye,int w,int h,int N,double gain,int[] gray,int[] dstRgb){
        int r=N/2;
        for(int y=ys;y<ye;y++){
            for(int x=0;x<w;x++){
                long gx=0, gy=0;
                for(int j=-r;j<=r;j++){
                    int yy=clampCoord(y+j,0,h-1);
                    for(int i=-r;i<=r;i++){
                        int xx=clampCoord(x+i,0,w-1);
                        int g=gray[idx(xx,yy,w)];
                        int sx=(i<0?-1:(i>0?+1:0));
                        int sy=(j<0?-1:(j>0?+1:0));
                        gx+=sx*g; gy+=sy*g;
                    }
                }
                long mag = Math.abs(gx)+Math.abs(gy);
                int val = (int)Math.round((mag * gain) / (N*(double)N));
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

    private static void appendCsv(File csv, String line) throws Exception{
        boolean exists=csv.exists();
        try(PrintWriter pw=new PrintWriter(new FileWriter(csv,true))){
            if(!exists) pw.println("filtro,ventana,ancho,alto,hilos,tiempo_ms,memoria_MB,impl");
            pw.println(line);
        }
    }
    private static int diffAndSave(int[] parRgb, int[] refRgb, int w, int h, String outPath) throws IOException{
        final int BLACK=0x000000, RED=0xFF0000;
        int mismatches=0;
        int[] D=new int[w*h];
        for(int i=0;i<w*h;i++){
            if(parRgb[i]==refRgb[i]) D[i]=BLACK; else { D[i]=RED; mismatches++; }
        }
        BufferedImage diff=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        diff.setRGB(0,0,w,h,D,0,w);
        ImageIO.write(diff, outPath.toLowerCase().endsWith(".jpg")?"jpg":"png", new File(outPath));
        return mismatches;
    }

    static class Res { int h; double ms,speed,eff,mem; Res(int h,double ms,double s,double e,double m){this.h=h;this.ms=ms;this.speed=s;this.eff=e;this.mem=m;} }

    public static void main(String[] args) throws Exception{
        BufferedImage src = load(INPUT_PATH);
        int w=src.getWidth(), h=src.getHeight();

        // Gris base (posible preblur)
        int[] grayBase = toGrayInt(src);
        if(PRE_BLUR_3x3){
            int[] tmp=new int[w*h];
            blur3x3(grayBase,tmp,w,h);
            grayBase = tmp;
        }
        final int[] grayFinal = grayBase; // <-- FINAL para lambdas

        // Referencia secuencial interna (mismas opciones)
        System.out.println("[Ref] Generando referencia secuencial interna…");
        int[] refRgb = new int[w*h];
        prewittSeqInto(grayFinal, refRgb, w, h, N_KERNEL, GAIN);
        if(CONTRAST_STRETCH) stretchInPlaceGrayRGB(refRgb);

        System.out.println("================ Java Paralelo – Prewitt (N general) ================");
        System.out.printf(Locale.US,"Imagen: %dx%d\tN=%d%n", w,h,N_KERNEL);
        System.out.println("Hilos\tTiempo(ms)\tSpeedup\tEficiencia(%)\tMem(MB)");

        File csv=new File(CSV_PATH);
        double t1_ms=-1.0;
        List<Res> rs=new ArrayList<>();

        for(int th: N_HILOS){
            int[] dstRgb = new int[w*h];

            System.gc();
            long memBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long t0 = System.nanoTime();

            ExecutorService pool=Executors.newFixedThreadPool(th);
            List<Future<?>> futures=new ArrayList<>();
            int rowsPerTask=Math.max(1, h/(th*2));
            for(int y=0;y<h;y+=rowsPerTask){
                final int ys=y, ye=Math.min(h,y+rowsPerTask);
                futures.add(pool.submit(() -> prewittRows(ys,ye,w,h,N_KERNEL,GAIN,grayFinal,dstRgb)));
            }
            for(Future<?> f: futures) f.get();
            pool.shutdown();

            if(CONTRAST_STRETCH) stretchInPlaceGrayRGB(dstRgb);

            long t1 = System.nanoTime();
            long memAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

            // Guardar bordes
            String outPath = String.format("%s_N%02d_T%02d.png", OUTPUT_PREFIX, N_KERNEL, th);
            BufferedImage out = new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
            out.setRGB(0,0,w,h,dstRgb,0,w);
            save(out, outPath);

            double ms=(t1-t0)/1e6;
            double memMB=toMB(Math.max(0L,memAfter-memBefore));
            if(th==1) t1_ms=ms;
            double speed=(t1_ms>0)?(t1_ms/ms):1.0;
            double eff=100.0*speed/th;

            System.out.printf(Locale.US,"%d\t%,.3f\t%.2fx\t%.2f\t%,.2f%n", th, ms, speed, eff, memMB);
            appendCsv(csv, String.format(Locale.US,
                "Prewitt,%dx%d,%d,%d,%d,%.3f,%.2f,JavaPar",
                N_KERNEL,N_KERNEL,w,h,th,ms,memMB));

            // Verificación visual
            if(VERIFY_VISUAL){
                String diffPath = String.format("%s_N%02d_T%02d.png", DIFF_PREFIX, N_KERNEL, th);
                int mismatches = diffAndSave(dstRgb, refRgb, w, h, diffPath);
                if(mismatches==0) System.out.println("[Verificación] ✅ T="+th+" -> Bit-a-bit idéntico. Mapa: "+diffPath);
                else {
                    double pct=(mismatches/(double)(w*h))*100.0;
                    System.out.printf(Locale.US,"[Verificación] ⚠ T=%d -> Diferencias: %,d (%.6f%%). Mapa: %s%n",
                            th, mismatches, pct, diffPath);
                }
            }
            rs.add(new Res(th,ms,speed,eff,memMB));
        }

        System.out.println("-----------------------------------------------------------------------");
        System.out.println("Resumen:");
        rs.forEach(r -> System.out.printf(Locale.US,"T=%2d  Tiempo=%,.3f ms  Speedup=%.2fx  Efic=%.2f%%  Mem=%,.2f MB%n",
                r.h, r.ms, r.speed, r.eff, r.mem));
    }
}
