import time
import math
import tracemalloc
import numpy as np
from PIL import Image

RUTA_IMAGEN = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/4.jpg"

def generar_kernel_laplaciano(tamano):
    kernel = -1.0 * np.ones((tamano, tamano), dtype=float)
    centro = tamano // 2
    kernel[centro, centro] = tamano * tamano - 1
    return kernel

def convolucion_laplaciana(imagen, kernel):
    alto, ancho = imagen.shape
    k = kernel.shape[0]
    margen = k // 2
    padded = np.zeros((alto + 2 * margen, ancho + 2 * margen), dtype=float)
    padded[margen:margen + alto, margen:margen + ancho] = imagen
    salida = np.zeros_like(imagen, dtype=float)

    tracemalloc.start()
    inicio = time.perf_counter()

    for i in range(alto):
        for j in range(ancho):
            acumulador = 0.0
            for ki in range(k):
                for kj in range(k):
                    acumulador += kernel[ki, kj] * padded[i + ki, j + kj]
            salida[i, j] = acumulador

    fin = time.perf_counter()
    memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tiempo = fin - inicio
    return salida, tiempo, memoria_pico

def pedir_tamano_mascara():
    while True:
        valor = input("Ingresa el tamaño de la máscara: ")
        try:
            n = int(valor)
            if n <= 0:
                print("Error: el tamaño debe ser un entero positivo.")
                continue
            if n < 3:
                print("Error: el tamaño mínimo permitido es 3.")
                continue
            if n % 2 == 0:
                n_ajustado = n + 1
                print("Se ingresó un número par. Se ajusta automáticamente a", n_ajustado)
                n = n_ajustado
            return n
        except ValueError:
            print("Error: debes ingresar un número entero válido.")

def main():
    try:
        imagen = Image.open(RUTA_IMAGEN).convert("L")
    except Exception as e:
        print("No se pudo abrir la imagen en la ruta:", RUTA_IMAGEN)
        print("Detalle del error:", e)
        return

    imagen_np = np.array(imagen, dtype=float)
    alto, ancho = imagen_np.shape

    tamano = pedir_tamano_mascara()
    kernel = generar_kernel_laplaciano(tamano)

    resultado, tiempo, memoria_pico = convolucion_laplaciana(imagen_np, kernel)

    resultado = np.abs(resultado)
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    imagen_salida = Image.fromarray(resultado)
    nombre_salida = f"filtro_laplaciano_{tamano}x{tamano}.png"
    imagen_salida.save(nombre_salida)

    print("--------- RESULTADOS ---------")
    print("Tamaño final de la máscara:", tamano, "x", tamano)
    print("Dimensiones de la imagen de entrada:", ancho, "x", alto)
    print("Tiempo de convolución (segundos):", tiempo)
    print("Memoria pico usada durante la convolución:")
    print("   ", memoria_pico, "bytes")
    print("   ", memoria_pico / (1024 * 1024), "MB")
    print("Imagen filtrada guardada como:", nombre_salida)

if __name__ == "__main__":
    main()
