import time
import math
import tracemalloc
import numpy as np
from PIL import Image

RUTA_IMAGEN = "/home/bryam/Escritorio/Ups/Computacion_Paralela/Proyecto/resultados/4.jpg"

def generar_kernel_gaussiano(tamano):
    sigma = tamano / 3.0
    centro = tamano // 2
    kernel = np.zeros((tamano, tamano), dtype=float)
    for i in range(tamano):
        for j in range(tamano):
            x = i - centro
            y = j - centro
            kernel[i, j] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
    suma = np.sum(kernel)
    kernel /= suma
    return kernel

def convolucion_gaussiana(imagen, kernel):
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
