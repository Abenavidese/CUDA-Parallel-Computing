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
