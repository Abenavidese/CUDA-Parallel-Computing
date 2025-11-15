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
