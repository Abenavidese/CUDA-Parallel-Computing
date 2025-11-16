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
