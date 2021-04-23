import numpy as np
import cv2 as cv

# Nuestras librerías
from FiltroSinusoide import filtradoIIR,filtradoFIR,filtradoFFT
from FiltroSgtPeppers import filtradoMedia


def denoise(frame):
    #return filtradoFIR(frame)
    #return filtradoFFT(frame)
    frame = filtradoIIR(frame) # Eliminar ruido de sal y pimienta
    return filtradoMedia(frame) # Elimina ruido sinusoidal

def code(frame):
    #
    # Implementa en esta función el bloque transmisor: Transformación + Cuantización + Codificación de fuente
    #
    return frame


def decode(message):
    #
    # Reemplaza la linea 24...
    #
    frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848, order='F')
    #
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #
    return frame
