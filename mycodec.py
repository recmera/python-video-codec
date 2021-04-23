import numpy as np
import cv2 as cv

# Nuestras librerías
from FiltroSinusoide import filtradoIIR,filtradoFIR,filtradoFFT

def filtroMediana(copia):
    rango = 30
    for x in range(1,848-1):
        for y in range(1,480-1):
            nR = np.abs(int(copia[y,x])-int(copia[y,x+1])) > rango
            nL = np.abs(int(copia[y,x])-int(copia[y,x-1])) > rango
            nU = np.abs(int(copia[y,x])-int(copia[y+1,x])) > rango
            nD = np.abs(int(copia[y,x])-int(copia[y-1,x])) > rango
            if nR and nL and nU and nD:
                copia[y,x] = copia[y,x+1]#(copia[y+1,x]+copia[y-1,x]+copia[y,x+1]+copia[y,x-1])/4
    return copia

def denoise(frame):
    #return filtradoFIR(frame)
    #return filtradoFFT(frame)
    #frame = filtroMediana(frame)
    frame = filtradoIIR(frame)
    return filtroMediana(frame)

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
