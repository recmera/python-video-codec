import numpy as np
import cv2 as cv

# Nuestras librerías
from filters import filtradoIIR,filtradoMedia
from jpeg import dctAndCuantize,idctAndCuantize

peviusFrame = np.zeros((480, 848),dtype='uint8')

def denoise(frame):
    frame = filtradoIIR(frame) # Elimina ruido sinusoidal
    return filtradoMedia(frame) # Eliminar ruido de sal y pimienta


def code(frame):
    data = dctAndCuantize(frame)
    return data


reconstructedFrame = np.zeros((480, 848),dtype='uint8')
def decode(message):
    data = np.frombuffer(bytes(memoryview(message)), dtype='int8')
    print("Kbps:",int((data.size*256*30)/1000))
    frame = idctAndCuantize(data)
    return frame
