import numpy as np
import cv2 as cv
import heapq
from collections import Counter

# Nuestras librerías
from filters import filtradoIIR,filtradoFIR,filtradoMedia
from jpeg import dctAndCuantize,idctAndCuantize,interFrameDiff,huffman,ihuffman,bit2Str,irunLenght,runLenght


def denoise(frame):
    frame = filtradoFIR(frame) # Elimina ruido sinusoidal
    return filtradoMedia(frame) # Eliminar ruido de sal y pimienta


frameSkip = 0
def code(frame):
    global frameSkip
    if frameSkip == 1:
        frameSkip = 0
        return bytearray(1)
    frameSkip = frameSkip + 1
    changes,frame = interFrameDiff(frame)
    data = dctAndCuantize(changes,frame)
    changes = np.frombuffer(bytes(memoryview(np.packbits(changes.astype('bool').flatten()))),dtype='int8')
    out = np.concatenate((changes, data)).astype('int8')
    return huffman(out)

reconstructedFrame = 0
def decode(message):

    global bitsAvg,count,reconstructedFrame

    bts = bytes(memoryview(message))
    if len(bts) == 1:
        return reconstructedFrame

    # Convierte a string de bits
    data = ihuffman(bit2Str(bts))

    # Obtiene la matriz de cambios
    changes = np.unpackbits(np.frombuffer(data[0:795], dtype='uint8')-128, axis=None)[:6360].reshape((60,106)).astype(np.bool)
    #changes = np.unpackbits(np.frombuffer(bts[0:795], dtype='uint8'), axis=None)[:6360].reshape((60,106)).astype(np.bool)

    # Obiene los cuadros 8x8
    dcts = np.frombuffer(data[795:],dtype='uint8').astype('int16')-128
    #dcts = np.frombuffer(bts[795:],dtype='int8')

    # Realiza proceso inverso
    frame = idctAndCuantize(changes,dcts)

    # Métricas
    bitsAvg = (len(bts)*8.0*15.0)/1000.0
    print("Kbps Avg:",bitsAvg)

    reconstructedFrame = frame
    return frame
