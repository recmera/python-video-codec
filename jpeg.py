import numpy as np
import cv2 as cv
import heapq
from collections import Counter
import ast

# Arreglo de índices para matrices de 8x8
x8 = np.arange(0,848,8,dtype='int')
y8 = np.arange(0,480,8,dtype='int')

# Matriz de cuantización JPEG
Q = np.array([[16.0, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])


def lastNonZeroIndex(arr):
    for i in range(arr.size-1,-1,-1):
        if arr[i] != 0:
            return i+1
    return 1

def MSE(a,b):
    a = a.astype('float32')
    b = b.astype('float32')
    return ((a - b)**2).mean(axis=None)


dendograma = None
idendograma = None
with open("pruebas/Dendograma.txt","rb") as f:
    dendograma = ast.literal_eval(f.read().decode())
    idendograma = {codigo:int(simbolo) for simbolo,codigo in dendograma.items()}

# Codificación de fuente
def huffman(data):
    coded =  ''
    for n in data:
        coded += dendograma[n]

    missingBits = 8 - len(coded)%8

    for i in range(missingBits):
        coded += "0"

    b = bytearray()

    b.append(missingBits)

    for i in range(0,len(coded),8):
        byte = coded[i:i+8]
        b.append(int(byte,2))

    return b

def ihuffman(data):
    codigo = ''
    b = bytearray()
    for bit in data:
        codigo += bit
        if codigo in idendograma:
            b.append(idendograma[codigo] + 128)
            codigo = ""
    return b

# Convierte a string de bits
def bit2Str(b):
    bitsStr = ''
    extra = int(b[0])
    for i in b[1:]:
        bitsStr += "{0:08b}".format(i)

    return bitsStr[:-extra]

peviusFrame = np.full((480, 848),-1000,dtype='uint8')
def interFrameDiff(frame):
    changes = np.zeros((60,106),dtype='bool')
    xx = 0
    cont = 0
    for x in x8:
        yy = 0
        for y in y8:
            diff = MSE(peviusFrame[y:y+8,x:x+8],frame[y:y+8,x:x+8])
            if diff > 25:
                peviusFrame[y:y+8,x:x+8] = frame[y:y+8,x:x+8]
                changes[yy,xx] = 1
                cont = cont + 1
            yy = yy + 1
        xx = xx + 1
    print('Cuadros enviados:',cont,' de ',60*106)
    return changes,frame

# Obtiene transformada de coseno cada matriz de 8x8 y les aplica cuantización
# Almacena la salida cuantizada
cArr = np.zeros(480*848 + 60*106,dtype='int8')
def dctAndCuantize(changes,frame):
    frame = frame.astype('float32')
    ind = 0
    xx = 0
    # Procesa cada cuadro de 8x8
    for x in x8:
        yy = 0
        for y in y8:
            if changes[yy,xx] == 1:
                # Calcula dct y cuantiza
                f_dct = np.round(cv.dct(frame[y:y+8,x:x+8]-128)/Q)

                # Convierte a arreglo 1D en orden zigZag
                zZag = zigZag(f_dct)

                # Aplica run length
                rLenght = runLenght(zZag)

                # Los almacena en el arreglo de salida
                cArr[ind:ind+rLenght.size] = rLenght

                ind = ind + rLenght.size
            yy = yy + 1
        xx = xx + 1

    return cArr[0:ind].astype('int8')

# Obtiene transformada inversa
reconstructedFrame = np.zeros((480, 848),dtype='uint8')
def idctAndCuantize(changes,data):

    # Aplica runLenght inverso
    data = irunLenght(data)

    ind = 0
    xx = 0
    for x in x8:
        yy = 0
        for y in y8:
            if changes[yy,xx] == 1:
                # Convierte arreglo 1D a 2D
                dct = izigZag(data[ind:ind+64]).astype('float32')

                # Calcula la idct
                reconstructedFrame[y:y+8,x:x+8] = (cv.idct(dct*Q)+128).astype('uint8')

                # Suaviza borde izquierdo
                if xx != 0 and changes[yy,xx-1] == 0:
                    reconstructedFrame[y:y+8,x-1] = (reconstructedFrame[y:y+8,x-1].astype('float32') + reconstructedFrame[y:y+8,x].astype('float32'))/2.0
                if yy != 0 and changes[yy-1,xx] == 0:
                    reconstructedFrame[y-1,x:x+8] = (reconstructedFrame[y-1,x:x+8].astype('float32') + reconstructedFrame[y,x:x+8].astype('float32'))/2.0
                if xx != 105 and changes[yy,xx+1] == 0:
                    reconstructedFrame[y:y+8,x+9] = (reconstructedFrame[y:y+8,x+9].astype('float32') + reconstructedFrame[y:y+8,x+8].astype('float32'))/2.0
                if yy != 59 and changes[yy+1,xx] == 0:
                    reconstructedFrame[y+9,x:x+8] = (reconstructedFrame[y+9,x:x+8].astype('float32') + reconstructedFrame[y+8,x:x+8].astype('float32'))/2.0

                ind = ind + 64
            yy = yy + 1
        xx = xx + 1
    return reconstructedFrame

# Aplana arreglo en el orden zig-zag
def zigZag(frame):
    flat = np.zeros(64,dtype=frame.dtype)
    ind = 0
    for i in range(8):
        x = 0
        y = i
        while x-1 != i:
            flat[63 - ind] = frame[7-y,7-x]
            flat[ind] = frame[y,x]
            ind = ind + 1
            x = x + 1
            y = y - 1
    return flat

def izigZag(flat):
    frame = np.zeros((8,8),dtype=flat.dtype)
    ind = 0
    for i in range(8):
        x = 0
        y = i
        while x-1 != i:
            frame[7-y,7-x] = flat[63 - ind]
            frame[y,x] = flat[ind]
            ind = ind + 1
            x = x + 1
            y = y - 1
    return frame

def runLenght(arr):
    a = 0
    ind = 0
    n = np.zeros(64*2,dtype=arr.dtype)
    while a < arr.size:
        b = a
        while b < arr.size and arr[a] == arr[b]:
            b = b + 1
        n[ind] = arr[a]
        n[ind + 1] = b - a
        ind = ind + 2
        a = b
    return n[0:ind]

def irunLenght(arr):
    out = np.zeros(sum(arr[1::2]),dtype=arr.dtype)
    ind = 0
    for i in range(0,arr.size,2):
        out[ind:ind+arr[i+1]] = arr[i]
        ind = ind+arr[i+1]
    return out
