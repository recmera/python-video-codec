import numpy as np
import cv2 as cv

# Arreglo de índices para matrices de 8x8
x8 = np.arange(0,848,8,dtype='int')
y8 = np.arange(0,480,8,dtype='int')

# Matriz de cuantización JPEG
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
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

# Obtiene transformada de coseno cada matriz de 8x8 y les aplica cuantización
# Almacena la salida cuantizada
cArr = np.zeros(480*848 + 60*106,dtype='int8')
def dctAndCuantize(frame):
    frame = frame.astype('float32')
    ind = 0
    # Procesa cada cuadro de 8x8
    for x in x8:
        for y in y8:
            # Calcula dct y cuantiza
            f_dct = np.round(cv.dct(frame[y:y+8,x:x+8]-127)/Q)

            # Convierte a arreglo 1D en orden zigZag
            zZag = zigZag(f_dct)

            # Aplica run length
            rLenght = runLenght(zZag)

            # Los almacena en el arreglo de salida
            cArr[ind:ind+rLenght.size] = rLenght

            ind = ind + rLenght.size

    return cArr[0:ind].astype('int8')

# Obtiene transformada inversa
def idctAndCuantize(data):

    # Aplica runLenght inverso
    data = irunLenght(data)

    frame = np.zeros((480, 848),dtype='uint8')
    ind = 0

    for x in x8:
        for y in y8:
            # Convierte arreglo 1D a 2D
            dct = izigZag(data[ind:ind+64]).astype('float32')

            # Calcula la idct
            frame[y:y+8,x:x+8] = (cv.idct(dct*Q)+127).astype('uint8')

            ind = ind + 64
    return frame

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
    n = np.zeros_like(arr)
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
