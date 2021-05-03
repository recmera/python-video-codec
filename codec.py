import numpy as np
import cv2 as cv
import heapq
from collections import Counter
from settings import conf


# Arreglo de índices para matrices de 8x8
x8 = np.arange(0,848,8,dtype='int')
y8 = np.arange(0,480,8,dtype='int')

def genQ(percent):
    # Matriz de cuantización JPEG
    Q = np.array([[16.0, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    if (percent < 50):
        S = 5000/percent
    else:
        S = 200 - 2*percent
    Q = np.floor((S*Q + 50) / 100);
    Q[Q == 0] = 1
    return Q



def lastNonZeroIndex(arr):
    for i in range(arr.size-1,-1,-1):
        if arr[i] != 0:
            return i+1
    return 1

def MSE(a,b):
    a = a.astype('float32')
    b = b.astype('float32')
    return ((a - b)**2).mean(axis=None)

def MSENorm(a,b):
    a = a.astype('float32')
    a = a/np.amax(a)
    b = b.astype('float32')
    b = b/np.amax(b)
    return ((a - b)**2).mean(axis=None)

# Codificación de fuente
def huffman(data,dendograma):
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

def ihuffman(data,idendograma):
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


QEntropies = np.loadtxt("QEntropias.txt").astype('float32')
peviusFrame = np.full((480, 848),255,dtype='uint8')
peviusFrameUnchanged = np.full((480, 848),255,dtype='uint8')
skipped = conf['interFrameSendAll']
def interFrameDiff(frame,translations):
    global skipped

    if skipped == conf['interFrameSendAll']:
        skipped = 0
        tresh = 1.5
    else:
        tresh = conf['interFrameThreshold']
        skipped = skipped + 1
    changes = np.zeros((60,106),dtype='bool')
    xx = 0
    a,b = -translations[:2]
    for x in x8:
        yy = 0
        for y in y8:

            #cond = (xx < 2 and a >=16) or (xx > 104 and a <= -16) or (yy < 2 and b >= 16) or (yy > 57 and b <= -16)
            mse = MSE(peviusFrame[y:y+8,x:x+8],frame[y:y+8,x:x+8])
            if changes[yy,xx] == 1 or mse > tresh:
                changes[yy,xx] = 1
                n = conf['interFrameNeighbors']
                if xx > n and xx < 106-n and yy > n and yy < 60 - n and mse*2.0 > tresh:
                    changes[yy-n:yy+n,xx-n:xx+n] = 1
            peviusFrameUnchanged[y:y+8,x:x+8] = frame[y:y+8,x:x+8]
            yy = yy + 1
        xx = xx + 1

    QPercent = conf['allQ']

    fps = 30.0 - (30.0 / (conf['skipFrameAfter']+1.0))

    cont = np.count_nonzero(changes)

    if cont < 3000:
        for i in range(0,81):
            if QEntropies[i,1]*cont*fps > conf['bandwidth']:
                QPercent = i+1-5
                break
        if QPercent > conf['maxQ']:
            QPercent = conf['maxQ']
        elif QPercent < conf['minQ']:
            QPercent = conf['minQ']

    print('Cuadros enviados:',cont,' de ',60*106,"Q:",QPercent)
    return changes,frame, QPercent

def detectMovement(frame):
    translations = np.zeros(8,dtype='int8')
    r = 18
    s = 40
    limits = np.array([[0,240,0,424],[0,240,424,848],[240,480,424,848],[240,480,0,424]]) #UL UR DR DL
    w = 848.0
    h = 480
    ind = 0
    for l in limits:
        minMSE = 10e16
        for a in range(-r,r+1):
            for b in range(-r,r+1):
                pre = peviusFrameUnchanged[ l[0]+r+b : l[1]-r+b:s, l[2]+r+a :l[3]-r+a:s]
                cur = frame[ l[0]+r: l[1]-r:s,l[2]+r:l[3]-r:s]
                currentMSE = MSE(pre,cur)
                if currentMSE < minMSE:
                    minMSE = currentMSE
                    translations[ind] = a
                    translations[ind+1] = b
        ind = ind + 2
    return translations

frac = 12.0
def applyMovementEmisor(translations):
    global frac
    rx = 848/frac
    ry = 480/frac
    src = np.float32([[0+rx,0+ry],[848-rx,0+ry],[0+rx,480-ry],[848-rx,480-ry]])
    dst = src - translations.reshape(4,2)

    M = cv.getPerspectiveTransform(src,dst)
    peviusFrame[:,:] = cv.warpPerspective(src=peviusFrameUnchanged,M=M,dsize=(848,480),borderMode=cv.BORDER_CONSTANT,flags=cv.INTER_NEAREST)


# Obtiene transformada de coseno cada matriz de 8x8 y les aplica cuantización
# Almacena la salida cuantizada
cArr = np.zeros(480*848 + 60*106,dtype='int8')
def dctAndCuantize(changes,frame,Q):
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

                peviusFrame[y:y+8,x:x+8] = frame[y:y+8,x:x+8]
                # Convierte a arreglo 1D en orden zigZag
                zZag = zigZag(f_dct)

                if conf['postQuantization'] == 'Truncate':
                    # Aplica truncamiento
                    lastIndex = lastNonZeroIndex(zZag)

                    # Los almacena en el arreglo de salida
                    cArr[ind] = lastIndex
                    cArr[ind+1:ind+lastIndex+1] = zZag[:lastIndex]

                    ind = ind + 1 + lastIndex
                else:
                    # Aplica run length
                    runLen = runLength(zZag)

                    L = len(runLen)

                    cArr[ind:ind+L] = runLen
                    ind = ind + L
            yy = yy + 1
        xx = xx + 1

    return cArr[0:ind].astype('int8')


# Obtiene transformada inversa
reconstructedFrame = np.zeros((480, 848),dtype='float32')
def idctAndCuantize(changes,data,Q):

    # Aplica runLength inverso
    if conf['postQuantization'] == 'runLength':
        data = irunLength(data)

    ind = 0
    xx = 0
    for x in x8:
        yy = 0
        for y in y8:
            if changes[yy,xx] == 1:

                if conf['postQuantization'] == 'Truncate':
                    lastIndex = data[ind]
                    flat = np.zeros(64)
                    flat[:lastIndex] = data[ind+1:ind+1+lastIndex]
                    flat[lastIndex:64] = 0
                    ind = ind + 1 + lastIndex

                else:
                    flat = data[ind:ind+64]
                    ind = ind + 64

                # Convierte arreglo 1D a 2D
                dct = izigZag(flat).astype('float32')

                B = conf['interFrameBlur']
                r = conf['interFrameBlurRadius']
                if r != 0:
                    # L
                    if xx != 0 and changes[yy,xx-1] == 0:
                        for d in range(1,r+1):
                            #reconstructedFrame[y:y+8,x-d] = 0
                            reconstructedFrame[y:y+8,x-d] = reconstructedFrame[y:y+8,x-d]*(1.0-B) + reconstructedFrame[y:y+8,x-d+1]*B
                    # Up
                    if yy != 0 and changes[yy-1,xx] == 0:
                        for d in range(1,r+1):
                            #reconstructedFrame[y-d,x:x+8] = 0
                            reconstructedFrame[y-d,x:x+8] = reconstructedFrame[y-d,x:x+8]*(1.0-B) + reconstructedFrame[y-d+1,x:x+8]*B
                    # R
                    if xx != 105 and changes[yy,xx+1] == 0:
                        for d in range(1,r+1):
                            #reconstructedFrame[y:y+8,x+7+d] = 0
                            reconstructedFrame[y:y+8,x+7+d] = reconstructedFrame[y:y+8,x+7+d]*(1.0-B) + reconstructedFrame[y:y+8,x+7+d-1]*B
                    # Down
                    if yy != 59 and changes[yy+1,xx] == 0:
                        for d in range(1,r+1):
                            #reconstructedFrame[y+7+d,x:x+8] = 0
                            reconstructedFrame[y+7+d,x:x+8] = reconstructedFrame[y+d+7,x:x+8]*(1.0-B) + reconstructedFrame[y+7+d-1,x:x+8]*B
                    # Up L
                    if yy != 0 and xx != 0 and changes[yy-1,xx-1] == 0:
                        #reconstructedFrame[y-4:y,x-4:x] = 0
                        for i in range(1,r+1):
                            for j in range(1,r+1):
                                if i < j:
                                    reconstructedFrame[y-j,x-i] = reconstructedFrame[y-j,x-i]*(1.0-B) + reconstructedFrame[y-j+1,x-i]*B
                                if i == j:
                                    reconstructedFrame[y-j,x-i] = reconstructedFrame[y-j,x-i]*(1.0-B) + reconstructedFrame[y-j+1,x-i+1]*B
                                else:
                                    reconstructedFrame[y-j,x-i] = reconstructedFrame[y-j,x-i]*(1.0-B) + reconstructedFrame[y-j,x-i+1]*B

                    # Up R
                    if yy != 0 and xx != 105 and changes[yy-1,xx+1] == 0:
                        #reconstructedFrame[y-4:y,x+8:x+8+4] = 0
                        for i in range(1,r+1):
                            for j in range(1,r+1):
                                if i < j:
                                    reconstructedFrame[y-j,x+7+i] = reconstructedFrame[y-j,x+7+i]*(1.0-B) + reconstructedFrame[y-j+1,x+7+i]*B
                                if i == j:
                                    reconstructedFrame[y-j,x+7+i] = reconstructedFrame[y-j,x+7+i]*(1.0-B) + reconstructedFrame[y-j+1,x+7+i-1]*B
                                else:
                                    reconstructedFrame[y-j,x+7+i] = reconstructedFrame[y-j,x+7+i]*(1.0-B) + reconstructedFrame[y-j,x+7+i-1]*B

                    # Down L
                    if yy != 59 and xx != 0 and changes[yy+1,xx-1] == 0:
                        #reconstructedFrame[y+8:y+8+4,x-4:x] = 0
                        for i in range(1,r+1):
                            for j in range(1,r+1):
                                if i < j:
                                    reconstructedFrame[y+7+j,x-i] = reconstructedFrame[y+7+j,x-i]*(1.0-B) + reconstructedFrame[y+7+j-1,x-i]*B
                                if i == j:
                                    reconstructedFrame[y+7+j,x-i] = reconstructedFrame[y+7+j,x-i]*(1.0-B) + reconstructedFrame[y+7+j-1,x-i+1]*B
                                else:
                                    reconstructedFrame[y+7+j,x-i] = reconstructedFrame[y+7+j,x-i]*(1.0-B) + reconstructedFrame[y+7+j,x-i+1]*B
                    # Down R
                    if yy != 59 and xx != 105 and changes[yy+1,xx+1] == 0:
                        #reconstructedFrame[y+8:y+8+4,x+8:x+8+4] = 0
                        for i in range(1,r+1):
                            for j in range(1,r+1):
                                if i < j:
                                    reconstructedFrame[y+7+j,x+7+i] = reconstructedFrame[y+7+j,x+7+i]*(1.0-B) + reconstructedFrame[y+7+j-1,x+7+i]*B
                                if i == j:
                                    reconstructedFrame[y+7+j,x+7+i] = reconstructedFrame[y+7+j,x+7+i]*(1.0-B) + reconstructedFrame[y+7+j-1,x+7+i-1]*B
                                else:
                                    reconstructedFrame[y+7+j,x+7+i] = reconstructedFrame[y+7+j,x+7+i]*(1.0-B) + reconstructedFrame[y+7+j,x+7+i-1]*B



                # Calcula la idct
                reconstructedFrame[y:y+8,x:x+8] = (cv.idct(dct*Q)+128)

            yy = yy + 1
        xx = xx + 1
    return reconstructedFrame.astype('uint8')

def applyMovement(changes,translations):
    global frac
    rx = 848/frac
    ry = 480/frac
    src = np.float32([[0+rx,0+ry],[848-rx,0+ry],[0+rx,480-ry],[848-rx,480-ry]])
    dst = src - translations.reshape(4,2)

    M = cv.getPerspectiveTransform(src,dst)
    reconstructedFrame[:,:] = cv.warpPerspective(src=reconstructedFrame,M=M,dsize=(848,480),borderMode=cv.BORDER_REPLICATE,flags=cv.INTER_NEAREST)

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

def runLength(arr):
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

def irunLength(arr):
    out = np.zeros(sum(arr[1::2]),dtype=arr.dtype)
    ind = 0
    for i in range(0,arr.size,2):
        out[ind:ind+arr[i+1]] = arr[i]
        ind = ind+arr[i+1]
    return out

def createDictionary(data):
    dendograma = [[frecuencia/256.0,[simbolo,""]] for simbolo,frecuencia in Counter(data).items()]
    heapq.heapify(dendograma)
    while len(dendograma) > 1:
        lo = heapq.heappop(dendograma)
        hi = heapq.heappop(dendograma)
        for codigo in lo[1:]:
            codigo[1] = '0' + codigo[1]
        for codigo in hi[1:]:
            codigo[1] = '1' + codigo[1]
        heapq.heappush(dendograma,[lo[0] + hi[0]] + lo[1:] + hi[1:])

    dendograma = sorted(heapq.heappop(dendograma)[1:])
    dendograma = {simbolo:codigo for simbolo,codigo in dendograma}
    return dendograma
