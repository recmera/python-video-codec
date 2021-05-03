import numpy as np
import cv2 as cv
import ast

# Nuestras librerías
from filters import filtradoIIR,filtradoFIR,filtradoFFT,filtradoMedia
from codec import dctAndCuantize,idctAndCuantize,interFrameDiff,huffman,ihuffman,bit2Str,createDictionary,detectMovement,applyMovement,genQ,MSE,applyMovementEmisor
from settings import conf


fps = 30.0 - (30.0 / (conf['skipFrameAfter']+1.0))

dendograma = None
idendograma = None

def loadDendogram():
    global dendograma,idendograma
    with open(conf['dendrograma'],"rb") as f:
        dendograma = ast.literal_eval(f.read().decode())
        idendograma = {codigo:int(simbolo) for simbolo,codigo in dendograma.items()}

fCount = 0.0
def denoise(frame):
    # Elimina ruido sinusoidal
    if conf['filtroSeno'] == 'IIR':
        frame = filtradoIIR(frame)
    elif conf['filtroSeno'] == 'FIR':
        frame = filtradoFIR(frame)
    else:
        frame = filtradoFFT(frame)

    # Eliminar ruido de sal y pimienta
    return filtradoMedia(frame)

frameSkip = 0
counts = np.zeros((256,899))
firstFrame = True
EMCount = 0.0
EMSum = 0.0
def code(framex):

    # Salta frame
    global frameSkip,firstFrame,dendograma,idendograma,EMCount,EMSum
    if frameSkip == conf['skipFrameAfter']:
        frameSkip = 0
        return bytearray(1)
    frameSkip = frameSkip + 1

    # Calcula movimiento de la cámara
    translations = detectMovement(framex)

    # Aplica traslación al frame anterior
    applyMovementEmisor(translations)

    # Evalua los frames que cambiaron
    changes,frame,QPercent = interFrameDiff(framex,translations)

    #chan = changes

    # Genera matriz Q
    Q = genQ(QPercent)

    # DCT y Cuantización + RunLength o Truncamiento
    data = dctAndCuantize(changes,frame,Q)

    # Transforma booleanos a bits
    changes = np.frombuffer(bytes(memoryview(np.packbits(changes.astype('bool').flatten()))),dtype='int8')





    # Aplica translaciones al frame anterior
    #applyMovement(chan,translations)
    # Realiza proceso inverso
    #dcts = data
    #newFrame = idctAndCuantize(chan,dcts,Q)
    #EMSum = EMSum + MSE(newFrame,framex)
    #EMCount = EMCount + 1
    #print("MSE:",EMSum/EMCount)
    #return 0

    out = np.concatenate((changes,translations,np.int8([QPercent]), data)).astype('int8')

    #for i in range(255):
        #counts[i,frameSkip] = np.count_nonzero(out == i-128)

    #frameSkip = frameSkip + 1
    #if frameSkip == 899:
        #np.savetxt('Histograma1000.txt',counts)

    #return framex.astype('uint8')

    if conf['dynamicDendrogram'] == True:
        dic = createDictionary(out)
        coded = huffman(out,dic)
        stringDic = str(dic).encode()
        dicLen = np.array([len(stringDic)],dtype="uint16")
        return bytes(dicLen) + bytes(stringDic) + bytes(coded)
    else:
        if firstFrame == True:
            loadDendogram()
            firstFrame = False
        coded = huffman(out,dendograma)
        return coded



reconstructedFrame = 0
kbps = 0.0
def decode(message):
    #return np.frombuffer(bytes(memoryview(message)),dtype='uint8').reshape(480,848)
    global bitsAvg,fCount,reconstructedFrame,firstFrame,dendograma,idendograma,fps,kbps

    bts = bytes(memoryview(message))

    if len(bts) == 1:
        return reconstructedFrame

    if conf['dynamicDendrogram'] == True:
        # Obtiene el largo del diccionario
        dicLen = np.frombuffer(bts[:2],dtype='uint16')[0]

        # Obtiene diccionario inverso
        dendograma = ast.literal_eval(bts[2:2+dicLen].decode())
        idendograma = {codigo:int(simbolo) for simbolo,codigo in dendograma.items()}

        # Convierte a string de bits
        data = ihuffman(bit2Str(bts[2+dicLen:]),idendograma)

        dicBitsAvg = (dicLen*8.0*fps)/1000.0
        print("Dic Bits Avg:",dicBitsAvg)
    else:
        if firstFrame == True:
            loadDendogram()
            firstFrame = False
        data = ihuffman(bit2Str(bts),idendograma)


    # Obtiene la matriz de cambios
    changes = np.unpackbits(np.frombuffer(data[0:795], dtype='uint8')-128, axis=None)[:6360].reshape((60,106)).astype(np.bool)

    # Obtiene traslaciones de la cámara
    translations = np.frombuffer(data[795:795+8],dtype='uint8').astype('float32') - 128

    # Aplica translaciones al frame anterior
    applyMovement(changes,translations)

    # Obtiene matriz de cuantización
    QPercent = np.frombuffer(data[795+8:795+9],dtype='uint8').astype('int16')[0] -128
    Q = genQ(QPercent)

    # Obiene los cuadros 8x8
    dcts = np.frombuffer(data[795+9:],dtype='uint8').astype('int16')-128

    # Realiza proceso inverso
    frame = idctAndCuantize(changes,dcts,Q)

    # Métricas
    if fCount == fps-1:
        print("Kbps:",int((kbps*8.0)/1000.0),"FPS:",fps)
        kbps = 0
        fCount = 0
    else:
        fCount = fCount + 1.0
        kbps = kbps + len(bts)


    reconstructedFrame = frame
    return frame
