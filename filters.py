import numpy as np
import cv2 as cv
import scipy.signal as signal

def filtradoMedia(frame):
    threshold = 30
    frame = frame.flatten().astype('int16')
    shiftUp= np.roll(frame, -1)
    shiftDown = np.roll(frame, 1)
    shiftLeft= np.roll(frame, -848)
    shiftRight = np.roll(frame, 848)
    resta1 = np.abs(frame - shiftUp)
    resta2 = np.abs(frame - shiftDown)
    resta3 = np.abs(frame - shiftLeft)
    resta4 = np.abs(frame - shiftRight)
    media = (shiftUp+shiftDown+shiftLeft+shiftRight)/4 #np.zeros(len(frame))#
    isItNoise = ((resta1 > threshold) & (resta2 > threshold)) | ((resta3 > threshold) & (resta4 > threshold))
    frame = np.where(isItNoise,media,frame)
    return frame.reshape(480,848)


# Filtro IIR
b1, a1 = signal.iirnotch(2.0455455455455454, 5.047047047047047, 425*2)
b2, a2 = signal.iirnotch(20.02892892892893,70.63163163163163, 240*2)

def filtradoIIR(frame):
    res1 = signal.lfilter(b1,a1,frame.flatten()).reshape(480, 848)
    return signal.lfilter(b2,a2,res1.flatten('F')).reshape((480, 848), order='F')


# Filtro FIR
FIRCoefs = np.loadtxt("FIRCoefs.txt").astype('float32')

def filtradoFIR(frame):
    for i in range(480):
        frame[i,:] = signal.correlate(frame[i,:],FIRCoefs,mode="same")
    return frame.flatten('F').reshape(848,480).T

# Filtro FFT
def filtradoFFT(img):
    img_dft = cv.dft(img.astype('float32'),flags=cv.DFT_COMPLEX_OUTPUT)
    mag_dft, phase_dft = cv.cartToPolar(img_dft[:,:,0],img_dft[:,:,1])
    mag_dft[12:26,2] = 0
    mag_dft[452:266,846] = 0
    inv = mag_dft*np.exp(1j*phase_dft)
    img_dft[:,:,0] = np.real(inv)
    img_dft[:,:,1] = np.imag(inv)
    img = cv.idft(img_dft,flags=cv.DFT_REAL_OUTPUT)
    img = 255*img/max(img.flatten())
    return img.flatten('F').reshape(848,480).T.astype('uint8')
