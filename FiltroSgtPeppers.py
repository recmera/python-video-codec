import numpy as np

def filtradoMedia(frame):
    threshold = 30
    frame = frame.flatten('F').astype('int16')
    shiftUp= np.roll(frame, -1)
    shiftDown = np.roll(frame, 1)
    shiftLeft= np.roll(frame, -480)
    shiftRight = np.roll(frame, 480)
    resta1 = np.abs(frame - shiftUp)
    resta2 = np.abs(frame - shiftDown)
    resta3 = np.abs(frame - shiftLeft)
    resta4 = np.abs(frame - shiftRight)
    media = (shiftUp+shiftDown+shiftLeft+shiftRight)/4 #np.zeros(len(frame))#
    isItNoise = ((resta1 > threshold) & (resta2 > threshold)) | ((resta3 > threshold) & (resta4 > threshold))
    frame = np.where(isItNoise,media,frame)
    return frame.reshape(848,480).T.astype('uint8')
