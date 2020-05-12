import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('camarao.mp4')

try:
    if not os.path.exists('imagensExtraidas'):
        os.makedirs('imagensExtraidas')
except OSError:
    print ('Error: Creating directory of imagens')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './imagensExtraidas/frame' + str(currentFrame) + '.jpg'
    print ('Criando...' + name)
    cv2.imwrite(name, frame)

    if(currentFrame == 500):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()