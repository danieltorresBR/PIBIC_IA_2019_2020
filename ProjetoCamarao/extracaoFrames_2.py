"""Extrai os frames do video. === COM BACKGROUND"""
import numpy as np
import cv2
import time
import os

try:
    if not os.path.exists('imagensExtraidas'):
        os.makedirs('imagensExtraidas')
except OSError:
    print ('Error: Creating directory of imagens')

video_name = 'camarao.mp4'

cap = cv2.VideoCapture(video_name)
print("Calculando background\n")

cont = 0
startTime = time.time()
while(cap.isOpened() is True):
    cont += 1
    ret, original = cap.read()
    if(ret is False):
        break
    #cv2.imwrite('imagensExtraidas/{:>05}.jpg'.format(cont), original)
    # Saves image of the current frame in jpg file
    name = './imagensExtraidas/frame' + str(cont) + '.jpg'
    cv2.imwrite(name, original)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if(cont == 1):
        x, y = gray.shape
        img_soma = np.zeros((x, y), dtype=float)
    if(cont % 10 == 0):
        print("Criando... {} ({:5.2f} fps)".format(cont, 10.0 / (time.time() - startTime)))
        #print ('Criando...' + name)
        startTime = time.time()
    if(cont == 500):
         break

    img_soma = img_soma + gray
img_bg = (img_soma / cont).astype(np.uint8)
cap.release()
cv2.imwrite('imagensExtraidas/bkd.bmp', img_bg)
