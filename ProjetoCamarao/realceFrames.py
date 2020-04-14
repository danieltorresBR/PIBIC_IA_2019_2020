import numpy as np
import cv2
import os

i = 0
try:
    if not os.path.exists('ImagensRealcadas'):
        os.makedirs('ImagensRealcadas')
except OSError:
    print ('Error: Creating directory of ImagensRealcadas')


while(i < 500):
    imagem = cv2.imread('ImagensRecortadas/recorte' + str(i) +'.jpg')
    #imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(imagem, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(imagem, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv2.bitwise_or(sobelX, sobelY)

    resultado = np.vstack([
        np.hstack([imagem, sobelX]),
        np.hstack([sobelY, sobel])
    ])

    cv2.imwrite("ImagensRealcadas/realce" + str(i) + ".jpg", sobel) #salva no disco
    name = './ImagensRecortadas/recorte' + str(i) +'.jpg'
    print ('Realçando...' + name)
    i = i + 1 