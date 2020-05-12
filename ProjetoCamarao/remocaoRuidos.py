import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

i = 0
try:
    if not os.path.exists('ImagensSemRuidos'):
        os.makedirs('ImagensSemRuidos')
except OSError:
    print ('Error: Creating directory of ImagensSemRuidos')

while(i < 500):
    imagem = cv2.imread('ImagensRecortadas/recorte' + str(i) +'.jpg')
    img2gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # Remove hair with opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img2gray,cv2.MORPH_OPEN,kernel)
    #opening = cv2.morphologyEx(img2gray,cv2.MORPH_CLOSE,kernel)
    cv2.imwrite("ImagensSemRuidos/ruido" + str(i) + ".jpg", opening) #salva no disco
    name = './ImagensSemRuidos/ruido' + str(i) + '.jpg'
    print ('Removendo ruidos...' + name)
    i = i + 1 
