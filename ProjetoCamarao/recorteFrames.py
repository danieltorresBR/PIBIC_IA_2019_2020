import cv2
import os

i = 0
try:
    if not os.path.exists('ImagensRecortadas'):
        os.makedirs('ImagensRecortadas')
except OSError:
    print ('Error: Creating directory of ImagensRecortadas')

while(i < 500):
    imagem = cv2.imread('imagensExtraidas/frame' + str(i) +'.jpg')
    recorte = imagem[300:800, 300:800]
    #cv2.imshow("Recorte da imagem", recorte)
    cv2.imwrite("ImagensRecortadas/recorte" + str(i) + ".jpg", recorte) #salva no disco
    name = './ImagensRecortadas/recorte' + str(i) + '.jpg'
    print ('Recortando...' + name)
    i = i + 1 