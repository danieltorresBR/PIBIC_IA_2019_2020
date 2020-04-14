"""Extrai os frames do video."""
import numpy as np
import cv2
import time

video_name = "../../../Dados/input_videos/NovoAquario/NovoTeste1.mp4"

cap = cv2.VideoCapture(video_name)
print("Calculando background\n")

cont = 0
startTime = time.time()
while(cap.isOpened() is True):
    cont += 1
    ret, original = cap.read()
    if(ret is False):
        break
    cv2.imwrite("../../../Dados/input_images/NovoTeste1/{:>05}.jpg".format(cont), original)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if(cont == 1):
        x, y = gray.shape
        img_soma = np.zeros((x, y), dtype=float)
    if(cont % 10 == 0):
        print("{} ({:5.2f} fps)".format(cont, 10.0 / (time.time() - startTime)))
        startTime = time.time()
    # if(cont == 1450):
    #     break

    img_soma = img_soma + gray
img_bg = (img_soma / cont).astype(np.uint8)
cap.release()
cv2.imwrite("../../../Dados/input_images/NovoTeste1/bkd.bmp", img_bg)
