"""Cria json com informações das imagens necessárias para o treinamento da detectron2."""
import os
import cv2
import sys
import json
import fnmatch


def criaJSON(infos, input_dir):
    json_content = {}
    for info in infos:
        regions = {}
        cont_reject = 0;
        for ind, contour in enumerate(info[2]):
            area = cv2.contourArea(contour)
            if(area < 100):
                cont_reject += 1
                continue
            regions[str(ind - cont_reject)] = {"shape_attributes" :
                                    {
                                        "name" : "polygon",
                                        "all_points_x": [int(x) for x in contour[:,:,0][:,0]],
                                        "all_points_y": [int(y) for y in contour[:,:,1][:,0]],
                                    },
                                 "region_attributes": {}
                                }
        arquivo = {
           "fileref" : "",
           "size": info[1],
           "filename": info[0],
           "base64_img_data": "",
           "file_attributes": {},
           "regions": regions
        }
        json_content[info[0] + str(info[1])] = arquivo
    with open(input_dir + "via_region_data.json", 'w') as outfile:
        # Melhor visualizacao
        # json.dump(json_content, outfile, indent=4)
        json.dump(json_content, outfile)
    return


def main():
    if(len(sys.argv) != 2):
        print("USO: python {} diretorio_imagens\n".format(sys.argv[0]))
        print("O script vai compilar as informações de todas as imagens jpg do diretorio diretorio_imagens.")
        print("O diretorio diretorio_imagens deve conter a imagem correspondente ao background no arquivo bkg.bmp")
        print("As anotacoes serao salvas no arquivo via_region_data.json no diretorio diretorio_imagens")
        return
    input_dir = sys.argv[1]
    if(input_dir[-1] != '/'):
        input_dir += '/'
    input_files = os.listdir(input_dir)
    input_files.sort()
    bg_img = cv2.imread(input_dir + "bkd.bmp")
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    infos = []
    for filename in input_files:
        if fnmatch.fnmatch(filename, "*.jpg"):
            img = cv2.cvtColor(cv2.imread(input_dir + filename), cv2.COLOR_BGR2GRAY)
            file_size = os.stat(input_dir + filename).st_size
            img_diff = cv2.subtract(bg_img, img)
            _, img_bw = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            infos.append([filename, file_size, contours])
    criaJSON(infos, input_dir)
    # criaXML(filename, input_dir, filename[:-3] + "xml", output_dir + "xmls/", img_rgb.shape, contours)


if __name__ == "__main__":
    main()