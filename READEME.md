Desenvolvimento de Um Sistema Para Rastreamento de Cardumes de Peixe Zebrafish

>> Script

Antes de tudo é necessário extrair os frames do vídeo e o background.
O script extractFrame.py faz isso.

Para criar um dataset é preciso:
1 - Criar um diretório contendo dois outros diretórios: train e val;
         - dataset
             | train
             | val
2 - Selecione frames que você deseja usar no dataset e coloque nesses diretórios;
3 - rodar o script segmenta.py em cada um dos diretórios internos do dataset (train e val).

Para treinar a rede você precisa:
1 - Copiar o diretório do dataset para dentro do diretório do detectron2;
2 - Configurar o script treinar.py com o diretório do dataset
(variável dataset_name) e a classe (variável class_name);
3 - rodar o script treinar.py.

extractFrames.py: extrai todos os frames de um vídeo e salva no
diretório Frames (no formato #####.jpg)
segmenta.py: recebe um diretório com várias imagens (em cores e no
formato jpg) e o background (preto e branco e no formato bmp) das
imagens e cria um arquivo json no formato de entrada da rede
detectron2
treinar.py: treina a rede
detectar.py: usa a rede treinada em todos os arquivos no diretório do
dataset/frames. A idéia é rodar em todos os frames do vídeo após o
treinamento e gerar um vídeo.
frame800.zip: é um dataset pronto com 20 frames de treino e 10 de
avaliação. Os frames foram reduzidos para 800x450px porque a rede
velha morria se rodasse com o frame original. Não testei com frames
fullhd nessa ainda:.

===============================================================================

Dataset Detectron 2

Detectron 2:
https://github.com/facebookresearch/detectron2

Informações sobre o formato:
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/

Ferramentas para fazer anotação:

https://github.com/jsbroks/coco-annotator
https://github.com/wkentaro/labelme
https://github.com/tzutalin/labelImg

Vídeo:
https://drive.google.com/file/d/1oP_DcUUlFexIOT6w-jAWp3yI3YqhBPec/view?usp=sharing

Script para extrair os frames em anexo. Se conseguir algum material da subtração de imagens envio depois.

===============================================================================

>> Tutorial Mask-RCNN

O tutorial que segui foi esse
https://github.com/vijendra1125/Custom-Mask-RCNN-using-Tensorfow-Object-detection-API

Mas como sempre não é perfeito, dei umas olhadas nesse outro aqui também

https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4

===============================================================================

>> LINKS EXTERNOS:

PESQUISAS:
http://www.dominiopublico.gov.br/pesquisa/PesquisaObraForm.jsp

PAULISTINHA:
https://www.redezebrafish.com.br/


CAMARÃO
http://g1.globo.com/espirito-santo/jornal-do-campo/videos/v laboratorio-de-pos-larva-de-camarao-comeca-a-funcionar-em-governador-lindenberg-es/4118824/

https://www.youtube.com/watch?v=QtYy0kFgdpQ

https://www.youtube.com/watch?v=LFKvxb8CS1w

https://www.facebook.com/macrolarves/videos/1788186971451383/?__so__=permalink&__rv__=related_videos

http://g1.globo.com/espirito-santo/estv-2edicao/videos/v/laboratorio-de-pos-larva-de-camarao-comeca-a-funcionar-no-sul-do-es/4096143/

https://www.youtube.com/watch?v=fLb9o6OhhHM

==========================================================================
FERRAMENTAS


https://github.com/abreheret/PixelAnnotationTool/releases