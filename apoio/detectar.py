import os
import cv2
import fnmatch
# from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

input_dir = "frame800/frames/"
dataset_name = "frame800"


cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = (dataset_name + "_train",)
cfg.DATASETS.TEST = (dataset_name + "_val",)   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
print("WEIGHTS = " + os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


input_files = os.listdir(input_dir)
input_files.sort()
dataset_metadata = MetadataCatalog.get(dataset_name + "_train")
for filename in input_files:
    if fnmatch.fnmatch(filename, "*.jpg"):
	    im = cv2.imread(input_dir + filename)
	    outputs = predictor(im)
	    v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=1.0) # , instance_mode=ColorMode.IMAGE_BW)
	    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	    cv2.imwrite(input_dir + "saida/" + filename, v.get_image()[:, :, ::-1])
	    # cv2.imshow("janela", v.get_image()[:, :, ::-1])
	    # cv2.waitKey()