import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
import matplotlib.pyplot as plt

def detect(img,model, confthre,nmsthre,imgsize):
    
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    img = Variable(img.type(torch.cuda.FloatTensor))
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    if outputs[0] is None:
        print("No Objects Deteted!!")
        return

    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

        cls_id = coco_class_ids[int(cls_pred)]
        if cls_id==1:
            #print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            #print('\t+ Label: %s, Conf: %.5f' %
            #    (coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)

    return bboxes

    #return None    