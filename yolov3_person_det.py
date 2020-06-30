import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
import matplotlib.pyplot as plt

def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    with open('config/yolov3_default.cfg', 'r') as f:
        cfg = yaml.load(f)

    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])

    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']

    img = cv2.imread('data/mpii_87.png')
    orig_img = img.copy()
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)

    if args.gpu >= 0:
        model.cuda(args.gpu)
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    parse_yolo_weights(model, 'weights/yolov3.weights')
    
    model.eval()

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
            print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            print('\t+ Label: %s, Conf: %.5f' %
                (coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            
            y_min = int(box[0])
            x_min = int(box[1])
            y_max = int(box[2])
            x_max = int(box[3])
            #cropped = orig_img[y_min:y_max,x_min:x_max]
            cv2.rectangle(orig_img,(x_min,y_min),(x_max,y_max),(0,0,255),2)
        cv2.imshow('img',orig_img)
        cv2.waitKey(0)
            #plt.show()


if __name__ == '__main__':
    main()
