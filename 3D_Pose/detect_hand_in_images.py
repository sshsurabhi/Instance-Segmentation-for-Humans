from __future__ import division

from models import Darknet
import os
import sys
import time
import datetime
import argparse
import cv2 
from PIL import Image
import numpy as  np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.utils import *

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_199.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### =============================================================================== Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    #### =============================================================================== LOad MOdel
    model.load_state_dict(torch.load(opt.weights_path))
    #### =============================================================================== Set in evaluation mode
    model.eval()  

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    #### ================================================================================ Process image
    img = transforms.ToTensor()(Image.open('data/samples/1_1_frame_0011.jpg'))
    img, _ = pad_to_square(img, 0)
    img = resize(img, 416)
    img.unsqueeze_(0)
    prev_time = time.time()
    img = Variable(img.type(Tensor))

    with torch.no_grad():
        detections = model(img)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print(inference_time)
    detect = detections[0].cpu().data.numpy()
    print(detect)
