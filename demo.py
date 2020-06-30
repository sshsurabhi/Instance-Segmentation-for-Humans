from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from image import get_affine_transform, transform_preds
from evaluat import get_preds
#import pyrealsense2 as rs
import math, time
################################################### yolov3 person det 
import argparse
import yaml
from torch.autograd import Variable
from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from person_det import detect
from mask_function import mask_plot

with open('config/yolov3_default.cfg', 'r') as f:
    cfg = yaml.load(f)

imgsize_y = cfg['TEST']['IMGSIZE']
model_Y = YOLOv3(cfg['MODEL'])

confthre = cfg['TEST']['CONFTHRE'] 
nmsthre = cfg['TEST']['NMSTHRE']

model_Y.cuda(0)
parse_yolo_weights(model_Y, 'weights/yolov3.weights')
model_Y.eval()
####################################################
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

mpii_edges = [[0, 1], [0, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [6, 12], [5, 11], [11, 12], [12, 14], [14, 16], [11, 13], [13, 15]]

def show_2d(pts, c, edges,orig_image,min_x,min_y):
  for i in range(len(pts)):
    points = pts[i]
    num_joints = points.shape[0]
    points = ((points.reshape(num_joints, -1))).astype(np.int32)
    #for j in range(num_joints):
    #  cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
    for e in edges:
      if points[e].min() > 0:
        cv2.line(orig_image, (points[e[0], 0]+min_x[i], points[e[0], 1]+min_y[i]),
                      (points[e[1], 0]+min_x[i], points[e[1], 1]+min_y[i]), c, 5)
  return orig_image

def draw_fill(imag, pts_u, pts_l):
  cv2.fillPoly(imag,[pts_u],(255,0,0))
  cv2.fillPoly(imag,[pts_l],(255,0,0))
  return imag


def demo_image(imgs, model, opt, orig_image,min_x,min_y):
  #print(len(imgs))
  preds =[]
  for i in range(len(imgs)):
    image = imgs[i]
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
        c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                          flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(opt.device)
    out = model(inp)[-1]
    pred = get_preds(out['hm'].detach().cpu().numpy())[0]
    pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
    preds.append(pred)
    #print(np.array(pred))

    #print(points_2d)
    pts_upper, pts_lower = mask_plot(image,np.array(pred),mpii_edges)
    pts_upper = np.array([pts_upper[:,0] + min_x[i] , pts_upper[:,1] + min_y[i]]).T
    pts_lower = np.array([pts_lower[:,0] + min_x[i], pts_lower[:,1] + min_y[i]]).T
    img = draw_fill(orig_image,pts_upper,pts_lower) 
  print('Predictions: '+str(len(preds)))
  #img = show_2d(preds,(0,0,255),mpii_edges,orig_image,min_x,min_y)
  
  
  return img

def main(opt):
  opt.heads['depth'] = opt.num_output
  if opt.load_model == '':
    opt.load_model = 'model_best_keypoints.pth'
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  else:
    opt.device = torch.device('cpu')
  
  model, _, _ = create_model(opt)
  model = model.to(opt.device)
  model.eval()
  count = 0
  
  files = os.listdir('images/')
  start = time.time()
  #Convert images to numpy arrays
  for i in range(len(files)):
    color_image = cv2.imread('images/'+files[i])
    orig=color_image.copy()
    boxes = detect(color_image,model_Y,confthre,nmsthre,imgsize_y)
    cropped_images=[]
    y_mins =[]
    x_mins =[]
    for i in range(len(boxes)):
      cropped_image = color_image[int(boxes[i][0]):int(boxes[i][2]),int(boxes[i][1]):int(boxes[i][3])]
      
      y_mins.append(int(boxes[i][0]))
      x_mins.append(int(boxes[i][1]))
      cropped_images.append(cropped_image)
    print('Number of persons: '+str(len(cropped_images)))

    if len(cropped_images) > 0:
      show_img = demo_image(cropped_images,model,opt,color_image,x_mins,y_mins)
    
    #print(1.0/(time.time()-start))
    #cv2.imwrite('color_image2.png',orig)
    #cv2.imwrite('skeleton2.png',show_img)
    cv2.imshow('image',show_img)
    key = cv2.waitKey(0)
    if key == ord('q'):
      break
    

if __name__ == '__main__':

  opt = opts().parse()
  
  main(opt)
