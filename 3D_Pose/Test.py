import yaml, time
import torch
import argparse
import numpy as np
import json, csv, cv2, os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from utils.utils import non_max_suppression, rescale_boxes
import datetime

bone_2d= [[0,1],[0,4],[0,7], [0,10], [0,13], [1,2], [2,3], [3,17], [4,5],[5,6], [6,18], [10,11], [11,12],[12,19], [7,8], [8,9], [9,20], [13,14],[14,15],[15,16]]

def plot_2d(x_,y_,act_x,act_y):
    for bone in bone_2d:
        keyp_x = x_
        keyp_y = y_
        plt.plot([keyp_x[bone[0]],keyp_x[bone[1]]],[keyp_y[bone[0]],keyp_y[bone[1]]],c='r',linewidth=3.0)
        #plt.plot([act_x[bone[0]],act_x[bone[1]]],[act_y[bone[0]],act_y[bone[1]]],c='g',linewidth=2.0)
        plt.legend(['Predicted','Actual'],fontsize=20)
        #plt.savefig('Results/image_1.png')

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

class Tester():
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #### ===================================================== Model 1
        from model_RegNet import EncoderDecoder
        self.model_RegNet = EncoderDecoder().to(self.device)
        #self.model_RegNet = self.model_RegNet.to(device)
        self.weight_file = config['weights_RegNet']
        checkpoint = torch.load(self.weight_file,map_location='cuda:0')
        self.model_RegNet.load_state_dict(checkpoint['model'])
        self.model_RegNet.eval()
        print('===================== Regression Model Loaded==================')
        #### ===================================================== Model 2
        from model_DarkNet import Darknet
        self.model_def = config['model_def']
        self.weights_path = config['weights_DarkNet']
        self.img_size = 416
        self.model_DarkNet = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        self.model_DarkNet.load_state_dict(torch.load(self.weights_path))
        self.model_DarkNet.eval()  
        print('======================= YOLO Model Loaded ========================')
        self.classes = self.load_classes(config['class_path'])
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        ##### ======================================================== Transformation
        self.transform_image = transforms.Compose([
                        transforms.Resize((128, 128)), 
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def load_classes(self,path):
        fp = open(path, "r")
        self.names = fp.read().split("\n")[:-1]

    def test(self, image, orig_cv):
        img = transforms.ToTensor()(image)
        img, _ = pad_to_square(img, 0)
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)
        img.unsqueeze_(0)
        prev_time = time.time()
        img = Variable(img.type(self.Tensor))

        with torch.no_grad():
            detections = self.model_DarkNet(img)
            detections = non_max_suppression(detections, config['conf_thres'], config['nms_thres'])
        
        for detection in detections:
            img = orig_cv
            if detection is not None:
                detection = rescale_boxes(detection,self.img_size, img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                    if conf > 0.99:
                        cropped_image = image.crop(box=(x1.item(),y1.item(),x2.item(),y2.item()))
                        oldX,oldY = cropped_image.size
                        cropped_resized = cropped_image.resize((128,128),resample=Image.BILINEAR)
                        test_tensor = self.transform_image(cropped_resized)
                        test_tensor.unsqueeze_(0)
                        test_image = test_tensor.to(self.device)
                        output = self.model_RegNet(test_image)
                        #tensor_out = output['heatmaps'].cpu().data.numpy()
                        vec_2d = output['vector_2d'].cpu().data.numpy()
                        vec_2d = vec_2d * 128 
                        pred_kpts_2d = vec_2d[0].reshape(-1,2)
                        
                        x = np.array(pred_kpts_2d[:,0]) *(oldX/128) +x1.item()
                        y = np.array( pred_kpts_2d[:,1]) *(oldY/128)+y1.item()
                        plt.imshow(image,aspect='auto')
                        plt.gca().add_patch(Rectangle((x1.item(),y1.item()),x2.item()-x1.item(),y2.item()-y1.item(), linewidth=2,edgecolor='r',facecolor='none'))
                        plot_2d(x,y,x,y)
                        plt.show()
        # Log progress
        #current_time = time.time()
        #inference_time = datetime.timedelta(seconds=current_time - prev_time)
        #prev_time = current_time
        #print(inference_time)
        print("FPS: ", 1.0 / (time.time() - prev_time))
        #detect = detections[0].cpu().data.numpy()
        #print(detect)
        #cv2.imshow('image',img)
        #key = cv2.waitKey(0)


if __name__=='__main__':
    config_file = 'config.yaml'
    images_files = os.listdir('data/')

    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    model_init_class = Tester(config)

    for i in range(len(images_files)):
        image_cv2 = cv2.imread('data/'+images_files[i])
        image_rgb = Image.open('data/'+images_files[i])
        model_init_class.test(image_rgb,image_cv2)

