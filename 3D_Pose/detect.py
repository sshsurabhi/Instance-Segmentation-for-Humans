import yaml
import torch
import argparse
import numpy as np
import json, csv, cv2
import matplotlib.pyplot as plt
import torch.nn as nn
#from metric import get_metric
from PIL import Image
from torchvision import transforms
bone_2= [[0,1],[0,4],
            [0,7],
            [0,10],
            [0,13],
            [1,2],
            [2,3],
            [3,17],
            [4,5],
            [5,6],
            [6,18],
            [10,11],
            [11,12],
            [12,19],
            [7,8],
            [8,9],
            [9,20],
            [13,14],
            [14,15],
            [15,16]]
bone_n= [[0,1],[0,12],[0,16],[1,20],[20,2],[20,4],[20,8],[2,3],[4,5],[5,6],[8,9],[9,10],[12,13],[13,14],[14,15],[16,17],[17,18],[18,19]]

def get_model():
    #from model_ import EncoderDecoder
    from SSM_for_single_image_detection import EncoderDecoder
    model = EncoderDecoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

def plot_2d(x_,y_,act_x,act_y):
    for bone in bone_n:
        keyp_x = x_
        keyp_y = y_
        plt.plot([keyp_x[bone[0]],keyp_x[bone[1]]],[keyp_y[bone[0]],keyp_y[bone[1]]],c='r',linewidth=2.0)
        plt.plot([act_x[bone[0]],act_x[bone[1]]],[act_y[bone[0]],act_y[bone[1]]],c='g',linewidth=2.0)
        plt.legend(['Predicted','Actual'],fontsize=20)
        plt.savefig('Results/image_1.png')

def line_plot(keypoints,actual, bone_list, ax):
    origin = actual[0,:]
    norm_kpts=[]
    for i in range(len(actual)):
        norm_kpts.append(-keypoints[i]+origin)
    norm_kpts = np.array(norm_kpts)
    print(actual-norm_kpts)

    actual_x = actual[:,0]
    actual_y = actual[:,1]
    actual_z = actual[:,2]
    keypoints_x = keypoints[:,0] 
    keypoints_y = keypoints[:,1] 
    keypoints_z = keypoints[:,2] 
    for bone in bone_list:
        if bone[1] != 20:
            ax.plot([keypoints_x[bone[0]],keypoints_x[bone[1]]],[keypoints_y[bone[0]],keypoints_y[bone[1]]],[keypoints_z[bone[0]],keypoints_z[bone[1]]],c='r')
            ax.plot([actual_x[bone[0]],actual_x[bone[1]]],[actual_y[bone[0]],actual_y[bone[1]]],[actual_z[bone[0]],actual_z[bone[1]]],c='b')
        else:
            ax.plot([keypoints_x[bone[0]],keypoints_x[bone[1]]],[keypoints_y[bone[0]],keypoints_y[bone[1]]],[keypoints_z[bone[0]],keypoints_z[bone[1]]],c='r')
            ax.plot([actual_x[bone[0]],actual_x[bone[1]]],[actual_y[bone[0]],actual_y[bone[1]]],[actual_z[bone[0]],actual_z[bone[1]]],c='b')            

appender=[]
keypoint_appender = []
bb_appender =[]
kp_3d_appender =[]

def data_loader(scope):
    with open('keypoints_2d_and_3d.csv','r') as f:
        count = 0
        reader = csv.reader(f)
        for row in reader:
            appender.append(row[0])
            keypoints = [float(i) for i in row[1:43]]
            keypoints = np.array(keypoints).reshape(-1,2)
            keypoint_appender.append(keypoints)
            bbox_col_min = min(keypoints[:,0])-15
            if bbox_col_min < 1:
                bbox_col_min = 1

            bbox_col_max = max(keypoints[:,0])+15
            if bbox_col_max > 640:
                bbox_col_max = 640

            bbox_row_min = min(keypoints[:,1])-15
            if bbox_row_min < 1:
                bbox_row_min = 1

            bbox_row_max = max(keypoints[:,1])+15
            if bbox_row_max > 480:
                bbox_row_max = 480

            bb_appender.append([bbox_col_min,bbox_row_min,bbox_col_max,bbox_row_max])

            keypoints_3d = [float(i) for i in row[43:]]
            keypoints_3d = np.array(keypoints_3d)#.reshape(-1,3)
            kp_3d_appender.append(keypoints_3d)
            count+=1
            if count >5000:
                break
    return appender, keypoint_appender, bb_appender, kp_3d_appender


def data_loader_ntu(scope):
    appender=[]
    keypoint_appender = []
    bb_appender =[]
    kp_3d_appender =[]
    with open('keypoints_body_2d_3d.csv','r') as f:
        count=0
        reader = csv.reader(f)
        for row in reader:
            appender.append(row[0])
            keypoints = [float(i) for i in row[1:51]]
            keypoints = np.array(keypoints).reshape(-1,2)
            keypoint_appender.append(keypoints)
            bbox_col_min = min(keypoints[:,0])-15
            bbox_col_max = max(keypoints[:,0])+15
            bbox_row_min = min(keypoints[:,1])-15
            bbox_row_max = max(keypoints[:,1])+15
            bb_appender.append([bbox_col_min,bbox_row_min,bbox_col_max,bbox_row_max])

            keypoints_3d = [float(i) for i in row[51:]]
            keypoints_3d = np.array(keypoints_3d)#.reshape(-1,3)
            kp_3d_appender.append(keypoints_3d)
        return appender, keypoint_appender, bb_appender, kp_3d_appender

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate(config):
    model = get_model()
    weight_file = 'experiment/' + config['weights']
    checkpoint = torch.load(weight_file,map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    n_kpoints = 21
    transform_image = transforms.Compose([
                           transforms.Resize((128, 128)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_names, actual_kpts_2d, bb_2d,kpts_3d = data_loader_ntu('dataset/')
    #images = images[:1000]
    pck_appender = []
    for i in range(len(image_names)):
        img_path = 'C:/Users/chait/OneDrive/Documents/NTU/'
        image =  Image.open(img_path +image_names[i])
        bb = bb_2d[i]
        cropped_image = image.crop(box=(bb[0],bb[1],bb[2],bb[3]))
        oldX,oldY = cropped_image.size
        cropped_resized = cropped_image.resize((128,128),resample=Image.BILINEAR)
        test_tensor = transform_image(cropped_resized)
        test_tensor.unsqueeze_(0)
        test_image = test_tensor.to(device)
        output = model(test_image)
        #tensor_out = output['heatmaps'].cpu().data.numpy()
        vec_2d = output['vector_2d'].cpu().data.numpy()
        vec_2d = vec_2d * 128 
        pred_kpts_2d = vec_2d[0].reshape(-1,2)
        
        x = np.array(pred_kpts_2d[:,0]) *(oldX/128) +bb[0]
        y = np.array( pred_kpts_2d[:,1]) *(oldY/128)+bb[1]
        
        plt.imshow(image,aspect='auto')
        actual=actual_kpts_2d[i]
        #plot_2d(x,y,actual[:,0],actual[:,1])
        #plt.show()

        vec_3d = output['vector_3d'].cpu().data.numpy()
        vec_3d =vec_3d[0].reshape(-1,3)
        actual_3d = kpts_3d[i].reshape(-1,3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        line_plot(vec_3d,actual_3d,bone_n, ax)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.legend(['predicted','actual'],fontsize=20)
        plt.show()


if __name__ == '__main__':
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    evaluate(config)