
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def mask_plot(resulted_img,points,skeleton):

    out_img = cv2.GaussianBlur(resulted_img,(5,5),cv2.BORDER_DEFAULT)
    #plt.imshow(out_img)
    #plt.show()
    edges = cv2.Canny(out_img,100,200)
    #plt.scatter(points[:,0],points[:,1])
    #plt.imshow(edges)
    #plt.show()

    new_points = points #np.array([points[:,0],points[:,1]])
    #print(new_points)

    def drawskeleton(image, points2d, bonelist):
        for bone in bonelist:
            points1 = points2d[bone[0]]
            points2 = points2d[bone[1]]#
            #print(points1,points2)
            cv2.line(image, (int(points1[0]),int(points1[1])),(int(points2[0]),int(points2[1])),(255,255,0),5)
        return image

    #resulted_img = drawskeleton(resulted_img,new_points,skeleton)
    #plt.imshow(resulted_img)
    #plt.show()

    ears = [3,4]
    x = new_points[3]
    y = new_points[4]
    z = [abs(x[0]-y[0])/2,abs(x[1]-y[1])/2]
    #print(z)

    center_coordinates = ((int(x[0]-z[0]),int(x[1])))
    #print(center_coordinates)
    axesLength = (int(z[0]),int(z[0]*2))
    #print(axesLength)
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (0, 255, 255) 
    thickness = -1
    image = cv2.ellipse(resulted_img, center_coordinates, axesLength, 
               angle, startAngle, endAngle, color, thickness)
    #plt.imshow(image)
    #plt.show()


    """
    for alpha in np.arange(0, 1.1, 0.1)[::-1]:
        overlay = image.copy()
        output = image.copy()
        cv2.ellipse(returned_img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
        
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
        plt.imshow(overlay)
        plt.show()
    """

    left_hip = new_points[11]
    right_hip = new_points[12]
    hip_mid = [abs(right_hip[0]-left_hip[0])/2, abs(right_hip[1]-left_hip[1])/2]
    print(hip_mid)
    new_hip_mid = [left_hip[0]-hip_mid[0],left_hip[1]-hip_mid[1]]
    print(new_hip_mid)
    hip_down = [int(new_hip_mid[0]-0.5), int(new_hip_mid[1]+10)]
    print(hip_down)


    # In[23]:

    """
    #y is row, x is column & x is in 0th position so, y in 1st position
    hip_down_y = new_hip_mid[1]  #row
    print(hip_down_y)
    hip_down_x = int(new_hip_mid[0])   #col
    print(hip_down_x)
    row = edges[:,hip_down_x]
    #print(row)
    #plt.plot(row)
    edg = np.where(row != 0)[0]
    print(edg)
    edg = edg[-1]
    print(edg)
    hip_down = [edg,hip_down_y]
    print(hip_down)

    """
    # In[24]:


    #y is row, x is column & x is in 0th position so, y in 1st position
    left_shoulder_y = new_points[5][1]  #row
    print(left_shoulder_y)
    left_shoulder_x = new_points[5][0]   #col
    print(left_shoulder_x)
    row = edges[int(np.round(left_shoulder_y))]
    #print(row)
    #plt.plot(row)
    edg = np.where(row != 0)[0]
    print(edg)
    edg = edg[-1]
    print(edg)
    left_shoulder_l = [edg,left_shoulder_y]
    print(left_shoulder_l)


    # In[ ]:





    # In[25]:


    left_elbow_y = new_points[7][1]
    print(left_elbow_y)
    left_elbow_x = new_points[7][0]
    #print(left_elbow_x)
    row = edges[int(np.round(left_elbow_y))]
    #print(row)
    #plt.plot(row)
    edg_1 = np.where (row!= 0)[0]
    #print(edg_1)
    for i in range(len(edg_1)):
        if edg_1[i] > left_elbow_x:
            col = edg_1[i]
            break
        else:
            col = left_elbow_x
    print(col)
    left_elbow_l = [col,left_elbow_y]
    print(left_elbow_l)


    # In[26]:


    left_wrist_y = new_points[9][1]
    print(left_wrist_y)
    left_wrist_x = new_points[9][0]
    #print(left_wrist_x)
    row = edges[int(np.round(left_wrist_y))]
    #print(row)
    #plt.plot(row)
    edg_2 = np.where (row!= 0)[0]
    #print(edg_2)
    for i in range(len(edg_2)):
        if edg_2[i] > left_elbow_x:
            col_1 = edg_2[i]
            break
        else:
            col_1 = left_elbow_x
    print(col_1)
    left_wrist_l = [col_1,left_wrist_y]
    print(left_wrist_l)


    # In[27]:


    right_shoulder_y = new_points[6][1]
    print(right_shoulder_y)
    right_shoulder_x = new_points[6][0]
    #print(right_shoulder_x)
    row = edges[int(np.round(right_shoulder_y))]
    #print(row)
    #plt.plot(row)
    edg_3 = np.where (row!= 0)[0]
    print(edg_3)
    for i in range(len(edg_3)-1,-1,-1):
        if edg_3[i] < right_shoulder_x:
            col_2 = edg_3[i]
            break
        else:
            col_2 = right_shoulder_x
    print(col_2)
    right_shoulder_r = [col_2,right_shoulder_y]
    print(right_shoulder_r)


    # In[28]:


    right_elbow_y = new_points[8][1]
    print(right_elbow_y)
    right_elbow_x = new_points[8][0]
    #print(right_elbow_x)
    row = edges[int(np.round(right_elbow_y))]
    #print(row)
    #plt.plot(row)
    edg_4 = np.where (row!= 0)[0]
    #print(edg_4)
    for i in range(len(edg_4)-1,-1,-1):
        if edg_4[i] < right_elbow_x:
            col_3 = edg_4[i]
            break
        else:
            col_3 = right_elbow_x
    print(col_3)
    right_elbow_r = [col_3,right_elbow_y]
    print(right_elbow_r)


    # In[29]:


    right_wrist_y = new_points[10][1]
    print(right_wrist_y)
    right_wrist_x = new_points[10][0]
    #print(right_wrist_x)
    row = edges[int(np.round(right_wrist_y))]
    #print(row)
    #plt.plot(row)
    edg_5 = np.where (row!= 0)[0]
    #print(edg_5)
    for i in range(len(edg_5)-1,-1,-1):
        if edg_5[i] < right_wrist_x:
            col_4 = edg_5[i]
            break
        else:
            col_4 = right_wrist_x
    print(col_4)
    right_wrist_r = [col_4,right_wrist_y]
    print(right_wrist_r)


    # In[30]:


    right_shoulder_y2 = new_points[6][1]
    print(right_shoulder_y2)
    right_shoulder_x2 = new_points[6][0]
    #print(right_shoulder_x2)
    row = edges[int(np.round(right_shoulder_y2))]
    #print(row)
    #plt.plot(row)
    edg_6 = np.where (row!= 0)[0]
    print(edg_6)
    for i in range(len(edg_6)):
        if edg_6[i] > right_shoulder_x2:
            col_5 = edg_6[i]
            break
        else:
            col_5 = right_shoulder_x2
    print(col_5)
    right_shoulder_l = [col_5,right_shoulder_y2]
    print(right_shoulder_l)


    # In[31]:


    right_elbow_y2 = new_points[8][1]
    print(right_elbow_y2)
    right_elbow_x2 = new_points[8][0]
    print(right_elbow_x2)
    row = edges[int(np.round(right_elbow_y2))]
    #print(row)
    #plt.imshow(edges)
    #plt.show()
    #plt.plot(row)
    edg_7 = np.where (row!= 0)[0]
    #print(edg_7)
    for i in range(len(edg_7)):
        print(edg_7[i])
        dummy_var = int(edg_7[i])
        if dummy_var > int(right_elbow_x2):
            col_6 = edg_7[i]
            break
        else:
            col_6 = right_elbow_x2

    print(col_6)

    right_elbow_l = [col_6,right_elbow_y2]
    print(right_elbow_l)


    # In[32]:


    right_knee_y2 = new_points[14][1]
    print(right_knee_y2)
    right_knee_x2 = new_points[14][0]
    #print(right_knee_x2)
    row = edges[int(np.round(right_knee_y2))]
    #print(row)
    #plt.plot(row)
    edg_8 = np.where (row!= 0)[0]
    #print(edg_8)
    for i in range(len(edg_8)):
        if edg_8[i] > right_knee_x2:
            col_7 = edg_8[i]
            break
        else:
            col_7 = right_knee_x2
    print(col_7)
    right_knee_l = [col_7,right_knee_y2]
    print(right_knee_l)


    # In[33]:


    right_ankle_y2 = new_points[16][1]
    print(right_ankle_y2)
    right_ankle_x2 = new_points[16][0]
    #print(right_ankle_x2)
    row = edges[int(np.round(right_ankle_y2))]
    #print(row)
    #plt.plot(row)
    edg_9 = np.where (row!= 0)[0]
    #print(edg_9)
    for i in range(len(edg_9)):
        if edg_9[i] > right_ankle_x2:
            col_8 = edg_9[i]
            break
        else:
            col_8 = right_ankle_x2
    print(col_8)
    right_ankle_l = [col_8,right_ankle_y2]
    print(right_ankle_l)


    # In[ ]:





    # In[34]:


    left_ankle_y = new_points[15][1]
    print(left_ankle_y)
    left_ankle_x = new_points[15][0]
    #print(left_ankle_x)
    row = edges[int(np.round(left_ankle_y))]
    #print(row)
    #plt.plot(row)
    edg_10 = np.where (row!= 0)[0]
    #print(edg_10)
    for i in range(len(edg_10)):
        if edg_10[i] > left_ankle_x:
            col_9 = edg_10[i]
            break
        else:
            col_9 = left_ankle_x
    print(col_9)
    left_ankle_l = [col_9,left_ankle_y]
    print(left_ankle_l)


    # In[35]:


    left_knee_y = new_points[13][1]
    print(left_knee_y)
    left_knee_x = new_points[13][0]
    #print(left_knee_x)
    row = edges[int(np.round(left_knee_y))]
    #print(row)
    #plt.plot(row)
    edg_11 = np.where (row!= 0)[0]
    #print(edg_11)
    for i in range(len(edg_11)):
        if edg_11[i] > left_knee_x:
            col_10 = edg_11[i]
            break
        else:
            col_10 = left_knee_x
    print(col_10)
    left_knee_l = [col_10,left_knee_y]
    print(left_knee_l)


    # In[36]:


    left_wrist_y2 = new_points[9][1]
    print(left_wrist_y2)
    left_wrist_x2 = new_points[9][0]
    #print(left_wrist_x2)
    row = edges[int(np.round(left_wrist_y2))]
    #print(row)
    #plt.plot(row)
    edg_12 = np.where (row!= 0)[0]
    #print(edg_12)
    for i in range(len(edg_12)-1,-1,-1):
        if edg_12[i] < left_wrist_x2:
            col_11 = edg_12[i]
            break
        else:
            col_11 = left_wrist_x2
    print(col_11)
    left_wrist_r = [col_11,left_wrist_y2]
    print(left_wrist_r)


    # In[37]:


    left_elbow_y2 = new_points[7][1]
    print(left_elbow_y2)
    left_elbow_x2 = new_points[7][0]
    #print(left_elbow_x2)
    row = edges[int(np.round(left_elbow_y2))]
    #print(row)
    #plt.plot(row)
    edg_13 = np.where (row!= 0)[0]
    #print(edg_13)
    for i in range(len(edg_13)-1,-1,-1):
        if edg_13[i] < left_elbow_x2:
            col_12 = edg_13[i]
            break
        else:
            col_12 = left_elbow_x2
    print(col_12)
    left_elbow_r = [col_12,left_elbow_y2]
    print(left_elbow_r)


    # In[38]:


    left_knee_y2 = new_points[13][1]
    print(left_knee_y2)
    left_knee_x2 = new_points[13][0]
    #print(left_knee_x2)
    row = edges[int(np.round(left_knee_y2))]
    #print(row)
    #plt.plot(row)
    edg_14 = np.where (row!= 0)[0]
    #print(edg_14)
    for i in range(len(edg_14)-1,-1,-1):
        if edg_14[i] < left_knee_x2:
            col_13 = edg_14[i]
            break
        else:
            col_13 = left_knee_x2
    print(col_13)
    left_knee_r = [col_13,left_knee_y2]
    print(left_knee_r)


    # In[39]:


    left_ankle_y2 = new_points[15][1]
    print(left_ankle_y2)
    left_ankle_x2 = new_points[15][0]
    #print(left_ankle_x2)
    row = edges[int(np.round(left_ankle_y2))]
    #print(row)
    #plt.plot(row)
    edg_15 = np.where (row!= 0)[0]
    #print(edg_15)
    for i in range(len(edg_15)-1,-1,-1):
        if edg_15[i] < left_ankle_x2:
            col_14 = edg_15[i]
            break
        else:
            col_14 = left_ankle_x2
    print(col_14)
    left_ankle_r = [col_14,left_ankle_y2]
    print(left_ankle_r)


    # In[40]:


    right_wrist_y2 = new_points[10][1]
    print(right_wrist_y2)
    right_wrist_x2 = new_points[10][0]
    #print(left_hip_x)
    row = edges[int(np.round(right_wrist_y2))]
    #print(row)
    #plt.plot(row)
    edg_16 = np.where (row!= 0)[0]
    #print(edg_16)
    for i in range(len(edg_16)):
        if edg_16[i] > right_wrist_x2:
            col_15 = edg_16[i]
            break
        else:
            col_15 = right_wrist_x2
    print(col_15)
    right_wrist_l = [col_15,right_wrist_y2]
    print(right_wrist_l)


    # In[41]:


    right_knee_y = new_points[14][1]
    print(right_knee_y)
    right_knee_x = new_points[14][0]
    #print(right_knee_x)
    row = edges[int(np.round(right_knee_y))]
    #print(row)
    #plt.plot(row)
    edg_17 = np.where (row!= 0)[0]
    #print(edg_17)
    for i in range(len(edg_17)-1,-1,-1):
        if edg_17[i] < right_knee_x:
            col_16 = edg_17[i]
            break
        else:
            col_16 = right_knee_x
    print(col_16)
    right_knee_r = [col_16,right_knee_y]
    print(right_knee_r)


    # In[42]:


    right_ankle_y = new_points[16][1]
    print(right_ankle_y)
    right_ankle_x = new_points[16][0]
    #print(right_ankle_x)
    row = edges[int(np.round(right_ankle_y))]
    #print(row)
    #plt.plot(row)
    edg_18 = np.where (row!= 0)[0]
    #print(edg_1)
    for i in range(len(edg_18)-1,-1,-1):
        if edg_18[i] < right_ankle_x:
            col_17 = edg_18[i]
            break
        else:
            col_17 = right_ankle_x
    print(col_17)
    right_ankle_r = [col_17,right_ankle_y]
    print(right_ankle_r)


    # In[43]:


    right_hip_y = new_points[12][1]
    print(right_hip_y)
    right_hip_x = new_points[12][0]
    #print(right_hip_x)
    row = edges[int(np.round(right_hip_y))]
    #print(row)
    #plt.plot(row)
    edg_19 = np.where (row!= 0)[0]
    #print(edg_19)
    for i in range(len(edg_19)-1,-1,-1):
        if edg_19[i] < right_hip_x:
            col_18 = edg_19[i]
            break
        else:
            col_18 = right_hip_x
    print(col_18)
    right_hip = [col_18,right_hip_y]
    print(right_hip)


    # In[44]:


    left_hip_y = new_points[11][1]
    print(left_hip_y)
    left_hip_x = new_points[11][0]
    #print(left_hip_x)
    row = edges[int(np.round(left_hip_y))]
    #print(row)
    #plt.plot(row)
    edg_20 = np.where (row!= 0)[0]
    #print(edg_20)
    for i in range(len(edg_20)):
        if edg_20[i] > left_hip_x:
            col_19 = edg_20[i]
            break
        else:
            col_19 = left_hip_x 
    print(col_19)
    left_hip = [col_19,left_hip_y]
    print(left_hip)


    # In[45]:
    """

    pts = np.array([])
    print(pts)
    #pts = pts.reshape((-1,1,2))
    cv2.polylines(returned_img,[pts],True,(0,255,255))
    plt.imshow(image)
    plt.show()


    # In[46]:


    #right_knee_l],right_ankle_l,right_ankle_r,right_knee_r,right_hip,right_elbow_l,right_wrist_l,right_wrist_r,right_elbow_r,right_shoulder_r,right_shoulder_r,left_shoulder_l


    # In[47]:


    pts = np.array([left_hip,left_knee_l,left_ankle_l,left_ankle_r,left_knee_r,left_hip])
    print(pts)
    #pts = pts.reshape((-1,1,2))
    cv2.polylines(returned_img,[pts],True,(0,255,255))
    plt.imshow(image)
    plt.show()
    """
    pts_upper = np.array([right_wrist_l,right_elbow_l,right_hip,left_hip,left_elbow_r,left_wrist_l,left_wrist_r,left_elbow_l,left_shoulder_l,right_shoulder_l,right_shoulder_r,right_elbow_r,right_wrist_r,right_wrist_l], dtype=np.int32)
    #print(pts)
    #pts = pts.reshape((-1,1,2))
    #cv2.fillPoly(resulted_img,[pts],(255,0,0))
    #plt.imshow(image)
    #plt.show()

    ##################################### if elbow left > hip and wrist > hip 

    pts_lower = np.array([right_ankle_r,right_knee_r,right_hip,left_hip,left_knee_l,left_ankle_l,left_ankle_r,left_knee_r,hip_down,right_knee_l,right_ankle_l,right_ankle_r], dtype=np.int32)
    #print(pts)
    #pts = pts.reshape((-1,1,2))
    #cv2.fillPoly(resulted_img,[pts],(255,0,0))
    #plt.imshow(image)
    #plt.show()
    return pts_upper, pts_lower
