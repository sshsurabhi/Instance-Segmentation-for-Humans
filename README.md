# Instance Segmentation for Localizing Humans in Monocular Images

This is an implementation of Instance Segmentation using Python 3, PyTorch, and Keras, and I took a different approach to achievethis Instance Segmentaion. Above, one can find two folders which having a Person object detection in Images using YOLO V3 algorithm and a hour-glass modelled Human pose estimation methos to detect the pose of the person in images.

Using the transfer learning technique, I have combined these person detection algorithm with pose estimation method to achieve my destination. After detecting the pose I applied some common OpenCV techniques such as "Edge detection", "Gaussian functions" to apply defferent masks to the detected person.
YOLO also localize the person in the image and the pose estimation alsorithm finds the pose of the detected person. The weight generated in YOLO training is utilized for Hour-Glass method to estimate the pose. Both methods both were rewritten in PyTorch version for experiencing no other complications in implementation. 
 

### Note:
* Included individual "requirement.txt" files to install the dependencies.
* The pose estimation algorithm which incluuded here is applicapable also for 3D version.

### Features:
* 17 keypoints will be generated for pose estimation which is similar with COCO dataset keypoint format.
* YOLO V3 algorithm for person detection.
* Pose 3D is for pose estimation for detected person.
* This implementation is only for 2D format but the pose estimation folders has an update for 3D version which I worked inmy later stages after my thesis.
* Included seperate terminal support for quick and easy execution of the individual projects.
+ Final output gives a boundary box with individual masks. The algorithm will runs as per the numbers of persons lied in the image and gives individual and random masks to the persons.
### Usage:
Training:
* yolov3_person_det is used to train the model
        >python yolov3_person_det.py
* Instance-Segmentation-for-Humans/3D_Pose/detect is used to train the pose estimation model
        >python detect.py
* final_ouput_demo file is for acheiving Instance segmentation for an image.
        >python final_output_demo.py

The model outputs the following:
+ Weight file will have 17 keypoints for person which is similar to COCO dataset.
+ Model weights in .pth format

### Note2:
+ Will update more information soon

## References
+ Implementation adopted from here and is based on the idea of Pose estimation using Hour-Glass method.
+ COCO dataset is helped alot for my implementation. I took help from [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)
