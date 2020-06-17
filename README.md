# Repository of Independent Projects, Courses and Competitions on Computer Vision

Courses
  - CS763: Computer Vision Spring 2020 course at IIT Bombay (Assignments and Lab Sessions)
  - Fundamentals of Image and Video Processing (NWU, Coursera)
  
Projects (code can be found in "projects" folder)
     
   1. Social Distancing Detection - COVID19 Challenge
    
   Through background seperation and frame-to-frame difference, humans are detected using motion detection. The countour detected bounding boxes then provide enough information to detect violation of social distancing using thresholding techniques. 
<p align="center">    
   <img src="images/tracksd_raw.png" width="400" title="canny">
   <img src="images/tracksd_nkgrd.png" width="400" title="lane">
   <img src="images/tracksd.png" width="400" title="lane video">
  <img src="images/tracksd1.png" width="400" title="lane video">
  <img src="images/tracksd2.png" width="400" title="lane video">
  <img src="images/tracksd3.png" width="400" title="lane video">
  <img src="images/tracksd4.png" width="400" title="lane video">
</p>

  2. Computer Vision based Pick and Place Robotic Manipulator Arm
      My work as a summer research intern at Bio-robotics Lab, Hanyang University, Seoul, South Korea. Detection, localization and pose estimation of object using Convolutional Neural Networks and depth cameras. Utilised the Faster-RCNN algorithm for object detection. Refer to this repository for details: [[Vision-Maniupulator](https://github.com/meghdeepj/vision-mani.git)]
 <p align="center">    
   <img src="images/vison.jpg" width="600" title="canny">
  <img src="images/vison1.jpg" width="600" title="canny">
 </p>   

   3. Objection Detection Challenge
    
   Using transfer learning on Yolov3 Object detection model pre-trained on COCO dataset, drawing inferences from the results. Interesting observation: Shifting the grip on my mobile phone from vertical to pointing, changes the detection to "remote" thus proving that the model detects context and subtleties
<p align="center">    
   <img src="images/yolo_cell.png" width="400" title="canny">
   <img src="images/yolo_remote.png" width="400" title="lane">
   <img src="images/yolo_antah.jpeg" width="400" title="lane video">
  <img src="images/yolo_graffiti.jpeg" width="400" title="lane video">
</p>

   4. Lane Detection for Self Driving Car
    
   Implemented using OpenCV, the lane detection algorithm makes use of canny edge detection with probabilistic Hough line transform to detect lanes for self driving cars. Thresholding and tuning is doen to ensure line transforms detect prominent lanes. The algorithm is robust to road curvatures and change in lighting conditions such as sunglight glare.
<p align="center">    
   <img src="images/road_canny.png" width="500" title="canny">
   <img src="images/lanes.png" width="600" title="lane">
<!--    <img src="images/face_kp_me.jpeg" width="1000" title="lane video"> -->
</p>

  5. Facial Keypoint Detection Challenge
    
   Using Convolutional Neural Networks trained a facial keypoint detector that can locate 15 facial keypoints from a grayscale cropped image of a person. Robust enough to work on my face, even though it wasn't part of the similar distribution testset.
<p align="center">    
   <img src="images/face_kp.jpeg" width="640" title="face1">
  <img src="images/face_kp_me.jpeg" width="300" title="face2">
</p>

  6. Oreo detection and Counter Challenge
    
   Detection and counting oreos during production. Using OpenCV made use of colour thresholding and contour detection to count the number of oreos being produced that are completely contained in the video frame.
<p align="center">    
   <img src="images/oreo_challenge.png" width="640" title="Oreo">
</p>

  7. Virtual Cam
  
   Visualising 2D objects from the perspective of a virtual camera that can move around in 3D space with 6degrees of freedom. Making use     of camera calibration and homography to reproject the 2D image to a virtual 3D space, even with distortions.
   Inspired by Kaustubh Sadekar's github repo:[[Virtual Cam](https://github.com/kaustubh-sadekar/VirtualCam.git)]
<p align="center">    
   <img src="images/virtual-cam_1.gif" width="300" title="3D Virtual cam">
   <img src="images/virtual-cam_2.gif" width="300" title="3D distortions">
</p>

  8. Invisibility Cloak
  
  Inspired by harry potter, the invisibility cloak one of the most popular image processing projects out there. Using OpenCV made use of colour thresholding mask on video frames and applying the background image data, the invisbility cloak effect can be recreated.
<p align="center">    
   <img src="images/Invs_cloak.jpeg" width="300">
</p>
