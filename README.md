# Real-time-Chickens-Detection-and-Tracking
Chickens detectetion &amp; tracking system that uses YOLOv8 for detection, Byte Track for tracking, and MobileNetv3 embeddings for chickens differentiation



# **Detection Model**

 * **Dataset 🗃**
   * *the Instance Segmentation dataset was manually created using Roboflow*
   * *current version of the dataset contains 282 labeled images, which was used to train the model*

<img width="316" height="210" alt="dataset_seg" src="https://github.com/user-attachments/assets/620e2f0e-5b2c-4e01-bcec-d88ac15d6acf" />


 * **Instance Segmentation Model ⚙**
   * *the `yolov8s-seg` model was for training and for making predictions*
   * *first, the object detection dataset was used with `yolov8s` model, but the segmentation model has better results*


**Check Object Detection dataset results (Bounding Box)**

https://github.com/user-attachments/assets/c8f771fe-d6a4-4069-b6cd-226fd5c5ea95


**Check Instance Segmentation dataset results (without tracking)**

https://github.com/user-attachments/assets/d20761b6-2c81-40c7-a760-bd0e3a426d53


# Tracking Predictions with ByteTrack

📽 Using YOLO Model alone isn't enough to achive real-time detections, even if trained for longer or with a bigger dataset, so to better optimize it for real-time detections we are using ByteTrack  
  * the model predicts on each frame of the video to detect the chickens, but the chickens move in those frames
  * we use a tracker so that the model "tracks" the movement of the chickens with the predictions to better adjust the bouding boxes, instead of creating multiple ones for each prediction


**Check the Bounding Box Model without the Tracker**
 * the model outputs multiple bounding boxes, so the outputed video looks "messy"

https://github.com/user-attachments/assets/ffcf6767-5c3a-484d-b8a3-ad8d990a849b


# **MobileNetv3 Embeddings - Chickens Differentiation**

🔎 We removed the classifer layer from the MobileNetv3 Model, so instead of the class logits we get the feature vector (embbeding)
   * when we make predictions on a frame, we crop the chickens bbox and send it to the MobileNet model, to get the embeddings
   * when we have a "new detection", we use the embedding to check if we already detected the chicken by checking the appereance similarity
   * using this we can better see how many unique detections is the model making


**Check MobileNet Emebddings**

https://github.com/user-attachments/assets/fc7ad379-6d20-4950-b6cc-6bde99678dcd


# Improvments

📊 The model was created to experiment with these tools, but it can be easly improved in a lot of different ways (depending on the goal). Below are just a few improvements - the model can do a great job at detecting chickens, but how the model is using those detections is just as important as accurate detections 

* **Better data** - the dataset used is very small, with just one label ("chicken"), a larger dataset with high-quality data will perform better
* **Better detections** - the model is doing not doing that great at detecting all the chickens from a frame, but we can improve this in different ways
    * *Edge Detection Filter* - we can change YOLO model arhitecture, changing the input from 3 channels (RGB) to 4, and add to the 4th channel an edge detection filter like the `Sobel Filter`
* **MobileNet Embeddings improvments** - even with the embeddings the model can still count the same chicken multiple times, for example if the model detects the first half of the chicken and in the next frames the other half of the chicken, when we check the similarity of those detections the model might see it as a different chicken
    * also, if the chickens look alike the model might get confused and assign the embedding to the wrong chicken
* **Smart Detections** - it's great if we can detect even if the chicken is not fully visible, like if only 10% or 20% of the chicken is visible, but in the next frames it might be completly visible
    * humans can do this easly when watching a video, if a chicken is not fully visible we can still identify it as the same chicken when it becomes fully visible in the next frames (not necessary always, but it's something we can do)
    * if it can be implemented in a simple way, so that we don't affect performance, we can have better detections and better ways of applying MobileNet embeddings (since our model "knows" what part of the chicken it has detected)












