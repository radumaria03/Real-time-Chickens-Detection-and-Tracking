# Real-time-Chickens-Detection-and-Tracking
Chickens detectetion &amp; tracking system that uses YOLOv8 for detection, Byte Track for tracking, and MobileNetv3 embeddings for chickens differentiation



# **Detection Model**

 * **Dataset 🗃**
   * *the model uses an Instance Segmentation dataset created using Roboflow, for labeling the images*
   * *the current version of the dataset contains 282 labeled images, which was used to train the model*

<img width="316" height="210" alt="dataset_seg" src="https://github.com/user-attachments/assets/620e2f0e-5b2c-4e01-bcec-d88ac15d6acf" />


 * **Instance Segmentation Model ⚙**
   * *the `yolov8s-seg` model was for training and for making predictions*
   * *first a bounding box dataset was used, but the segmentation model has better results*


**Check Object Detection dataset results (Bounding Box)**

https://github.com/user-attachments/assets/c8f771fe-d6a4-4069-b6cd-226fd5c5ea95


**Check Instance Segmentation dataset results (without tracking)**

https://github.com/user-attachments/assets/d20761b6-2c81-40c7-a760-bd0e3a426d53


# Tracking Predictions with ByteTrack

📽 Making predictions for images is easy, but for real-time detection in videos in order for the model to perform well we to take care of more things
  * the model predicts on each frame of the video to detect the chickens, but the chickens move in those frames
  * we use a tracker so that the model "tracks" the movement of the chickens with the predictions to better adjust the bouding boxes, instead of creating multiple ones for each prediction


**Check the Bounding Box Model without the Tracker**
 * the model outputs multiple bounding boxes, so the outputed video looks "messy"

https://github.com/user-attachments/assets/ffcf6767-5c3a-484d-b8a3-ad8d990a849b


# **MobileNetv3 Embeddings - Chickens Differentiation**

🔎 We removed the classifer layer from the MobileNetv3 Model, so instead of the class logits we get the feature vector (embbeding)
   * when we have a "new detection", we use the embedding to check if we already detected the chicken by checking the "appereance similarity"
   * when we make predictions on a frame, we crop the bbox and send it to the MobileNet model to get the embeddings
   * using this we can better see how many unique detections is the model making


**Check MobileNet Emebddings**

https://github.com/user-attachments/assets/fc7ad379-6d20-4950-b6cc-6bde99678dcd










