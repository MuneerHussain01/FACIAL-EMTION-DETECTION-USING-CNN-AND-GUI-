# FACIAL-EMOTION-DETECTION-USING-CNN-AND-GUI-

Objective 
To design and implement a facial emotion classification system that: 
-	Learns emotional features from facial images using CNN. 
-	Detects and classifies emotions in real-time using webcam or uploaded images. 
-	Displays top-3 predictions with a confidence bar chart. 
-	Achieves high accuracy (>90%) on a custom dataset. 
Tools & Technologies 
	Component 	Tool/Library 	
	Programming 	Python 	
	Deep Learning 	TensorFlow, Keras 	
	GUI 	Tkinter 	
	Image Processing 	OpenCV, Pillow 	
	Visualization 	Matplotlib 	
	Dataset Format 	Folder-wise (train/test) 	
	Image Size 	48×48 (grayscale) 	
 
Dataset Structure 
train/ 
  ├── happy/ 
  ├── sad/ 
  ├── angry/ 
  └── ... (other emotions) 
 
test/ 
  ├── happy/ 
  ├── sad/   ├── angry/ 
  └── ... 
 

Methodology 
 
1. Preprocessing: 
-	Grayscale conversion 
-	Resize to 48x48 
-	Normalization (pixel values 0-1) - Data augmentation (rotation, zoom, flip) 
2. Model Architecture: 
A custom CNN built from scratch: 
-	3 × Conv2D + MaxPooling layers 
-	Flatten 
-	Dense layer with dropout 
-	Final Dense (Softmax) for classification 
3. Training: 
-	Optimizer: Adam 
-	Loss: Categorical Crossentropy 
-	Epochs: 25 
-	Accuracy Achieved: ~92% on test set 
4. GUI Features: 
-	Upload image or use webcam 
-	Show face image and predictions 
-	Top-3 emotions with confidence 
-	Bar chart of all emotion probabilities 
Final Results 
	Feature 	Status 	
	Emotion Classification 	   	
	Train/Test Accuracy  	   	
	Webcam Integration 	   	
	Top-3 Emotion Predictions 	   	
	Confidence Bar Chart 	   	
	GUI (Tkinter) 	   	
 
Conclusion 
This lab successfully demonstrates machine learning-based image classification for detecting human emotions from facial images. With the integration of real-time webcam, custom CNN training, and a user-friendly GUI, the project meets all objectives of Lab 14 and reflects practical application of digital image processing with ML.


 DATASET: https://www.kaggle.com/datasets/msambare/fer2013?resource=download


![image](https://github.com/user-attachments/assets/6c1d814c-8d1a-49fc-9400-138eedaad078)

![image](https://github.com/user-attachments/assets/62698070-e4f2-4bb3-be3c-c674d517ba1b)






 



 
