# Face Detection and Recognition System

This project is an end-to-end deep learning pipeline for detecting and recognizing human faces from static images using **FaceNet** and **MTCNN**. The system is trained on the **LFW (Labeled Faces in the Wild)** dataset and includes:

- Data preprocessing & augmentation  
- Face detection using MTCNN  
- Face embedding using FaceNet  
- Classification using an SVM  
- A user-friendly Streamlit web app  
- Accuracy reporting and confidence filtering

- Here we have trained our model and saved it as face_classifier.pkl and later access it everytime we run our app.
we run the facenew.py. It identifies the images with some degree of confidence. If the predict probability is >10% then
we get an output name and the probability %. If less we get a warning unknown image.
