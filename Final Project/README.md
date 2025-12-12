Real-Time Object Detection for Autonomous Vehicles

🚗 Overview

This project focuses on building a real-time object detection system for autonomous driving using YOLOv11.
The goal is to accurately detect vehicles, pedestrians, traffic signs, and other essential road objects to support safer autonomous navigation.

We trained our model using a large, diverse dataset containing over 1.8 million bounding boxes, combined from BDD100K and additional sources.
All annotations and dataset preparation were managed through Roboflow.

 🎯Project Objectives

Detect and classify road objects in real-time

Build a model suitable for autonomous driving applications

Train a YOLOv11 model with high accuracy and fast inference

Perform data preprocessing, EDA, augmentation, and annotation cleanup

Evaluate performance using industry-standard metrics

 🧰Technologies Used

Python

YOLOv11

Kaggle (For Training) --> https://share.google/IefNjV0o5b3nIbiqF

Ultralytics

OpenCV

Roboflow

CUDA (GPU Acceleration)

 📊Model Results

Our YOLOv11 model achieved:

86% mAP@50

62% mAP@50–95

These metrics reflect strong performance in both object localization and multi-scale recognition.

 👥Team Members

This project was completed as a group collaboration:

Samer Magdy

Youssef Diaa

Sara Mohamed

Salma Mustafa

📌 Future Improvements

Integrate lane detection and depth estimation

Improve night-time and adverse weather performance

Deploy on edge devices (NVIDIA Jetson, Raspberry Pi)

Expand training data with rare edge-case scenarios

🔗 Live Deployment (Streamlit App)

👉 Try the real-time object detection demo here:
https://vechiledetection-n7ju7pdqd4sdgbuq8t7ket.streamlit.app/

📽️ Project Presentation

You can view the full project presentation here:

🔗 Presentation Link: (https://gamma.app/docs/Real-Time-Object-Detection-for-Autonomous-Vehicles-DEPI-Round-3-al50y441z0kd0f4?mode=doc)

📚 Final Data

The model was trained on a final merged dataset created and prepared using Roboflow.
This dataset includes all cleaned, normalized, and augmented labels used for training.
 🔗https://www.kaggle.com/datasets/samermagdy/vecchile-detection


Roboflow – Custom Annotated Dataset

A fully processed dataset containing all bounding box annotations and class labels prepared specifically for this project.
🔗 Dataset Workspace: (https://app.roboflow.com/bdd100k-bs1dh/cleaned-0onjw/4)


