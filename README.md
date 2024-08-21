# Emergency-Vehicle-Detection-and-Classification
This project focuses on the detection and classification of emergency vehicles using deep learning algorithms. 
The project uses a trained MobileNetV2 model to accurately identify vehicles like normal, police, ambulances, and fire trucks, processing video inputs and sending notifications when an emergency vehicle is detected.

# Files
  1. Emergency_vehicles.ipynb is the main Python notebook containing the entire Emergency Vehicle Detection and Classification project workflow, including data preprocessing, model  training, evaluation, and real-time application implementation. This is the main notebook. 
  2. resize_and_update.ipynb notebook contains the script that automates image resizing and updates CSV files, ensuring consistent dimensions for smoother model training and evaluation processes.
  3. Dataset folder ( Train-test folder )
     a. Train.csv and test.csv from Kaggle's Emergency Vehicles Identification
     b. Train_vt.csv is created from the above train.csv ( See Emergency_vehicles.ipynb for a detailed explanation of data used in this project.
  4. Video_tester folder
     a. Contains a bunch of videos of emergency vehicles for testing purposes. 
