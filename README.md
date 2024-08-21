Emergency Vehicle Detection and Classification

This project focuses on the detection and classification of emergency vehicles using deep learning algorithms. The project leverages a trained MobileNetV2 model to accurately identify different types of vehicles, such as normal vehicles, police cars, ambulances, and fire trucks. It processes video inputs and sends notifications when an emergency vehicle is detected.
Files

    Emergency_vehicles.ipynb
    The main Python notebook containing the entire Emergency Vehicle Detection and Classification project workflow. This includes data preprocessing, model training, evaluation, and real-time application implementation. It serves as the primary notebook for this project.

    resize_and_update.ipynb
    A notebook containing the script that automates image resizing and updates CSV files. This ensures consistent image dimensions for smoother model training and evaluation processes.

    Dataset Folder
    Contains the necessary datasets used in this project:
        Train.csv and Test.csv: Sourced from Kaggle's Emergency Vehicles Identification dataset.
        Train_vt.csv: Created from the Train.csv file (refer to Emergency_vehicles.ipynb for a detailed explanation of the data used).
        Train-test images: A folder containing images used for model training.

    Video_tester Folder
    Contains a collection of videos of emergency vehicles used for testing the model's performance.
