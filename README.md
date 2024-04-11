# plant_recognition_CNN
Basic implementation of using a CNN to identify different types of plants.

# File Explanations
1. preprocessing.py
   This script is used for preprocessing image data. It reads images from specified directories (train, test, validation), and converts them from BGR to grayscale format, resizes them to a specified size (150x150). It then appends the processed image and its corresponding label to a list. After processing all images, it shuffles the data and separates it into features (X) and labels (y). It is saved as a numpy file in the preproccessed_images directory.

