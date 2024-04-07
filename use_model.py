import cv2
import numpy as np
import tensorflow as tf
import os
# Define image size (it should be the same size as the one used during training)
img_size = 150

# Define categories (it should be the same order as the one used during training)
directory = 'dataset\split_ttv_dataset_type_of_plants\split_ttv_dataset_type_of_plants\Test_Set_Folder' 
categories = os.listdir(directory)
print (categories)
# Function to preprocess new images
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

# Load the model
model = tf.keras.models.load_model('plant_model.h5')

# Make a prediction on a new image
filepath = "yourplantimagehere.jpg"  # replace with your file path
prediction = model.predict([prepare(filepath)])

# Print the category with the largest prob
print(categories[np.argmax(prediction[0])])
