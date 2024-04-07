import cv2
import numpy as np
import tensorflow as tf
import os
# Define image size (it should be the same size as the one used during training)
img_size = 150

#! WRITE YOUR OWN DIRECTORY TO THE DATASET HERE
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

# Load preprocessed validation data
X_validation = np.load('preprocessed_images/X_validation.npy')
y_validation = np.load('preprocessed_images/y_validation.npy')

# Make predictions on the validation set
predictions = model.predict(X_validation)
print(predictions.shape)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Convert predictions from probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)

# If y_validation is a 1D array of class labels, you don't need to use np.argmax
true_classes = y_validation

# Calculate and print statistics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print the first 10 predictions and true labels for comparison
for i in range(10):
    print(f"Predicted class: {categories[predicted_classes[i]]}, True class: {categories[true_classes[i]]}")
