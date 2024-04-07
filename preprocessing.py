import os
import cv2
import numpy as np

# Define directories
data_directories = {
    "train": "dataset/split_ttv_dataset_type_of_plants/split_ttv_dataset_type_of_plants/Train_Set_Folder",
    "test": "dataset/split_ttv_dataset_type_of_plants/split_ttv_dataset_type_of_plants/Test_Set_Folder",
    "validation": "dataset/split_ttv_dataset_type_of_plants/split_ttv_dataset_type_of_plants/Validation_Set_Folder"
}

# Define image size
img_size = 150

def create_preprocessed_data(data_type):
    # Initialize data
    data = []

    # Define categories
    categories = os.listdir(data_directories[data_type])

    for category in categories:
        
        print (category)
        path = os.path.join(data_directories[data_type], category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (img_size, img_size))

                normalized_array = new_array / 255.0

                # Append image array and corresponding category label to data
                data.append([normalized_array, class_num])
            except Exception as e:
                pass

    # Shuffle the data
    np.random.shuffle(data)

    # Separate features (X) and labels (y)
    X = []
    y = []

    for features, label in data:
        X.append(features)
        y.append(label)


    X = np.array(X).reshape(-1, img_size, img_size, 1)

    y = np.array(y)
    
    os.makedirs('preprocessed_images', exist_ok=True)
    # Save the preprocessed data
    np.save(f'preprocessed_images/X_{data_type}.npy', X)
    np.save(f'preprocessed_images/y_{data_type}.npy', y)


for data_type in data_directories.keys():
    create_preprocessed_data(data_type)

print("Data preprocessing completed.")
