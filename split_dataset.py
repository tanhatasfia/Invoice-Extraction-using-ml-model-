import os
import numpy as np
import shutil
import glob
import pandas as pd
from tqdm import tqdm

# Creating Train / Val / Test folders (One time use)
dataset_dir = 'D:/Extraction of invoice/'
image_dir = 'Dataset-images'
image_dir2 = 'augmented_dataset'

train_ratio = 0.80
val_ratio = 0.11
test_ratio = 0.09

# Ensure ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

# Create directories for train, val, and test
os.makedirs(os.path.join(dataset_dir,image_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir,image_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir,image_dir, 'test'), exist_ok=True)

# List all images
myFileList = glob.glob(os.path.join(image_dir, "*.jpg"))
print("\nThere are", len(myFileList), "images read by Python")

# Shuffle the file list
np.random.shuffle(myFileList)

# Calculate split indices
train_end_idx = int(len(myFileList) * train_ratio)
val_end_idx = train_end_idx + int(len(myFileList) * val_ratio)

# Split the data
train_FileNames = myFileList[:train_end_idx]
val_FileNames = myFileList[train_end_idx:val_end_idx]
test_FileNames = myFileList[val_end_idx:]

print('Total images: ', len(myFileList))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy images to respective directories
for name in train_FileNames:
    shutil.copy(name, os.path.join(dataset_dir,image_dir, 'train'))
for name in val_FileNames:
    shutil.copy(name, os.path.join(dataset_dir,image_dir, 'val'))
for name in test_FileNames:
    shutil.copy(name, os.path.join(dataset_dir, image_dir, 'test'))

# Remove original images
for file in os.listdir(os.path.join(dataset_dir, image_dir)):
    if file.endswith('.jpg'):
        os.remove(os.path.join(dataset_dir, image_dir, file))

# Generate CSV files for train, val, and test
labels_df = pd.read_csv('D:/Extraction_of_invoice/augmented_dataset/new_labels.csv')
train_images_path = os.listdir(os.path.join(dataset_dir, image_dir, 'train'))
val_images_path = os.listdir(os.path.join(dataset_dir, image_dir, 'val'))
test_images_path = os.listdir(os.path.join(dataset_dir, image_dir, 'test'))

train_list = []
val_list = []
test_list = []

for i in tqdm(range(len(labels_df))):
    if labels_df['filename'][i] in train_images_path:
        train_list.append(labels_df.iloc[i])
    elif labels_df['filename'][i] in val_images_path:
        val_list.append(labels_df.iloc[i])
    elif labels_df['filename'][i] in test_images_path:
        test_list.append(labels_df.iloc[i])

# Convert lists to dataframes
train_df = pd.DataFrame(train_list)
val_df = pd.DataFrame(val_list)
test_df = pd.DataFrame(test_list)

# Save dataframes as csv files
train_df.to_csv(os.path.join(dataset_dir, 'train_labels.csv'), index=False)
val_df.to_csv(os.path.join(dataset_dir, 'val_labels.csv'), index=False)
test_df.to_csv(os.path.join(dataset_dir, 'test_labels.csv'), index=False)

print('Successfully split images into train, val, and test')
