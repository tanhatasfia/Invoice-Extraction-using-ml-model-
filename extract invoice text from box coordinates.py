!apt-get install tesseract-ocr
!apt-get install libtesseract-dev
!apt-get install -y poppler-utils
!pip install pandas pytesseract opencv-python


import pandas as pd
import pytesseract
import cv2
import os

# Load the CSV file
csv_file_path = '/content/predicted_coordinates.csv'
df = pd.read_csv(csv_file_path)

# Directory containing the images
new_folder_path = '/content/Dataset-images/test'

# List of unique image filenames from the CSV file
unique_images = df['Image'].unique()

# Create a list to hold the extracted data
extracted_data = []

# Iterate through each unique image filename/path
for image_filename in unique_images:
    # Construct the full image path
    image_path = os.path.join(new_folder_path, image_filename)

    # Check if image path is valid and exists
    if not os.path.isfile(image_path):
        print(f"Warning: Image path {image_path} does not exist. Skipping.")
        continue

    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Warning: Image {image_path} could not be loaded. Skipping.")
        continue

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Filter rows in the DataFrame that correspond to the current image
    image_df = df[df['Image'] == image_filename]

    # Iterate over the rows in the filtered DataFrame
    for index, row in image_df.iterrows():
        # Extract bounding box coordinates, ensuring they are within image bounds
        xmin = int(max(0, min(row['xmin'] * image_width, image_width - 1)))
        ymin = int(max(0, min(row['ymin'] * image_height, image_height - 1)))
        xmax = int(max(0, min(row['xmax'] * image_width, image_width)))
        ymax = int(max(0, min(row['ymax'] * image_height, image_height)))
        label_name = row['Class']

        # Extract the region of interest (ROI) from the image
        roi = image[ymin:ymax, xmin:xmax]

        # Check if the ROI is valid (non-empty)
        if roi.size == 0:
            print(f"Warning: Empty ROI detected for {label_name} in {image_filename}. Skipping OCR.")
            extracted_text = ""
        else:
            # Use OCR to extract text from the ROI
            extracted_text = pytesseract.image_to_string(roi)

        # Append the extracted data
        extracted_data.append({
            'image_name': image_filename,
            'label_name': label_name,
            'extracted_text': extracted_text.strip()
        })

# Convert extracted data to a DataFrame
extracted_df = pd.DataFrame(extracted_data)

# Save the DataFrame to an Excel file
excel_file_path = '/content/extracted_data.xlsx'
extracted_df.to_excel(excel_file_path, index=False)

print(f"Data saved to {excel_file_path}")
