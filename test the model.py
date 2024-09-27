output_directory = 'inference_graph'
labelmap_path = '/content/label_map.pbtxt'
import tensorflow as tf
from object_detection.utils import label_map_util
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'/content/kol/content/inference_graph/saved_model')


import pandas as pd
import os

# Read the CSV file
test = pd.read_csv('/content/test_labels.csv')

# Get all filenames from the CSV file
csv_images = list(test['filename'])

# Directory containing the images
directory = '/content/Dataset-images/test/'

# List all files in the directory
all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Filter the files to get only those listed in the CSV
images = [f for f in all_files if f in csv_images]

# Print the filenames
print(images)



for image_name in images:
    image_np = load_image_into_numpy_array('/content/Dataset-images/test/' + image_name)
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    display(Image.fromarray(image_np))



import pandas as pd

rows = []

final_dataframe = pd.DataFrame(columns=['Image','Class','Xmin','Ymin','Xmax','Ymax'])
for image_name in images:

    image_np = load_image_into_numpy_array('/content/Dataset-images/test/' + image_name)

    output_dict = run_inference_for_single_image(model, image_np)

    # store boxes in dataframe!
    cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

    for j in range(cut_off_scores):
        name = image_name
        score = output_dict['detection_scores'][j]
        classes = output_dict['detection_classes'][j]
        for i in range(1, len(category_index) + 1):
            if output_dict['detection_classes'][j] == category_index[i]['id']:
                classes = category_index[i]['name']
        ymin = output_dict['detection_boxes'][j][0]
        xmin = output_dict['detection_boxes'][j][1]
        ymax = output_dict['detection_boxes'][j][2]
        xmax = output_dict['detection_boxes'][j][3]

        row = [name, classes, xmin, ymin, xmax, ymax, score]
        rows.append(row)

# Convert to DataFrame
final_df = pd.DataFrame(rows, columns=['Image', 'Class', 'xmin', 'ymin', 'xmax', 'ymax', 'Score'])

# Remove duplicates based on class and coordinates (with some tolerance for minor variations)
final_df = final_df.drop_duplicates(subset=['Class', 'xmin', 'ymin', 'xmax', 'ymax'])

# Save to CSV
final_df.to_csv('predicted_coordinates.csv', index=False)

final_df
