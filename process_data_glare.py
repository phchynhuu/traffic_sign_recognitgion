import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

# Define paths
dataset_dir = 'images'
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

# Create directories for YOLO format
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/test'), exist_ok=True)

# Class mapping
class_mapping = {
    'bicycleCrossing': 0,
    'school': 1,
    'roadworkAhead': 2,
    'workersAhead': 3,
    'oneWay': 4,
    'endRoadwork': 5,
    'keepRight': 6,
    'speedLimit45': 7,
    'pedestrianCrossing': 8,
    'doNotBlock': 9,
    'doNotEnter': 10,
    'noLeftTurn': 11,
    'noUTurn': 12,
    'speedLimit25': 13,
    'stop': 14,
    'laneEnds': 15,
    'signalAhead': 16,
    'speedLimit30': 17,
    'keepLeft': 18,
    'addedLane': 19,
    'doNotStop': 20,
    'exitSpeedAdvisory25': 21,
    'exitSpeedAdvisory30': 22,
    'exitSpeedAdvisory45': 23,
    'speedLimit55': 24,
    'merge': 25,
    'rampSpeedAdvisory30': 26,
    'rampSpeedAdvisory25': 27,
    'shiftLeft': 28,
    'shiftRight': 29,
    'turnRight': 30,
    'speedLimit55Ahead': 31,
    'speedLimit65': 32,
    'noLeftOrUTurn': 33,
    'noRightTurn': 34,
    'yield': 35,
    'curveRightOnly': 36,
    'speedLimit35': 37,
    'curveLeftOnly': 38,
    'curveLeft': 39,
    'speedLimit40': 40
}

# Process each video folder
all_images = []
all_labels = []

for video_folder in os.listdir(dataset_dir):
    video_path = os.path.join(dataset_dir, video_folder)
    if not os.path.isdir(video_path):
        continue

    # Read frameAnnotations.csv
    annotation_file = os.path.join(video_path, 'frameAnnotations.csv')
    if not os.path.exists(annotation_file):
        print(f"Missing annotations in {video_folder}")
        continue

    df = pd.read_csv(annotation_file)

    # Process each row in the annotations
    for _, row in df.iterrows():
        filename = row['Filename']
        label = row['Annotation tag']
        x_min = row['Upper left corner X']
        y_min = row['Upper left corner Y']
        x_max = row['Lower right corner X']
        y_max = row['Lower right corner Y']

        # Skip if label is not in the class mapping
        if label not in class_mapping:
            continue

        # Get image path
        image_path = os.path.join(video_path, filename)
        if not os.path.exists(image_path):
            print(f"Image {filename} not found in {video_folder}")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Normalize bounding box coordinates
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width_bbox = (x_max - x_min) / img_width
        height_bbox = (y_max - y_min) / img_height

        # Prepare YOLO annotation
        class_id = class_mapping[label]
        annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_bbox:.6f} {height_bbox:.6f}\n"

        # Save annotation file
        annotation_file_path = os.path.join(
            output_dir, 'labels', os.path.splitext(filename)[0] + '.txt'
        )
        with open(annotation_file_path, 'a') as f:
            f.write(annotation_line)

        # Add image and annotation paths to lists
        all_images.append(image_path)
        all_labels.append(annotation_file_path)

# Split dataset into train and test
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

# Function to move files to their respective directories
def move_files(files, dest_folder):
    for file in files:
        dest_path = os.path.join(dest_folder, os.path.basename(file))
        shutil.copy(file, dest_path)

# Move files to train and test folders
move_files(train_images, os.path.join(output_dir, 'images/train'))
move_files(test_images, os.path.join(output_dir, 'images/test'))
move_files(train_labels, os.path.join(output_dir, 'labels/train'))
move_files(test_labels, os.path.join(output_dir, 'labels/test'))

print("Train-test split completed!")
