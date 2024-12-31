import os
import shutil

# Define directories and file paths
source_folder = "labels"
destination_folder = "labels_val"
test_file_path = "annotation/mtsd_v2_fully_annotated/splits/val.txt"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the filenames from test.txt
with open(test_file_path, 'r') as file:
    filenames_to_move = {line.strip() for line in file}

# Move the matching files
for filename in filenames_to_move:
    source_file = os.path.join(source_folder, f"{filename}.txt")
    destination_file = os.path.join(destination_folder, f"{filename}.txt")
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)

f"Moved {len(filenames_to_move)} files to {destination_folder}."
