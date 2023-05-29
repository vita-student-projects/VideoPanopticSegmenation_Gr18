import os

# Folder path containing the images
folder_path = "/home/mahassan/waymo_out/video_sequence/train"

# List all files in the folder
file_list = os.listdir(folder_path)

# Set to store image names
image_names = set()

# Iterate through the file list
for filename in file_list:
    # Check if the file ends with 'panoptic.png' or 'semantic.png'
    if filename.endswith("panoptic.png") or filename.endswith("leftImg8bit.png"):
        # Extract the image name without the extension and suffix
        image_name = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        image_names.add(image_name)

# Find image names without pairs
missing_pairs = []
for image_name in image_names:
    if (
        image_name + "_panoptic.png" not in file_list
        or image_name + "_leftImg8bit.png" not in file_list
    ):
        missing_pairs.append(image_name)

# Print image names without pairs
if missing_pairs:
    print("Images without pairs:")
    for image_name in missing_pairs:
        print(image_name)
else:
    print("All images have pairs.")
