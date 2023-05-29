import os

import cv2

# Path to the directory containing the images
image_folder = "C:/Users/Nay Abi Akl/Desktop/EPFL/Spring2023/Deep Learning for Autonomous Vehicles/Project/Results NEW/fpn_approach_1/6/"

# Output video file name
video_name = "output.mp4"

images = sorted(os.listdir(image_folder))  # Get a sorted list of image files

# Read the first image to get its size
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Define the video codec, frames per second (fps), and output video object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# Iterate over each image and write it to the video
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

# Release the video object and close the video file
video.release()
cv2.destroyAllWindows()
