import os

image_folder = "/home/mahassan/waymo_out/train"

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        # Extract the second series of numbers from the filename
        parts = filename.split("_")
        series_numbers = parts[1]

        if series_numbers.startswith("4") or series_numbers.startswith("5"):
            # Delete the image file
            os.remove(os.path.join(image_folder, filename))
            print(f"Deleted {filename}")
