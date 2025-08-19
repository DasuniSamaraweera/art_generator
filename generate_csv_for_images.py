import os
import csv

# Path to your images folder and CSV
images_folder = "data/images"
csv_path = "data/art_references.csv"

# Choose a style label for all new images (edit as needed)
default_style = "Unlabeled"

# Get all image files in the folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Read existing CSV entries to avoid duplicates
existing = set()
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(os.path.basename(row["image_path"]))

# Write new entries for images not already in the CSV
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write header if file is empty
    if os.stat(csv_path).st_size == 0:
        writer.writerow(["image_path", "style"])
    for img in image_files:
        if img not in existing:
            writer.writerow([os.path.join(images_folder, img), default_style])
            print(f"Added {img} to CSV with style '{default_style}'")
