import os
import shutil
import random


RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset"

CLASSES = ["tshirt", "shirt", "jeans", "dress", "jacket"]

TRAIN_RATIO = 0.8


for class_name in CLASSES:

    raw_class_folder = os.path.join(RAW_DIR, class_name)
    train_folder = os.path.join(OUTPUT_DIR, "train", class_name)
    validation_folder = os.path.join(OUTPUT_DIR, "validation", class_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)

    if not os.path.exists(raw_class_folder):
        print(f"Folder not found: {raw_class_folder}")
        continue

   
    images = []

    for file in os.listdir(raw_class_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(file)

   
    random.shuffle(images)

    
    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    validation_images = images[train_count:]

    
    for image in train_images:
        source = os.path.join(raw_class_folder, image)
        destination = os.path.join(train_folder, image)
        shutil.copy2(source, destination)

    
    for image in validation_images:
        source = os.path.join(raw_class_folder, image)
        destination = os.path.join(validation_folder, image)
        shutil.copy2(source, destination)

    print("--------------------------------")
    print("Class:", class_name)
    print("Total:", len(images))
    print("Train:", len(train_images))
    print("Validation:", len(validation_images))

print("\n================================")
print("DATASET PREPARATION COMPLETED")
print("================================")