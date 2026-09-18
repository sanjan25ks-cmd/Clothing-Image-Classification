import os

classes = [
    "dress",
    "jacket",
    "jeans",
    "shirt",
    "tshirt"
]

folders = [
    "dataset/raw",
    "dataset/train",
    "dataset/validation"
]

print("\n======================================")
print("       CLOTHING DATASET CHECK")
print("======================================")

for folder in folders:

    print(f"\n📁 {folder}")

    total = 0

    for class_name in classes:

        class_folder = os.path.join(
            folder,
            class_name
        )

        if os.path.exists(class_folder):

            files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            count = len(files)
            total += count

            print(
                f"{class_name:10s} : {count} images"
            )

        else:

            print(
                f"{class_name:10s} : FOLDER NOT FOUND"
            )

    print(f"Total       : {total}")

print("\n======================================")
print("CHECK COMPLETED")
print("======================================")