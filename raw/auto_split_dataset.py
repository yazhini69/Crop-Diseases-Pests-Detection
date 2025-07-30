import os
import shutil
import random

# Path to your dataset (all current class folders are inside here)
DATASET_DIR = "."   # <- put all your original folders here (PlantVillage + IP102)
OUTPUT_DIR = "dataset/split"  # <- output will be created here

# Train/Val/Test split ratio
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def create_dirs(base_path, classes):
    """Create train/val/test directories with class subfolders"""
    for split in SPLIT_RATIOS.keys():
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)

def split_dataset():
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"🔹 Found {len(classes)} classes")

    create_dirs(OUTPUT_DIR, classes)

    for cls in classes:
        class_path = os.path.join(DATASET_DIR, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(SPLIT_RATIOS["train"] * n_total)
        n_val = int(SPLIT_RATIOS["val"] * n_total)

        for i, img in enumerate(images):
            src = os.path.join(class_path, img)
            if i < n_train:
                dst = os.path.join(OUTPUT_DIR, "train", cls, img)
            elif i < n_train + n_val:
                dst = os.path.join(OUTPUT_DIR, "val", cls, img)
            else:
                dst = os.path.join(OUTPUT_DIR, "test", cls, img)
            shutil.copy(src, dst)

        print(f"✅ {cls}: {n_total} images → train:{n_train}, val:{n_val}, test:{n_total-n_train-n_val}")

if __name__ == "__main__":
    split_dataset()
