import os
import shutil
import random

SOURCE_DIR = "../raw"
TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
TEST_DIR = os.path.join(SOURCE_DIR, "test")
SPLIT_RATIO = 0.8  # 80% training, 20% testing

def prepare_data():
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name in ['train', 'test']:
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_point = int(len(images) * SPLIT_RATIO)

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in images[:split_point]:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in images[split_point:]:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

    print("✅ Dataset split completed!")

if __name__ == "__main__":
    prepare_data()
