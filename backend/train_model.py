import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Dataset directories (adjust if needed)
TRAIN_DIR = "backend/dataset/split/train"
VAL_DIR = "backend/dataset/split/val"  # or test, whichever you want to use for validation

print("Train directory exists?", os.path.exists(TRAIN_DIR))
print("Validation directory exists?", os.path.exists(VAL_DIR))

# Data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save the model after training
output_path = os.path.join( "models", "plant_disease_model.h5")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create folder if it doesn't exist
model.save(output_path)
print(f"✅ Model saved to {output_path}")
