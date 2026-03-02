import os

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError as ie:
    raise ImportError(
        "TensorFlow must be installed to run data_preprocessing.py. "
        "Use 'pip install tensorflow' in the appropriate environment."
    ) from ie

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Training data generator with minimal augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,        # minimal rotation
    horizontal_flip=True      # light augmentation
)

# Validation data generator (no augmentation)
valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


def build_generators(train_path="dataset/train", valid_path="dataset/valid"):
    """Create and return train/validation generators, verifying directories.

    Raises:
        FileNotFoundError: if a required folder is missing.
    """
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"training directory not found: {train_path}")
    if not os.path.isdir(valid_path):
        raise FileNotFoundError(f"validation directory not found: {valid_path}")

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    valid_gen = valid_datagen.flow_from_directory(
        valid_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, valid_gen


if __name__ == "__main__":
    try:
        tr, val = build_generators()
        print(f"train: {tr.samples} images, {tr.num_classes} classes")
        print(f"valid: {val.samples} images, {val.num_classes} classes")
    except Exception as exc:
        import sys
        print("Error during data preprocessing:", exc)
        sys.exit(1)
