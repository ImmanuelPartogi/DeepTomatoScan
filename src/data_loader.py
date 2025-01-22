import tensorflow as tf
from config import Config

def load_data():
    """
    Memuat dan memproses dataset gambar
    """
    # Data augmentasi untuk training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Hanya rescaling untuk testing
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    # Memuat data training
    train_generator = train_datagen.flow_from_directory(
        Config.TRAIN_DIR,
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical'
    )

    # Memuat data testing
    test_generator = test_datagen.flow_from_directory(
        Config.TEST_DIR,
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, test_generator