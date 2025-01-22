import tensorflow as tf
from config import Config

def create_model():
    """
    Membuat arsitektur model CNN yang lebih dalam untuk 11 kelas
    """
    model = tf.keras.Sequential([
        # Layer Konvolusi 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer Konvolusi 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer Konvolusi 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer Konvolusi 4
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten layer
        tf.keras.layers.Flatten(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    return model