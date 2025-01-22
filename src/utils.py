import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import logging
import time

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration class
class Config:
    IMAGE_SIZE = 224  # Sesuaikan ukuran dengan model Anda

def plot_training_history(history, save_path="training_history.png"):
    """
    Memvisualisasikan hasil training model.
    
    Parameters:
    - history: History object dari model training.
    - save_path: Lokasi untuk menyimpan gambar plot.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        save_path = f"{os.path.splitext(save_path)[0]}_{int(time.time())}.png"
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Training history saved to {save_path}")
    except Exception as e:
        logging.error(f"Error in plot_training_history: {e}")

def plot_confusion_matrix(model, test_generator, save_path="confusion_matrix.png"):
    """
    Membuat confusion matrix dari prediksi model pada data uji.
    
    Parameters:
    - model: Model yang sudah dilatih.
    - test_generator: Generator data uji.
    - save_path: Lokasi untuk menyimpan gambar confusion matrix.
    """
    try:
        # Prediksi pada data test
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Buat confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_generator.class_indices.keys(),
                    yticklabels=test_generator.class_indices.keys(),
                    annot_kws={"size": 10})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        save_path = f"{os.path.splitext(save_path)[0]}_{int(time.time())}.png"
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logging.error(f"Error in plot_confusion_matrix: {e}")

def load_and_preprocess_image(image_path):
    """
    Memuat dan memproses gambar untuk prediksi.
    
    Parameters:
    - image_path: Lokasi file gambar.
    
    Returns:
    - Numpy array gambar yang sudah diproses.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
        
        img = image.load_img(image_path, target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisasi
        logging.info(f"Gambar berhasil dimuat dan diproses: {image_path}")
        return img_array
    except Exception as e:
        logging.error(f"Error in load_and_preprocess_image: {e}")
        raise
