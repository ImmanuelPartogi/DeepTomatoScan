import tensorflow as tf
import numpy as np
import logging
from utils import load_and_preprocess_image
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiseasePredictor:
    def __init__(self, model_path='models/best_model.h5'):
        """
        Initialize the disease predictor
        """
        try:
            # Periksa keberadaan file model
            if not os.path.exists(model_path):
                available_models = self._find_available_models()
                if available_models:
                    # Gunakan model yang tersedia
                    model_path = available_models[0]
                    logger.info(f"Using available model: {model_path}")
                else:
                    raise FileNotFoundError(
                        "\nModel tidak ditemukan! Pastikan:\n"
                        "1. Anda sudah menjalankan training dengan 'python src/train.py'\n"
                        "2. File model tersimpan di folder 'models/'\n"
                        "3. Jika menggunakan path kustom, pastikan path sudah benar"
                    )
            
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.class_names = self._get_class_names()
            logger.info("Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            raise

    def _find_available_models(self):
        """
        Mencari model-model yang tersedia di direktori
        """
        model_files = []
        possible_locations = ['models/', './', '../models/']
        
        for location in possible_locations:
            if os.path.exists(location):
                for file in os.listdir(location):
                    if file.endswith('.h5'):
                        model_files.append(os.path.join(location, file))
        
        return model_files
    
    def _get_class_names(self):
        """
        Get class names from training directory
        """
        train_dir = 'data/train'
        if not os.path.exists(train_dir):
            # Jika direktori train tidak ada, gunakan nama kelas default
            logger.warning("Training directory not found. Using default class names.")
            return [
                'Bacterial_spot', 'Early_blight', 'Late_blight', 
                'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites', 
                'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato_mosaic_virus', 'Healthy'
            ]
        return sorted(os.listdir(train_dir))
    
    def predict_image(self, image_path):
        """
        Predict disease from image
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Gambar tidak ditemukan di {image_path}\n"
                    "Pastikan path gambar sudah benar."
                )
            
            # Load and preprocess image
            processed_image = load_and_preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Get class name
            predicted_class = self.class_names[predicted_class_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                (self.class_names[idx], predictions[0][idx] * 100)
                for idx in top_3_idx
            ]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def predict_example():
    """
    Example usage of the predictor
    """
    try:
        print("\n=== Deteksi Penyakit Tanaman Tomat ===")
        print("Pastikan model sudah dilatih dengan menjalankan 'python src/train.py' terlebih dahulu")
        
        # Initialize predictor
        predictor = DiseasePredictor()
        
        while True:
            try:
                # Example prediction
                print("\nMasukkan path gambar (atau 'q' untuk keluar):")
                image_path = input("> ")
                
                if image_path.lower() == 'q':
                    break
                    
                result = predictor.predict_image(image_path)
                
                # Display results
                print("\nHasil Prediksi:")
                print(f"Kelas yang diprediksi: {result['predicted_class']}")
                print(f"Tingkat keyakinan: {result['confidence']:.2f}%")
                
                print("\nTop 3 Prediksi:")
                for class_name, confidence in result['top_3_predictions']:
                    print(f"{class_name}: {confidence:.2f}%")
            
            except FileNotFoundError as e:
                print(f"\nError: {str(e)}")
                continue
            except KeyboardInterrupt:
                print("\nProgram dihentikan oleh pengguna.")
                break
            
    except Exception as e:
        logger.error(f"Prediction example failed: {str(e)}")
        raise

if __name__ == "__main__":
    predict_example()