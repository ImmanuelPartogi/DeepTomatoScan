import tensorflow as tf
from data_loader import load_data
from utils import plot_confusion_matrix, plot_training_history
import numpy as np

def evaluate_model(model, test_generator):
    """
    Evaluasi model dan tampilkan metrik detail
    """
    # Evaluasi model
    scores = model.evaluate(test_generator)
    print("\nModel Evaluation:")
    print(f"Test Loss: {scores[0]:.4f}")
    print(f"Test Accuracy: {scores[1]:.4f}")
    
    # Buat confusion matrix
    plot_confusion_matrix(model, test_generator)
    
    # Prediksi pada beberapa sampel
    predictions = model.predict(test_generator)
    class_names = list(test_generator.class_indices.keys())
    
    # Tampilkan contoh prediksi
    print("\nContoh Prediksi:")
    for i in range(5):
        true_class = class_names[np.argmax(test_generator.labels[i])]
        pred_class = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        print(f"Sample {i+1}:")
        print(f"True Class: {true_class}")
        print(f"Predicted Class: {pred_class}")
        print(f"Confidence: {confidence:.2f}%\n")