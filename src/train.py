import tensorflow as tf
from data_loader import load_data
from model import create_model
from config import Config
from utils import plot_training_history
from evaluate import evaluate_model
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_callbacks(model_dir='models'):
    """
    Setup training callbacks
    """
    # Buat direktori untuk menyimpan model jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        # Model checkpoint untuk menyimpan model terbaik
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    return callbacks

def train_model():
    """
    Train the CNN model with comprehensive logging and error handling
    """
    try:
        logger.info("Starting model training process...")
        
        # Load data
        logger.info("Loading and preparing data...")
        train_generator, test_generator = load_data()
        logger.info(f"Found {train_generator.samples} training samples")
        logger.info(f"Found {test_generator.samples} validation samples")
        
        # Create model
        logger.info("Creating model...")
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Setup callbacks
        callbacks = setup_callbacks()
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            train_generator,
            epochs=Config.EPOCHS,
            validation_data=test_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(history)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluate_model(model, test_generator)
        
        # Save final model
        final_model_path = os.path.join('models', 'final_model.h5')
        model.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        return history, model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """
    Main execution function
    """
    try:
        # Set memory growth for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        
        # Start training
        history, model = train_model()
        logger.info("Training completed successfully!")
        
        # Final message
        logger.info("\nTraining pipeline completed! You can find:")
        logger.info("- Best model at: models/best_model.h5")
        logger.info("- Final model at: models/final_model.h5")
        logger.info("- Training plots at: training_history.png")
        logger.info("- Training logs at: logs/")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()