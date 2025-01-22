class Config:
    # Parameter dasar
    IMAGE_SIZE = 224  # Ukuran gambar
    BATCH_SIZE = 32   # Jumlah gambar per batch
    EPOCHS = 10       # Jumlah epoch training
    NUM_CLASSES = 11  # Jumlah kelas yang ditemukan di dataset
    
    # Path dataset
    TRAIN_DIR = 'data/train'
    TEST_DIR = 'data/test'