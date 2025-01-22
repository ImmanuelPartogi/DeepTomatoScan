# 🍅 Tomato Plant Disease Detection

An AI-powered system to detect diseases in tomato plants using computer vision and deep learning.

## 📋 Features

- Detects multiple tomato plant diseases
- Built with modern deep learning architecture
- Easy-to-use command line interface
- High accuracy disease classification

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Dataset Preparation

1. Organize your tomato plant into the following structure, as an example:
```
data/
├── healthy/          # Healthy tomato leaf images
├── leaf_spot/        # Leaf spot disease images
└── leaf_blight/      # Leaf blight disease images
```

2. Each category should contain:
   - Clear, well-lit images
   - Consistent image resolution
   - Multiple angles of the leaves
   - Various disease stages (for diseased samples)

## 💻 Usage

### Training

Train the model on your dataset:

```bash
python src/train.py
```

The training process will:
- Split data into training and validation sets
- Train the disease detection model
- Save the best performing model
- Generate training metrics and charts

### Prediction

Analyze new tomato plant images:

```bash
python src/predict.py --image path/to/your/image.jpg
```

## 📊 Results

The model will classify the input image into one of three categories:
- Healthy (✅)
- Leaf Spot Disease (🔴)
- Leaf Blight Disease (🔴)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the NERO License.

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository.

---
Made with ❤️ for healthier tomato plants 🍅