# Dog Classifier

A machine learning project for identifying and classifying different dog breeds from images using deep learning techniques.

## Project Overview

This project implements a dog breed classification system using convolutional neural networks (CNNs). The classifier is trained on a dataset of dog images and can identify multiple breeds with high accuracy. The system includes preprocessing pipelines, model training workflows, and evaluation tools.

## Project Structure

```
DOG-CLASSIFIER/
├── app/                      # Application directory
│   ├── api/                  # API endpoints for serving the model
│   │   ├── main.py           # Main API entry point
│   │   └── predict.py        # Prediction API functions
│   ├── data/                 # Data directory (not included in repo)
│   ├── models/               # Trained model files directory
│   └── utils/                # Utility functions
├── models                    
├── preprocess.py             # Data preprocessing functions
├── train.py                  # Model training script
├── kaggle_dataset/           # Location for Kaggle dataset (not included in repo)
├── tests/                    # Testing scripts
│   ├── test.py               # Basic model testing
│   └── evalution.py          # Model evaluation metrics
├── organize_data.py          # Script for organizing dataset
├── predict.py                # Standalone prediction script
├── rename_image.py           # Utility for renaming image files
├── requirements.txt          # Required Python packages
└── test.jpg                  # Sample test image
```

## Installation and Usage

### 1. Install Dependencies

First, ensure you have Python 3.7+ installed. Then install all required libraries using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including TensorFlow, NumPy, Pandas, Matplotlib, and other dependencies required for the project.

### 2. Prepare Dataset

The project expects a dataset organized in a specific structure. If your data isn't already organized:

```bash
python -m organize_data.py
```

The dataset should be organized as follows:
```
app/data/
├── train/
│   ├── bangkeaw/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── german_shepherd/
│   │   └── ...
│   └── ...
└── test/
│   ├── bangkeaw/
│   │   └── ...
│   └── ...
└── val/
    ├── bangkeaw/
    │   └── ...
    └── ...
```

### 3. Train the Model

The training script will process the dataset and train the CNN model. This will generate three model files in the `app/models` folder:

```bash
python -m app.train
```

The training process includes:
- Data preprocessing and augmentation
- Model training with specified hyperparameters
- Regular checkpointing
- Validation on a separate dataset
- Saving the final model

Training output files:
- Model architecture file
- Trained weights file
- Class mapping file

### 4. Test Model Loading

To verify that the model can be loaded correctly:

```bash
python -m tests.test
```

This script checks if:
- All model files exist
- The model can be loaded without errors
- The model architecture matches expectations

### 5. Evaluate Model Confidence

Run the evaluation script to check the model's performance metrics:

```bash
python -m tests.evalution
```

This provides detailed metrics including:
- Accuracy
- Precision and recall
- F1 score
- Confusion matrix
- Classification report per breed

### 6. Test with Sample Image

Test the model with the provided sample image:

```bash
python -m predict
```

This will:
1. Load the trained model
2. Preprocess the test.jpg image
3. Make a prediction
4. Display the image with the predicted breed and confidence score

### 7. API Usage (Optional)

To start the API server for model serving:

```bash
python -m app.api.main
```

The API will be available at `http://localhost:8000` with the following endpoints:
- `POST /predict`: Send an image file to get prediction results

## Model Details

The classifier uses a deep convolutional neural network architecture based on [specify architecture, e.g., ResNet50, MobileNet, etc.] with transfer learning. The model is fine-tuned on the dog breed dataset after being pre-trained on ImageNet.

Key features:
- Input image size: 224x224x3
- Output: Probabilities for [X] different dog breeds
- Training: Using Adam optimizer with learning rate scheduling
- Data augmentation: Random flips, rotations, and color adjustments to improve generalization

## Requirements

Major dependencies include:
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn

See `requirements.txt` for a complete list of required Python packages.

## Performance

The model achieves approximately [X]% accuracy on the test set, with the following performance characteristics:
- Top-1 accuracy: [X]%
- Top-5 accuracy: [X]%
- Average inference time: [X] ms per image

## Troubleshooting

Common issues:
- **Missing model files**: Ensure training has completed successfully
- **CUDA/GPU errors**: Check TensorFlow GPU configuration
- **Memory errors during training**: Reduce batch size
- **Import errors**: Verify all dependencies are installed correctly

## Future Improvements

Planned enhancements:
- Implement additional model architectures for comparison
- Add web interface for easy image upload and testing
- Improve performance on minority breed classes
- Deploy model to mobile platforms
