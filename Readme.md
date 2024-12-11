#  Lumbar-Spine-Degenerative-Classification

The goal is to predict the severity of lumbar spine degeneration from medical images. The project involves processing DICOM medical images, applying machine learning algorithms, and training deep learning models to classify the severity of spine degeneration into three classes: **Normal/Mild**, **Moderate**, and **Severe**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Training](#training)
6. [Model Architecture](#model-architecture)
7. [License](#license)
8. [References](#references)

## Project Overview

The Lumbar Spine Degenerative Classification project involves:
- Reading and processing DICOM files to extract medical images of the lumbar spine.
- Preprocessing images and data.
- Creating custom datasets for training machine learning models.
- Training EfficientNetV2 models for classifying lumbar spine degeneration severity.
- Evaluating the performance of the model with visualizations.

### Task
- **Objective**: Classify medical images of lumbar spine degeneration into three categories: **Normal/Mild**, **Moderate**, and **Severe**.
- **Input**: DICOM images from the RSNA dataset.
- **Output**: Severity of degeneration for each image.

## Requirements

To run this project, you need to install the following dependencies:

```bash
pip install torch torchvision tqdm pandas matplotlib pydicom scikit-learn
```

### Additional Tools
- **CUDA**: If you're using a GPU, make sure CUDA is installed to speed up model training.
- **PyTorch**: The project uses PyTorch for deep learning model development.
- **DICOM**: The images are in DICOM format and require the `pydicom` library to handle them.

## Dataset

The dataset contains DICOM images of lumbar spines with associated metadata:
- **train.csv**: Metadata for training images.
- **train_label_coordinates.csv**: Coordinates of specific regions of interest (e.g., severity level markers).
- **train_series_descriptions.csv**: Descriptions of the medical image series.
- **test_series_descriptions.csv**: Descriptions of the test image series.

## Usage

### Step 1: Setup

Make sure you have installed all required libraries from the **Requirements** section.

### Step 2: Load the Data

The dataset can be loaded by using the `load_dicom()` function that reads the DICOM images and normalizes them. The images are then split into training and validation sets.

### Step 3: Preprocess the Data

The DICOM images are preprocessed using the following steps:
1. Resizing images to 224x224.
2. Converting images to grayscale.
3. Normalizing the pixel values to [0, 1].

### Step 4: Train the Model

The model is built using **EfficientNetV2** for image classification. The training loop uses **CrossEntropyLoss** as the loss function and **Adam** as the optimizer.

```bash
python src/train.py
```

## Training

The model is based on **EfficientNetV2**, which is a highly efficient convolutional neural network architecture. The key steps in the training pipeline are as follows:
1. Load and preprocess the images.
2. Split the dataset into training and validation sets.
3. Train the model with the **Adam optimizer** and **CrossEntropyLoss**.
4. Monitor the validation accuracy and loss to prevent overfitting.
5. Save the best-performing model.

## Model Architecture

The model is based on the EfficientNetV2 architecture, which has been fine-tuned for this task. It performs well on image classification tasks by using a combination of convolutional layers and attention mechanisms inspired by transformers.

```python
class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. RSNA 2024 Lumbar Spine Degenerative Classification Challenge: [https://www.rsna.org/](https://www.rsna.org/)
2. EfficientNetV2: [https://arxiv.org/abs/2104.00298](https://arxiv.org/abs/2104.00298)
3. PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
   
## Link for data: 
- `https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data`
