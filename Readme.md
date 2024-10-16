#  Lumbar-Spine-Degenerative-Classification - 15 Oct 24

This project implements a DICOM image processing pipeline that extracts regions of interest from medical images, applies severity analysis, and prepares data for further machine learning tasks as of Oct 15,2024.

## Features
- Load and preprocess DICOM images.
- Extract specified regions based on coordinates.
- Apply image enhancement techniques.
- Visualize images and extracted regions.
- Encode categorical labels for machine learning.
- Prepare datasets for model training and testing.


## Usage
1. Set the paths for the CSV files containing training and test data at the beginning of `main.ipynb`:
   ```python
   train_path = r'path_to/train.csv'
   train_label_coordinates_path = r'path_to/train_label_coordinates.csv'
   train_descriptions_path = r'path_to/train_series_descriptions.csv'
   test_descriptions_path = r'path_to/test_series_descriptions.csv'
   ```


## Functions
### `load_dicom_image(path)`
Loads a DICOM image from the specified path and normalizes it to an 8-bit format.

### `extract_region(image, x, y, width=128, height=128)`
Extracts a square region of interest from the image centered at `(x, y)`.

### `draw_rectangle(image, x_coord, y_coord, size, color, label)`
Visualizes the original image with a highlighted rectangle around the specified coordinates.

### `draw_severe(image, x_coord, y_coord, severity)`
Applies color mapping based on severity levels and visualizes the result.

### `load_images_from_study(df, folder)`
Loads images and associated metadata from a specified folder based on the study data.

## Data Sources
Ensure that the following CSV files are present in the `Csv` directory:
- `train.csv`: Contains training data with study IDs and associated conditions.
- `train_label_coordinates.csv`: Contains coordinates for extracting regions from the images.
- `train_series_descriptions.csv`: Contains descriptions related to the series of images.
- `test_series_descriptions.csv`: Contains test series descriptions.

## Link for data: 
- `https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data`
