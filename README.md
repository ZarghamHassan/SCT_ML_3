# Cats vs. Dogs Classifier

## Project Overview

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using a dataset from Kaggle. The model is trained to distinguish between the two categories based on image features extracted from resized images.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Results](#results)
- [License](#license)

## Dataset

The dataset used in this project can be downloaded from Kaggle:

- [Cats vs. Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

Please download the dataset and extract it into the `dogs-vs-cats` folder within the project directory.

## Installation

To run this project, you'll need to have Python installed along with the required libraries. You can create a virtual environment and install the necessary packages using the following commands:

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

## Usage

1. **Load and Train the Model**: Run the following command to train the model:
   ```bash
   python prediction_model.py
This script will:

Load images from the training dataset.
Resize and normalize the images.
Split the dataset into training and validation sets.
Train the SVM model using the training set.
Evaluate the model's performance on the validation set and print the results.
Save the trained model to a file named svm_cats_dogs_model.pkl.
Make Predictions: To classify a single image, modify the image_path variable in predict.py to point to your image and run:

```bash
Copy code
python predict.py
This script will:

Load the trained model from svm_cats_dogs_model.pkl.
Preprocess the specified image.
Print the prediction result (either "Cat" or "Dog").
Graphical User Interface: For an easy-to-use interface, run:

bash
Copy code
python gui.py
The GUI will allow you to:

Load an image file.
Display the uploaded image.
Show the prediction result on the screen.
