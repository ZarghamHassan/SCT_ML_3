# Cats vs. Dogs Classifier

## Project Overview

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using a dataset from Kaggle. The model is trained to distinguish between the two categories based on image features extracted from resized images.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Results](#results)

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
```

## Usage
1.Load and Train the Model: Run the prediction_model.py file to load the dataset, train the SVM model, and save it for future use.


```bash
python prediction_model.py
```
2.Make Predictions: Use the predict.py file to classify new images. Update the image_path variable to point to your image.


```bash
python predict.py

```
3.Graphical User Interface: Run the gui.py file to launch a simple GUI for image classification.


```bash
python gui.py   

```

## Files
prediction_model.py: Contains the code for loading data, training the SVM model, and saving it.
predict.py: Used for making predictions on a single image.
gui.py: Provides a graphical user interface for the classifier.
requirements.txt: Lists the Python packages required to run this project.



