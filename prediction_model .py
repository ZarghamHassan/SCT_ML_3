import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib  # To save and load the trained model
import random

# Function to load and preprocess the images
def load_data(directory, categories, has_labels=True, limit=None):
    flat_data_arr = []
    target_arr = []
    print(f"Loading data from: {directory}")
    
    # List all files in the directory
    image_files = os.listdir(directory)
    
    # Shuffle and limit the dataset size if necessary
    if limit:
        image_files = random.sample(image_files, limit)
    
    # Counter for tracking progress
    image_count = 0
    total_images = len(image_files)
    
    for img in image_files:
        try:
            img_array = imread(os.path.join(directory, img))
            img_resized = resize(img_array, (64, 64, 3), preserve_range=True)  # Resize images to 64x64 for memory efficiency
            flat_data_arr.append(img_resized.flatten())  # Flatten the image for SVM
            
            if has_labels:  # Only append labels if the dataset has labels
                if 'cat' in img:
                    target_arr.append(0)
                elif 'dog' in img:
                    target_arr.append(1)
                else:
                    print(f"Error: Unknown category for image '{img}'")
            image_count += 1
            
            # Print progress for every 500 images
            if image_count % 500 == 0 or image_count == total_images:
                print(f"Processed {image_count}/{total_images} images...")
                
        except IOError as e:
            print(f"Error loading image: {img}")
            print(str(e))
    
    flat_data_arr = np.array(flat_data_arr)  # Convert to numpy array
    flat_data_arr = flat_data_arr / 255.0  # Normalize pixel values to [0, 1]
    
    print(f"Finished loading {image_count} images.")
    
    if has_labels:
        return flat_data_arr, np.array(target_arr)
    return flat_data_arr

# Function to train the SVM model
def train_model(X, y):
    print("Starting grid search...")
    param_grid = {'C': [1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=3, scoring='accuracy')  # Reduced cross-validation folds for speed
    grid.fit(X, y)
    print("Grid search complete.")
    print(f"Best parameters: {grid.best_params_}")
    
    # Save the trained model
    joblib.dump(grid, 'svm_cats_dogs_model.pkl')
    print("Model saved as 'svm_cats_dogs_model.pkl'")
    
    return grid

# Function to evaluate the model
def evaluate_model(model, X, y, target_names):
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred, average='weighted'))
    print("Recall:", recall_score(y, y_pred, average='weighted'))
    print("F1-score:", f1_score(y, y_pred, average='weighted'))
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=target_names))

# Function to load the saved SVM model
def load_saved_model(filepath):
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Define the categories and data directories
categories = ['cats', 'dogs']
train_dir = r'C:\Users\syedz\OneDrive\Desktop\Codes\cats and dogs\dogs-vs-cats\train'
test_dir = r'C:\Users\syedz\OneDrive\Desktop\Codes\cats and dogs\dogs-vs-cats\test1'

# Set a limit on the number of images for testing (e.g., 1000 images)
image_limit = 1000  # Adjust or remove limit for full dataset

# Load the training data
X_train, y_train = load_data(train_dir, categories, limit=image_limit)
print("Training data loaded.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Data split into training and validation sets.")

# Train an SVM model and save it
model = train_model(X_train, y_train)

# Evaluate the performance of the SVM model on the validation set
evaluate_model(model, X_val, y_val, target_names=['cats', 'dogs'])

# Load the testing data (test set doesn't have labels)
X_test = load_data(test_dir, categories, has_labels=False, limit=100)  # Limit test data to 100 for now

# Load the previously saved model for prediction
svm_model = load_saved_model('svm_cats_dogs_model.pkl')

# Make predictions on the test set using the loaded model
if svm_model:
    y_test_pred = svm_model.predict(X_test)
    print("Predictions for the test set:", y_test_pred[:10])  # Show the first 10 predictions
