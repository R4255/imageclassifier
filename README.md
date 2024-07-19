# Animal Heads Image Classifier

This repository contains an image classification application that identifies and classifies images of animal heads into 20 different classes. The project involves various stages of image preprocessing, feature extraction, and model training.

## Dataset Description

- **Classes**: 20 different animal heads
- **Total Images**: 2057
- **Image Size**: 80x80 pixels, RGB format

## Built With

- **Flask Framework**: For handling HTTP requests and serving the web application.
- **TensorFlow and Keras**: For building and training the convolutional neural network model.
- **Pickle**: For serializing the model and integrating it into the Flask application.
- **Render**: For hosting and deploying the web application.
- **Scikit-learn**: For machine learning and feature extraction.

## Project Overview

The project involves:
1. Preprocessing images to grayscale.
2. Extracting features using Histogram of Oriented Gradients (HOG).
3. Normalizing features and training a Stochastic Gradient Descent (SGD) classifier.
4. Evaluating the model using Grid Search for hyperparameter tuning.
5. Deploying the model in a Flask application for image classification.

## Code Overview

### HOG Feature Extraction

A custom transformer class for extracting HOG features from images:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import skimage.feature

class hogtransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def local_hog(img):
            hog_features = skimage.feature.hog(img, orientations=self.orientations,
                                               pixels_per_cell=self.pixels_per_cell,
                                               cells_per_block=self.cells_per_block)
            return hog_features
        
        hfeatures = np.array([local_hog(x) for x in X])
        return hfeatures

##**Pipeline for Model Training**
**A pipeline that integrates grayscale conversion, HOG feature extraction, scaling, and classification:**
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray

model_pipe = Pipeline([
    ('grayscale', rgb2gray_transform()),
    ('hogtransform', hogtransformer(orientations=8, pixels_per_cell=(10, 10), cells_per_block=(3, 3))),
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(loss='hinge', learning_rate='adaptive', early_stopping=True, eta0=0.1))
])

model_pipe.fit(x_train, y_train)
##**Hyperparameter Tuning with Grid Search**
**Optimize hyperparameters using Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

estimator = Pipeline([
    ('grayscale', rgb2gray_transform()),
    ('hogtransform', hogtransformer()),
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier())
])

param_grid = [
    {
        'hogtransform__orientations': [7, 8, 9, 10],
        'hogtransform__pixels_per_cell': [(7, 7), (8, 8), (9, 9)],
        'hogtransform__cells_per_block': [(2, 2), (3, 3)],
        'sgd__loss': ['hinge', 'squared_hinge', 'perceptron'],
        'sgd__learning_rate': ['optimal']
    },
    {
        'hogtransform__orientations': [7, 8, 9, 10],
        'hogtransform__pixels_per_cell': [(7, 7), (8, 8), (9, 9)],
        'hogtransform__cells_per_block': [(2, 2), (3, 3)],
        'sgd__loss': ['hinge', 'squared_hinge', 'perceptron'],
        'sgd__learning_rate': ['adaptive'],
        'sgd__eta0': [0.001, 0.01]
    }
]

grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

##**Deployment**
The application has been deployed to Render, making it accessible globally. You can use the live application to classify your own images by uploading them to the web interface.

Live link: Image Classifier Live  
https://imageclassifier-ca7c.onrender.com/
