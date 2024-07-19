# Image Classifier Application

This repository contains the code for an image classifier machine learning application deployed on Flask, a popular Python web framework. The application allows users to upload images and receive predictions about the contents of those images based on a pre-trained machine learning model.

## Application Overview

The backend of the application utilizes Flask to handle incoming HTTP requests, process uploaded images, and interface with the machine learning model. This setup enables a seamless interaction where users can easily upload images and get instant predictions on what the images contain.

## Machine Learning Model

The core of this application is a convolutional neural network (CNN) designed for image classification. The model has been trained on the CIFAR-10 dataset, which includes images across 10 different categories such as airplanes, dogs, cars, and more.

### Model Pipeline and Serialization

To streamline the process of using the machine learning model within the Flask application, the model has been saved in a pickle format. This serialization facilitates easy loading and inference within the application's environment. Additionally, the model has been integrated into a pipeline, ensuring that image data is appropriately pre-processed before making predictions.

### Hyperparameter Tuning

The model's performance was optimized using grid search for hyperparameter tuning. This process involved systematically varying model parameters to find the combination that results in the best prediction accuracy.

## Deployment

The application has been deployed to Render, making it accessible via the following live link: [https://imageclassifier-ca7c.onrender.com/](https://imageclassifier-ca7c.onrender.com/). This deployment allows users from anywhere to access the application and use it to classify their own images.

## Built With

- **Flask Framework:** Used for handling HTTP requests, serving the web application, and interfacing with the machine learning model.
- **TensorFlow and Keras:** For building and training the convolutional neural network model.
- **Pickle:** For serializing the model and integrating it into the Flask application.
- **Render:** For hosting and deploying the web application.

