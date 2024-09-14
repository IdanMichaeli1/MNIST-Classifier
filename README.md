# MNIST Classifier with Optuna for Hyperparameter Tuning

This project implements a Multilayer Perceptron (MLP) to classify the MNIST dataset using PyTorch. The model's hyperparameters are tuned using **Optuna**, a powerful and flexible optimization framework. This repository includes the training pipeline, the neural network model, and tools for data preprocessing, model training, and evaluation.

## Features

- **MLP Architecture**: The model consists of fully connected layers with ReLU activations, dropout regularization, and batch normalization.
- **Custom Training Loop**: A modular and extendable training loop is provided for training the model with flexibility over multiple epochs.
- **Optuna for Hyperparameter Optimization**: Optuna is used to automate the process of finding the best hyperparameters such as learning rate, number of layers, hidden dimensions, and dropout rates.
- **Early Stopping**: The training loop supports early stopping to avoid overfitting and save computational resources.
- **GPU Support**: The training process can be run on both CPU and GPU for efficient computation.

## Files

- `MnistClassifier.ipynb`: The main Jupyter Notebook containing code for training the model and running the hyperparameter search.
- `utils.py`: Utility functions and classes, including the `MLP` model, training routines, and data processing utilities.

## Requirements

- Python 3.8+
- PyTorch
- Optuna

# Results

I added in the results folder the results I got with the current setup.
<br>
This includes the best hyperparameters, the evaluation results on the test set and the best model script.
