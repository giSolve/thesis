import numpy as np
import pandas as pd
from keras.datasets import mnist

def load_percentage_of_dataset(percentage, X, y, seed=42):
    X_sample = X.sample(frac=percentage, random_state=seed).reset_index(drop=True)
    y_sample = y.sample(frac=percentage, random_state=seed).reset_index(drop=True) 
    return X_sample, y_sample


def load_mnist(): 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test]).reshape(-1, 28*28) / 255.0  # Flatten images
    y = np.concatenate([y_train, y_test])
    return X, y