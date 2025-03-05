import numpy as np
import pandas as pd
import flowkit as fk 
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

import gzip
import pickle

def load_percentage_of_dataset(percentage, X, y, seed=42):
    X_sample = X.sample(frac=percentage, random_state=seed).reset_index(drop=True)
    y_sample = y.sample(frac=percentage, random_state=seed).reset_index(drop=True) 
    return X_sample, y_sample

def load_n_samples(n_samples, X, y, seed=42):
    X_sample = X.sample(n_samples, random_state=seed).reset_index(drop=True)
    y_sample = y.sample(n_samples, random_state=seed).reset_index(drop=True) 
    return X_sample, y_sample

def pca(data, dim=50): 
    """reduce dimensionality of data to dim=50 (default) using PCA"""
    pca = PCA(n_components=dim)
    return pca.fit_transform(data)

def load_mouse_retina(): 
    with gzip.open("/Users/soli/Desktop/uni/thesis/code/data/macosko_2015.pkl.gz", "rb") as f:
        data = pickle.load(f)

    X = data["pca_50"]
    y = data["CellType1"] 

    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)

    # returns dataframe 
    return pd.DataFrame(X), pd.DataFrame(y_numeric) 

def load_iris_data(): 
    dataset = load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'

    iris = pd.DataFrame(
        dataset.data,
        columns=features)

    iris[target] = dataset.target

    # returns dataframe 
    return iris[features], iris[target]

def load_mnist(): 
    # use both test and train data 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # Flatten data and normalize values to [0, 1]
    X = X.reshape(X.shape[0], -1) / 255.0 
    
    # preprocess data using PCA 
    return pd.DataFrame(pca(X)), pd.DataFrame(y) 

def load_flow18(): 
    # specify path to data 
    scripts_dir = Path(__file__).parent
    file_path = scripts_dir.parent / "data" / "flow18_annotated.fcs"

    # pre-process data 
    sample = fk.Sample(file_path, sample_id='flow18', channel_labels=('Parameter_1', 'Parameter_10', 'Parameter_11', 'Parameter_12', 'Parameter_13', 'Parameter_14', 'Parameter_15', 'Parameter_16', 'Parameter_17', 'Parameter_18', 'Parameter_19', 'Parameter_2', 'Parameter_20', 'Parameter_21', 'Parameter_22', 'Parameter_23', 'Parameter_24', 'Parameter_3', 'Parameter_4', 'Parameter_5', 'Parameter_6', 'Parameter_7', 'Parameter_8', 'Parameter_9', 'SampleID', 'class'))
    df_events = sample.as_dataframe(source="raw")

    # only use selected columns (same as in the Belkina paper)
    selected_columns = [
        'Parameter_10', 'Parameter_11', 'Parameter_12', 
        'Parameter_13', 'Parameter_15', 'Parameter_18', 'Parameter_20', 
        'Parameter_21', 'Parameter_23', 'Parameter_8', 'Parameter_9', 'class'
    ]

    df_filtered = df_events[selected_columns]

    # Define class mapping for merging & renaming (in order to only display classes that are in plots in the paper)
    class_mapping = {
        1: "Lin-",
        2: "NK CD56highCD16-",
        3: "iNKT",
        4: "CD4+ non-Treg",
        5: "CD4+ Tregs",
        6: "CD8+ T CD56+CD16+",
        7: "Mono CD14+CD16-",
        9: "Mono CD14varCD16+",
        10: "CD8 T CD56-",
        11: "NK CD16+CD56lo", 
        14: "γδ T cells",
        15: "Dead or B cells",
        16: "Dead or B cells"
    }
    
    # Apply mapping
    df_filtered["class"] = df_filtered["class"].map(class_mapping)

    # Leave out any rows where the class is NaN (i.e., ignored classes)
    df_filtered = df_filtered.dropna(subset=[('class', '')])

    # Split into data (X) and labels (y)
    X = df_filtered.drop(columns=['class'])
    y = df_filtered['class']

    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)

    return pd.DataFrame(X), pd.DataFrame(y_numeric)