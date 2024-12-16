import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 42


def load_data(data_path, mode, train_percentage = 0.8, validation_percentage = 0.1, test_percentage = 0.1):
    if mode == 'binary':
        return _split_binary_data(data_path, train_percentage, validation_percentage, test_percentage)
    
    else:
        return _split_data(data_path, train_percentage, validation_percentage, test_percentage)


def _split_data(data_path, train_percentage = 0.8, validation_percentage = 0.1, test_percentage = 0.1):
    if train_percentage + validation_percentage + test_percentage != 1:
        raise ValueError("Sum of percentages must be 1")
    
    df = pd.read_csv(data_path)

    X = df.text.values
    y = pd.get_dummies(df["class"])
    labels = list(y.keys())
    y = y.iloc[:,:].values

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size = 1 - train_percentage, train_size = train_percentage, random_state = SEED, shuffle = True, stratify = y)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size = test_percentage / (validation_percentage + test_percentage), train_size = validation_percentage / (validation_percentage + test_percentage), random_state = SEED, shuffle = True, stratify = y_rest)

    return (X_train, X_validation, X_test, y_train, y_validation, y_test, labels)     


def _split_binary_data(data_path, train_percentage = 0.8, validation_percentage = 0.1, test_percentage = 0.1):
    if train_percentage + validation_percentage + test_percentage != 1:
        raise ValueError("Sum of percentages must be 1")
    
    df = pd.read_csv(data_path)

    X = df.text.values
    y = []

    for _, line in df.iterrows():
        if line["class"] == "neither":
            y.append(0)
        else:
            y.append(1)

    y = np.array(y)

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size = 1 - train_percentage, train_size = train_percentage, random_state = SEED, shuffle = True, stratify = y)
    X_validation, X_test, y_validation, y_test = train_test_split(X_rest, y_rest, test_size = test_percentage / (validation_percentage + test_percentage), train_size = validation_percentage / (validation_percentage + test_percentage), random_state = SEED, shuffle = True, stratify = y_rest)

    return (X_train, X_validation, X_test, y_train, y_validation, y_test, None)  
