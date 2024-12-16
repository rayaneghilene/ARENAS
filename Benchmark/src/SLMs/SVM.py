#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import joblib

def main(args):
    # Load the dataset
    df = pd.read_csv(args.data)
    
    # Display NaN values in the dataset
    nan_text = df['text'].isna().any()
    nan_class = df['class'].isna().any()
    print(f"NaN values in 'text' column: {nan_text}")
    print(f"NaN values in 'class' column: {nan_class}")
    
    # Preprocessing
    max_features = 10000
    sequence_length = 250

    # Text vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(df['text'])
    
    # Map class labels to integers
    label_mapping = {subtype: label for label, subtype in enumerate(df['class'].unique())}
    num_classes = len(label_mapping)

    # Create TensorFlow datasets
    batch_size = 16
    def vectorize_text(text, label):
        return vectorize_layer(text), label
    df['label'] = df['class'].map(label_mapping)
    train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=42)
    raw_train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'], train_df['label']))
    train_ds = raw_train_ds.map(vectorize_text)
    train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
    raw_val_ds = tf.data.Dataset.from_tensor_slices((val_df['text'], val_df['label']))
    val_ds = raw_val_ds.map(vectorize_text)
    val_ds = val_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    raw_test_ds = tf.data.Dataset.from_tensor_slices((test_df['text'], test_df['label']))
    test_ds = raw_test_ds.map(vectorize_text)
    test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Model training
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Vectorize the text data
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(train_df['text'])
    X_test_counts = count_vectorizer.transform(test_df['text'])

    # Support Vector Machine classifier
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train_counts, train_df['label'])
    svm_score = svm_classifier.score(X_test_counts, test_df['label'])
    print(f'Test accuracy: {svm_score}')

    # Generate predictions
    predictions = svm_classifier.predict(X_test_counts)

    # Generate a classification report
    classification_rep = classification_report(test_df['label'], predictions, target_names=label_mapping, output_dict=True)
    print(classification_report(test_df['label'], predictions, target_names=label_mapping))

    # Save classification report to a text file
    output_file_path = 'output.txt'
    with open(output_file_path, 'w') as file:
        file.write(classification_report(test_df['label'], predictions, target_names=label_mapping))

    print(f"Classification report saved to: {output_file_path}")

    # Save the model
    joblib.dump(svm_classifier, 'model_name.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Classifier for Text Data")
    parser.add_argument('--data', type=str, help='Path to the dataset')
    args = parser.parse_args()
    main(args)
