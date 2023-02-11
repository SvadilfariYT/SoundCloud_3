import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
from sklearn.cluster import KMeans

import joblib

class PredictedObj:
    def __init__(self, name, feature1, feature2, feature3):
        self.name = name
        self.x = feature1
        self.y = feature2
        self.z = feature3

def load_data():
    train_directory = './data/train/'
    test_directory = './data/test/'

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory, labels='inferred', label_mode='int', image_size=(256, 256), seed=123, 
        validation_split=0.2, subset='validation')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory, labels='inferred', label_mode='int', image_size=(256, 256), 
        validation_split=None, subset=None)

    class_names = train_ds.class_names
    print("Loaded classes: " + str(class_names))

    return train_ds, test_ds

# CREATE MODEL
def create_model(learning_rate : float, num_classes : int, pictureSize):
    img_height, img_width = pictureSize

    model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    metrics = ['accuracy']
    model.compile(optimizer, loss_fn, metrics)

    return model

# TRAIN MODEL
def train_model(model, epochs : int, train_ds):
    print('\nFitting:')

    # Train the model.
    history = model.fit(train_ds, epochs=epochs)

# TEST MODEL
def test_model(model, test_ds, detailed_print):
    correct = 0
    batch_size = 0

    for batch_num, (X, Y) in enumerate(test_ds):
        batch_size = len(Y)
        pred = model.predict(X)
        
        # print features
        features = pred[0]
        for i, feature in enumerate(features):
            print("Feature {}: {}".format(i, feature))

        # print Accuracy
        for i in range(batch_size):
            predicted = np.argmax(pred[i], axis=-1)
            actual = Y[i]

            if detailed_print:
                print(f'predicted {predicted}, actual {actual}')

            if predicted == actual:
                correct += 1
        break

    print(f'Number correct: {correct} out of {batch_size}')
    print(f'Accuracy {(correct / batch_size)*100}%')

    return pred

def predict(model, img_path):
    pil_img = tf.keras.preprocessing.image.load_img(
    img_path, grayscale=False, color_mode='rgb', target_size=[256,256],
    interpolation='nearest'
    )

    img_tensor = np.array(pil_img)
    pred = model.predict(img_tensor[None,:,:])
    return pred

def cluster_data(test_ds, features):
    # Convert the features to a NumPy array
    features = np.array(features)

    # Initialize the KMeans algorithm with the number of clusters you want to form
    kmeans = KMeans(n_clusters=3)

    # Fit the KMeans algorithm to the features
    kmeans.fit(features)

    # Predict the cluster assignments for each feature
    cluster_assignments = kmeans.predict(features)

    # Print the cluster assignments for each feature
    #for i, assignment in enumerate(cluster_assignments):
    #    print("Feature {}: assigned to Cluster {}".format(i, assignment))

    create_scatter_plot(features, cluster_assignments)
    

def create_scatter_plot(features, cluster_assignments):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, assignment in enumerate(cluster_assignments):
        if assignment == 0:
            color = 'red'
        elif assignment == 1:
            color = 'blue'
        else:
            color = 'green'

        x = features[i, 0]
        y = features[i, 1]
        z = features[i, 2]
        ax.scatter(x, y, z, c=color)

    # Show the plot
    plt.show()
    # Save the plot as an image file
    plt.savefig('3d_scatter_plot.png')

def create_model_cnn():
    print("Loading Data...")
    train_ds, test_ds = load_data()
    print("Done!")

    print("Creating Model...")
    model = create_model(learning_rate = 0.025, num_classes=len(train_ds.class_names), pictureSize=(256,256))
    print("Done!")

    print("Training Model...")
    train_model(model = model, epochs = 20, train_ds = train_ds)
    print("Done!")

    print("Testing Model...")
    test_model(model, test_ds, False)
    print("Done!")

    return model

