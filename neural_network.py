import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import umap
import matplotlib.pyplot as plt

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
    tf.keras.layers.Dense(32, activation='relu', name='cluster_layer'),
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
        # X = Input Data
        # Y = Label
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

def predict_slices(model, img_paths, layer = None):
    """
    Predict the class of each image given a list of image paths.

    Args:
    model (tf.keras.Model): A trained TensorFlow model for image classification.
    img_paths (List[str]): A list of image file paths.

    Returns:
    List[np.ndarray]: A list of predictions corresponding to the input images.
    """
    predictions = []

    if (layer is not None):
        cnn_shortened = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("cluster_layer").output
        )

    # Iterate over the image paths and predict the class for each image
    for img_path in img_paths:
        pred = predict(model, img_path)
        predictions.append(pred)
        
    return get_average_predictions(predictions)

def predict(model, img_path):
    """
    Predict the class of a single image given its file path.

    Args:
    model (tf.keras.Model): A trained TensorFlow model for image classification.
    img_path (str): The image file path.

    Returns:
    np.ndarray: The prediction for the input image.
    """
    # Preprocess image
    pil_img = tf.keras.preprocessing.image.load_img(
    img_path, grayscale=False, color_mode='rgb', target_size=[256,256],
    interpolation='nearest'
    )

    # Convert to numpy Array
    img_tensor = np.array(pil_img)
    # Add a batch dimension to the image tensor
    img_tensor = img_tensor[None, :, :]

    # Make the prediction using the model
    pred = model.predict(img_tensor)
    return pred

def get_average_predictions(predictions):
    """
    Calculate the average prediction values for each class.

    Args:
    predictions (List[np.ndarray]): A list of predictions.

    Returns:
    Tuple[float, float, float, float]: A tuple containing the average prediction values for each class.
    """
    # Convert the list of predictions to a NumPy array
    predictions_array = np.array(predictions)

    # Calculate the mean along the first axis (rows)
    mean_predictions = np.mean(predictions_array, axis=0)

    return mean_predictions

def cluster_data_kmeans(features, train_ds): 
    number_classes = len(train_ds.class_names)

    # Initialize the KMeans algorithm with the number of clusters you want to form
    kmeans = KMeans(n_clusters = number_classes)

    # Fit the KMeans algorithm to the features
    kmeans.fit(features)

    # Predict the cluster assignments for each feature
    cluster_assignments = kmeans.predict(features)

    # Print the cluster assignments for each feature
    #for i, assignment in enumerate(cluster_assignments):
    #    print("Feature {}: assigned to Cluster {}".format(i, assignment))

    #create_scatter_plot_2d(features, cluster_assignments, True)

    return kmeans

def cluster_data_agglomerative(features, train_ds):    
    number_classes = len(train_ds.class_names)

    # Initialize the Agglomerative Clustering algorithm with the number of clusters you want to form
    agglomerative = AgglomerativeClustering(n_clusters=number_classes)

    # Fit the Agglomerative Clustering algorithm to the features
    agglomerative.fit(features)

    # Predict the cluster assignments for each feature
    cluster_assignments = agglomerative.labels_

    # Print the cluster assignments for each feature
    # for i, assignment in enumerate(cluster_assignments):
    #     print("Feature {}: assigned to Cluster {}".format(i, assignment))

    # create_scatter_plot_2d(features, cluster_assignments, True)

    return agglomerative



def cluster_data_dbscan(features, train_ds, eps=0.5, min_samples=100):   
    # Initialize the DBSCAN algorithm with the desired parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Fit the DBSCAN algorithm to the features
    dbscan.fit(features)
    
    # Predict the cluster assignments for each feature
    cluster_assignments = dbscan.labels_
    
    # Print the cluster assignments for each feature
    # for i, assignment in enumerate(cluster_assignments):
    #     print("Feature {}: assigned to Cluster {}".format(i, assignment))
    
    create_scatter_plot_2d(features, cluster_assignments, True)

    return dbscan
    

def create_scatter_plot_3d(features, cluster_assignments, use_dimension_reduction):
    if (use_dimension_reduction):
        # create a UMAP model with 2 output dimensions
        umap_model = umap.UMAP(n_components=3)

        # fit the model to your feature vectors
        features = umap_model.fit_transform(features)

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

def create_scatter_plot_2d(features, cluster_assignments, use_dimension_reduction):
    if (use_dimension_reduction):
        # create a UMAP model with 2 output dimensions
        umap_model = umap.UMAP(n_components=2)

        # fit the model to your feature vectors
        features = umap_model.fit_transform(features)

    # Create a 2D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, assignment in enumerate(cluster_assignments):
        if assignment == 0:
            color = 'red'
        elif assignment == 1:
            color = 'blue'
        else:
            color = 'green'

        x = features[i, 0]
        y = features[i, 1]
        ax.scatter(x, y, c=color)

    # Show the plot
    plt.show()
    # Save the plot as an image file
    plt.savefig('2d_scatter_plot.png')

def test_clustering_accuracy_kmeans(kmeans, cnn_shortened, test_ds):
    accuracies = []
    for batch_num, (X, Y) in enumerate(test_ds):
        # X = Input Data
        # Y = Label
        batch_size = len(Y)
        features = cnn_shortened.predict(X)
        predicted_labels = kmeans.predict(features)
        accuracy = test_clustering_accuracy(Y, predicted_labels)
        accuracies.append(accuracy)

    print(f'Accuracy Keans: {(sum(accuracies) / len(accuracies))*100}%')

def test_clustering_accuracy_agglomerative(agglomerative, cnn_shortened, test_ds):
    accuracies = []
    for batch_num, (X, Y) in enumerate(test_ds):
        # X = Input Data
        # Y = Label
        batch_size = len(Y)
        features = cnn_shortened.predict(X)
        predicted_labels = agglomerative.fit_predict(features)
        accuracy = test_clustering_accuracy(Y, predicted_labels)
        accuracies.append(accuracy)

    print(f'Accuracy Agglomerative: {(sum(accuracies) / len(accuracies))*100}%')

def test_clustering_accuracy_dbscan(dbscan, cnn_shortened, test_ds):
    accuracies = []
    for batch_num, (X, Y) in enumerate(test_ds):
        # X = Input Data
        # Y = Label
        batch_size = len(Y)
        features = cnn_shortened.predict(X)
        predicted_labels = dbscan.fit_predict(features)
        accuracy = test_clustering_accuracy(Y, predicted_labels)
        accuracies.append(accuracy)

    print(f'Accuracy DBSCAN: {(sum(accuracies) / len(accuracies))*100}%')


def test_clustering_accuracy(test_labels, predicted_labels):    
    from sklearn.metrics import precision_score, recall_score, f1_score

    #test_data = test_data.map(lambda x, y: x)
    # Predict cluster assignments for the test dataset
    #predicted_labels = model.predict(test_data)
    
    # Calculate external evaluation metrics for each label
    precision = precision_score(test_labels, predicted_labels, average=None)
    recall = recall_score(test_labels, predicted_labels, average=None)
    f1 = f1_score(test_labels, predicted_labels, average=None)
    
    # Calculate the average F1-score across all labels
    accuracy = f1.mean()
    
    print(f'Accuracy {(accuracy)*100}%')
    return accuracy

def create_model_cnn(epochs = 10, learning_rate = 0.025, cnn = None):
    print("Loading Data...")
    train_ds, test_ds = load_data()
    print("Done!")

    if (cnn is None):
        print("Creating Model...")
        cnn = create_model(learning_rate = learning_rate, num_classes=len(train_ds.class_names), pictureSize=(256,256))
        print("Done!")

        print("Training Model...")
        train_model(model = cnn, epochs = epochs, train_ds = train_ds)
        print("Done!")

        print("Testing Model...")
        test_model(model = cnn, test_ds = test_ds, detailed_print = True)
        print("Done!")

    print("Testing Clustering Kmeans...")
    
    cnn_shortened = tf.keras.models.Model(
        inputs=cnn.input,
        outputs=cnn.get_layer("cluster_layer").output
    )

    # Get the Predictoin out of the shortened CNN
    features = cnn_shortened.predict(train_ds)

    print("Clustering Data Kmeans...")    
    kmeans = cluster_data_kmeans(features, train_ds)
    print("Done!")

    print("Testing Clustering KMeans...")
    #test_clustering_accuracy_kmeans(kmeans, cnn_shortened, test_ds)
    print("Done!")

    print("Clustering Data Agglomerative...")    
    agglomerative = cluster_data_agglomerative(features, train_ds)
    print("Done!")
    
    print("Testing Clustering Agglomerative...")
    #test_clustering_accuracy_agglomerative(agglomerative, cnn_shortened, test_ds)
    print("Done!")
    
    print("Clustering Data DBSCAN...")    
    dbscan = cluster_data_dbscan(features, train_ds)
    print("Done!")

    print("Testing Clustering DBSCAN...")
    test_clustering_accuracy_dbscan(dbscan, cnn_shortened, test_ds)
    print("Done!")

    return cnn