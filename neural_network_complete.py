# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define path to spectrogram images
spectrogram_path = "path/to/spectrogram/images"

# Define function to load and preprocess images
def load_images():
    images = []
    for file in os.listdir(spectrogram_path):
        img = cv2.imread(os.path.join(spectrogram_path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128)) # Resize image to (128, 128)
        images.append(img)
    return np.array(images)

# Load and preprocess images
images = load_images()

# Define number of clusters
num_clusters = 5

# Create CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_clusters, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape images for input into CNN
images = images.reshape((-1, 128, 128, 1))

# Fit model on images
model.fit(images, epochs=10, batch_size=32)

# Extract features from CNN
feature_extractor = Sequential(model.layers[:-1])
features = feature_extractor.predict(images)

# Cluster extracted features using k-means algorithm
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Print cluster labels for each image
for i, label in enumerate(kmeans.labels_):
    print(f"Image {i} is in cluster {label}")
