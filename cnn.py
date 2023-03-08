from neural_network import create_model_cnn
from neural_network import predict
from neural_network import cluster_data
from neural_network import create_scatter_plot
from spectrogram_creation import load_audio
from spectrogram_creation import get_spectrogram
import numpy as np

import sys
import os
import joblib
import pickle
import matplotlib.pyplot as plt
import random

def save_model_cnn(model, model_path):
    # Save the model to disk
    sys.stdout = open('NUL', 'w')
    joblib.dump(model, model_path)
    sys.stdout = sys.__stdout__

def load_model_cnn(model_path):
    print("Loading Model CNN...")
    # Load the model from disk
    sys.stdout = open('NUL', 'w')
    model = joblib.load(model_path)
    sys.stdout = sys.__stdout__
    print("Done!")

    return model

def create_and_save_predictions(filename:str):
    features_no = predict_all_in_directory('./data/no', './data/no/myvoice', 'no', 'dense_layer')
    features_yes = predict_all_in_directory('./data/yes', './data/yes/myvoice', 'yes', 'dense_layer')
    features = {**features_no, **features_yes}
    #print(features["0a9f9af7_nohash_0.wav"])

    # Save the dictionary to the file
    with open(filename, 'wb') as f:
        pickle.dump(features, f)
    return features

def load_predictions(filename:str):
    print("Loading Predictions...")
    # Load the dictionary from the file
    with open(filename, 'rb') as f:
        features = pickle.load(f)
    print("Done!")
    return features

def predict_by_wav(audio_path:str, img_path:str, output_layer=None):
    waveform, sample_rate = load_audio(audio_path)
    spectrogram, spect = get_spectrogram(waveform)

    plt.imsave(img_path, spectrogram.numpy(), cmap='gray')

    prediction = predict(model, img_path, output_layer)

    return prediction

def predict_all_in_directory(input_directory, output_directory, prefix:str, output_layer=None):
    # Make sure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    features = {}

    count = 0
    max_files = 100
    
    # Loop over all files in the input directory
    for filename in os.listdir(input_directory):
        # Ignore any files that aren't wav files
        if not filename.endswith('.wav'):
            continue
        
        # Construct the input and output file paths
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f'{prefix}_{os.path.splitext(filename)[0]}.png')
        
        # Call the predict_by_wav function
        prediction = predict_by_wav(input_path, output_path, output_layer)
        features[prefix + "_" + filename] = prediction
        
        count += 1
        if max_files is not None and count >= max_files:
            break
        
    return features

# MAIN METHOD
if __name__ == '__main__':

    createCNN = True
    doPrediction = True
    doClustering = True
    
    model_path = 'model.joblib'

    # Step 0: PreProcess Data

    # Step 1: Create/Load the Convolutional Neural Network
    if (createCNN):
        # Create Model and save it
        model = create_model_cnn()
        save_model_cnn(model, model_path)
    else:
        # Load Model
        model = load_model_cnn(model_path)

    return

    # Step 2: Extract Features out of CNN for Clustering
    if (doPrediction):
        # Create Predictions and save them
        features = create_and_save_predictions("features.pickle")
    else:
        # Load Predictions
        features = load_predictions("features.pickle")
    # Get a random key from the dictionary
    random_key = random.choice(list(features.keys()))
    print(features[random_key])

    # Dictionary to List of values
    features = list(features.values())
    # Convert the features to a NumPy array
    features = np.array(features)
    # Reshape the array to have 2 dimensions
    features = features.reshape(features.shape[0], -1)

    # Step 3: Cluster Data
    if (doClustering):
        # Cluster Data
        cluster_data(features, 2)
        print("")
    else:
        # Load Model
        print("")



