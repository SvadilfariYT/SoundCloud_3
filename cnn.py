from neural_network import create_model_cnn
from neural_network import predict
from neural_network import predict_slices
from spectrogram_creation import slice_audio
from spectrogram_creation import load_audio
from spectrogram_creation import get_spectrogram
from spectrogram_creation import create_trainData_csv
from spectrogram_creation import convert_all_files_in_directory_to_16bit

import sys
import os
import joblib
import matplotlib.pyplot as plt

def save_model_cnn(model, model_path):
    # Save the model to disk
    sys.stdout = open('NUL', 'w')
    joblib.dump(model, model_path)
    sys.stdout = sys.__stdout__

def load_model_cnn(model_path):
    # Load the model from disk
    sys.stdout = open('NUL', 'w')
    model = joblib.load(model_path)
    sys.stdout = sys.__stdout__

    return model

def predict_by_wav(model, audio_path : str):    
    filepaths = slice_audio (audio_path, "", 3000, 1500)

    spectrogram_paths = []
    for i, filepath in enumerate(filepaths):
        waveform, sample_rate = load_audio(audio_path)
        spectrogram, spect = get_spectrogram(waveform)

        file_name, file_extension = os.path.splitext(filepath)

        output_path = file_name + "_" + str(i) + ".png"

        plt.imsave(output_path, spect.numpy(), cmap='gray')
        spectrogram_paths.append(output_path)

    predictions_categorization = predict_slices(model, spectrogram_paths)
    predictions_clustering = predict_slices(model, spectrogram_paths, "cluster_layer")

    return predictions_categorization, predictions_clustering

# MAIN METHOD
if __name__ == '__main__':
    
    create_spectrogramm = False
    force_create_model = False

    model_path = 'model.joblib'

    cnn = load_model_cnn(model_path)

    if (create_spectrogramm):
        create_trainData_csv("./data/AudioDataScience_Data_Assignment.csv")
    if (cnn == None or force_create_model):
        # Create Model and save it
        cnn = create_model_cnn(epochs=20)
        save_model_cnn(cnn, model_path) 

    predictions_categorization, predictions_clustering = predict_by_wav(cnn, './data/AudioSamples_16Bit/Car_003.WAV')



