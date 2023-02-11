from neural_network import create_model_cnn
from neural_network import predict
from spectrogram_creation import load_audio
from spectrogram_creation import get_spectrogram

import sys
import joblib
import matplotlib.pyplot as plt

create = False

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

def predict_by_wav(audio_path : str):
    img_path = './data/myvoice/no_myspec_img.png'
    waveform, sample_rate = load_audio(audio_path)
    spectrogram, spect = get_spectrogram(waveform)

    plt.imsave(img_path, spectrogram.numpy(), cmap='gray')

    prediction = predict(model, img_path)

    print(prediction)

# MAIN METHOD
if __name__ == '__main__':
    
    model_path = 'model.joblib'

    if (create):
        # Create Model and save it
        model = create_model_cnn()
        save_model_cnn(model, model_path)
    else:
        # Load Model
        model = load_model_cnn(model_path)


    predict_by_wav('./data/no/0a2b400e_nohash_0.wav')


