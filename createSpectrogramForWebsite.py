import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def createSpectrogramForWebsite(audio_file, spectrogramFilePath):
    y, sr = librosa.load(audio_file)
    
    D = np.abs(librosa.stft(y))
    power = librosa.power_to_db(D**2)

    plt.figure(figsize=(10, 5), frameon=False)
    librosa.display.specshow(power, sr=sr, x_axis='linear', y_axis='time')
    plt.axis('off')
    plt.savefig(spectrogramFilePath, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    createSpectrogramForWebsite("static/uploadedData/Bus_004.wav", "static/spectrograms/Bus_004.png")