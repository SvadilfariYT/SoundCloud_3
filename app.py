from multiprocessing import Process
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import threading

import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

from matplotlib import pyplot as plt

from spectrogram_creation import load_audio 
from createSpectrogramForWebsite import createSpectrogramForWebsite

app = Flask(__name__)

CORS(app)

@app.route('/', methods= ['GET', 'POST'])
def get_message():
    # if request.method == "GET":
    print("Got request in main function")
    return render_template("index.html")

@app.route('/analyze', methods= ['GET', 'POST'])
def analyze():
    # if request.method == "GET":
    return render_template("analyze.html")

@app.route('/upload_static_file', methods=['POST'])
def upload_static_fileo():
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']

    savedFilePath = os.path.join("static" + "/uploadedData/" + f.filename)
    f.save(savedFilePath)

    spectrogramFilePath = os.path.join("static" + "/spectrograms/" + "spect" + ".png")
    p = Process(target=createSpectrogramForWebsite, args=(savedFilePath, spectrogramFilePath))
    p.start()


    p.join()
    resp = {"success": True, "response": savedFilePath, "spectrogram": spectrogramFilePath, "name": f.filename}
    # resp = {"success": True, "response": savedFilePath}
    return jsonify(resp), 200


# def generate_spectrogram(audioFilePath, spectrogramFilePath):
#     # Erstelle das Spektrogramm
#     y, sr = librosa.load(audioFilePath)
#     spec = librosa.feature.melspectrogram(y=y, sr=sr)
#     spec_db = librosa.power_to_db(spec, ref=np.max)
#     plt.figure(figsize=(10, 5))
#     librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
    
#     # Speichern Sie das Spektrogramm als PNG-Datei
#     plt.savefig(spectrogramFilePath)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)

    #  t = threading.Thread(target=createSpectrogramForWebsite, args=(savedFilePath, spectrogramFilePath))
    # t.start()