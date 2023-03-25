from multiprocessing import Process, Queue
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
from cnn import analyze_uploaded_audio

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

    q = Queue()
    p2 = Process(target=analyze_uploaded_audio, args=(q, savedFilePath))
    p2.start()
    p2.join()
    categorization, clustering, similarSounds = q.get()
    print("____________________" + str(categorization) + "____________________" + str(clustering))

    resp = {"success": True, "response": savedFilePath, "spectrogram": spectrogramFilePath, "name": f.filename, "categorization": categorization, "clustering": clustering, "similarSounds": similarSounds }
    return jsonify(resp), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)