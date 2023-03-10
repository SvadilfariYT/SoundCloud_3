from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)

@app.route('/', methods= ['GET', 'POST'])
def get_message():
    # if request.method == "GET":
    print("Got request in main function")
    return render_template("index.html")

@app.route('/upload_static_file', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    print(request.files)
    f = request.files['static_file']
    f.save(os.path.join("frontend/" + "uploadedData/" + f.filename))

    resp = {"success": True, "response": "file saved!"}
    return jsonify(resp), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)