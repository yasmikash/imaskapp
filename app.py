import flask
from flask import jsonify
import pandas as pd

# Model librabries
from models.breathing.lib.BreathingDetector import BreathingDetector

app = flask.Flask(__name__)

# Keep the model initialization only at the startup
breathing_detector = BreathingDetector(
    sampling_rate=250,
    stride=250,
    threshold=0.05,
)

@app.route("/breathing", methods=["POST"])
def home():
    provided_breathing_signal = pd.read_csv("models/breathing/data/fast.csv")["read"].values
    peaks, probs = breathing_detector.find_peaks(provided_breathing_signal[:500])
    return jsonify(peaks.tolist())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)