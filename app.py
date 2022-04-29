from flask import Flask, jsonify, request
import numpy as np

# Model librabries
from models.breathing.lib.BreathingDetector import BreathingDetector

app = Flask(__name__)

# Keep the model initialization only at the startup
breathing_detector = BreathingDetector(
    sampling_rate=250,
    stride=250,
    threshold=0.05,
)

@app.route("/breathing", methods=["POST"])
def home():
    request_data = request.json
    provided_breathing_signal = np.array(request_data["readings"])
    peaks = breathing_detector.find_peaks(provided_breathing_signal)
    response = {
        "bpm": len(peaks[0].tolist())
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)