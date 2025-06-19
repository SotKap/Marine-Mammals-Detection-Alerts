import time
import threading
import logging
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import pyaudio
from scipy.signal import resample
import tflite_runtime.interpreter as tflite

# === CONFIGURATION ===
FIREBASE_CRED_FILE = "/home/pi/Desktop/marine-mammals-detection-firebase-adminsdk-i5yak-7fdd87a1e1.json"
FIREBASE_DB_URL = "https://marine-mammals-detection-default-rtdb.firebaseio.com/"

MODEL_PATH = "/home/pi/Desktop/model.tflite"
LABELS_PATH = "/home/pi/Desktop/labels.txt"  # optional label file
CLASS_LABELS = ["Background Noise", "Common Dolphin", "Humpback Whale", "Sperm Whale"]

MIC_SAMPLE_RATE = 384000     # mic sample rate
MODEL_SAMPLE_RATE = 44100    # model expects 44.1kHz
MODEL_NUM_SAMPLES = 44032    # model input shape
CHUNK_DURATION = 1           # seconds
DEVICE_INDEX = 1             # USB mic index
CONFIDENCE_THRESHOLD = 0.90  # only send alerts above this

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# === FIREBASE INIT ===
cred = credentials.Certificate(FIREBASE_CRED_FILE)
firebase_admin.initialize_app(cred, {
    "databaseURL": FIREBASE_DB_URL
})
alerts_ref = db.reference("alerts")
latest_alert_ref = db.reference("latest-alert")
status_ref = db.reference("status")  # New reference for status path

# === ALERT FUNCTION ===
def send_alert_to_firebase(alert):
    try:
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, list):
                return [convert(x) for x in obj]
            else:
                return obj

        alert = convert(alert)
        alerts_ref.push(alert)
        latest_alert_ref.set(alert)

        # Update last-mammal-timestamp in /status
        last_mammal_timestamp = alert["timestamp"]
        status_ref.child("last-mammal-timestamp").set(last_mammal_timestamp)

        logger.info("Alert and status timestamp sent to Firebase.")
    except Exception as e:
        logger.error(f"Firebase error: {e}")

# === MODEL LOADING ===
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# === AUDIO CLASSIFICATION ===
def classify_audio(interpreter, input_details, output_details, audio_data):
    input_data = np.expand_dims(audio_data, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# === MONITORING LOOP ===
def continuous_monitoring():
    audio = pyaudio.PyAudio()
    interpreter, input_details, output_details = load_model(MODEL_PATH)

    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=MIC_SAMPLE_RATE,
                        input=True,
                        input_device_index=DEVICE_INDEX,
                        frames_per_buffer=int(MIC_SAMPLE_RATE * CHUNK_DURATION))

    try:
        while True:
            logger.info("Capturing 1 second of audio...")
            raw_data = stream.read(int(MIC_SAMPLE_RATE * CHUNK_DURATION), exception_on_overflow=False)
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

            # Resample
            logger.info("Resampling...")
            audio_resampled = resample(audio_np, MODEL_NUM_SAMPLES)
            audio_normalized = audio_resampled / 32768.0

            # Run inference
            logger.info("Running model inference...")
            result = classify_audio(interpreter, input_details, output_details, audio_normalized)
            pred_idx = np.argmax(result)
            confidence = float(result[pred_idx])
            label = CLASS_LABELS[pred_idx] if pred_idx < len(CLASS_LABELS) else f"Class {pred_idx}"

            logger.info(f"Prediction: {label} (confidence: {confidence:.3f})")

            if label != "Background Noise" and confidence >= CONFIDENCE_THRESHOLD:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                alert = {
                    "timestamp": current_time,
                    "last_mammal_timestamp": current_time,
                    "predicted_class": label,
                    "confidence": round(confidence, 3),
                    "location": {
                        "latitude": 36.2517,
                        "longitude": 22.5923,
                    },
                    "device_id": "Device_001",
                    "message": f"Detected {label} with confidence {confidence:.3f}",
                    "threshold_exceeded": True
                }
                send_alert_to_firebase(alert)

            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Monitoring error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# === START MONITORING THREAD ===
if __name__ == "__main__":
    t = threading.Thread(target=continuous_monitoring, daemon=True)
    t.start()

    while True:
        time.sleep(10)
