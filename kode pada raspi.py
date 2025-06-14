from flask import Flask, Response, send_from_directory, jsonify, request, redirect, url_for
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import RPi.GPIO as GPIO
import BlynkLib
import threading
import os

# === BLYNK SETUP ===
BLYNK_AUTH_TOKEN = 'JMSJoJYzqTPfOYA6P3ywfot82MfIySRt'
blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)

# Virtual Pins
VPIN_LED = 1
VPIN_STATUS = 2
VPIN_TERMINAL = 3
VPIN_FAULT = 4

# === GPIO SETUP ===
GPIO.setmode(GPIO.BCM)
RED_LED = 17
BUZZER = 25
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.output(RED_LED, GPIO.LOW)
GPIO.output(BUZZER, GPIO.LOW)

# === YOLO SETUP ===
model = YOLO("/home/klp6/Downloads/best(1).pt")
TARGET_LABEL = "drowsy"

# === SNAPSHOT SETUP ===
LOG_IMAGE_DIR = "snapshots"
os.makedirs(LOG_IMAGE_DIR, exist_ok=True)

# === FLASK SETUP ===
app = Flask(__name__)
drowsy_count = 0
camera = cv2.VideoCapture(0)

# State untuk deteksi kantuk 3 detik
drowsy_start_time = None
saved_drowsy_snapshot = False

def draw_boxes(frame, results):
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        conf = box.conf[0].cpu().numpy()
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        
        if label == TARGET_LABEL:  # "drowsy"
            color = (0, 0, 255)  # merah
        else:
            color = (0, 255, 0)  # hijau
        
        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Tampilkan label + confidence 2 decimal
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def detect_drowsiness(frame):
    global drowsy_count, drowsy_start_time, saved_drowsy_snapshot

    results = model(frame, verbose=False)[0]
    detected = False

    # Cek ada kantuk
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label == TARGET_LABEL:
            detected = True
            break

    if detected:
        GPIO.output(RED_LED, GPIO.HIGH)
        GPIO.output(BUZZER, GPIO.HIGH)
        blynk.virtual_write(VPIN_LED, 255)
        blynk.virtual_write(VPIN_TERMINAL, f"Kantuk terdeteksi pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        blynk.virtual_write(VPIN_STATUS, "Kantuk!")

        now = time.time()
        if drowsy_start_time is None:
            drowsy_start_time = now
            saved_drowsy_snapshot = False

        if (now - drowsy_start_time >= 3) and (not saved_drowsy_snapshot):
            drowsy_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(LOG_IMAGE_DIR, filename)
            frame_with_boxes = draw_boxes(frame.copy(), results)
            cv2.imwrite(filepath, frame_with_boxes)
            with open("drowsiness_log.txt", "a") as f:
                f.write(f"{timestamp} | {filename}\n")
            saved_drowsy_snapshot = True
    else:
        GPIO.output(RED_LED, GPIO.LOW)
        GPIO.output(BUZZER, GPIO.LOW)
        blynk.virtual_write(VPIN_LED, 0)
        blynk.virtual_write(VPIN_STATUS, "Normal")
        drowsy_start_time = None
        saved_drowsy_snapshot = False

    # Gambar semua bounding box (kantuk & wake)
    frame = draw_boxes(frame, results)
    return frame

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sistem Deteksi Kantuk</title>
            <style>
                body { font-family: 'Segoe UI'; background: #f0f8ff; color: #333; padding: 20px; }
                .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
                img { border-radius: 10px; border: 1px solid #ccc; }
                pre { background: black; color: lime; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sistem Deteksi Kantuk</h1>
                <img src="/video" width="640" height="480"><br><br>
                <p>Total Deteksi Mengantuk: <span id="count">0</span></p>
                <h3>Log Deteksi</h3>
                <pre id="log">Memuat...</pre>
                <a href="/history">&#8594; Lihat Riwayat</a>
            </div>
            <script>
                function refresh() {
                    fetch('/log').then(r => r.text()).then(t => {
                        const log = document.getElementById('log');
                        log.textContent = t;
                        log.scrollTop = log.scrollHeight;
                    });
                    fetch('/stats').then(r => r.json()).then(data => {
                        document.getElementById('count').textContent = data.total_drowsy;
                    });
                }
                setInterval(refresh, 3000);
                refresh();
            </script>
        </body>
        </html>
    '''

@app.route('/video')
def video():
    def generate_frames():
        while True:
            if camera is None:
                time.sleep(1)
                continue
            success, frame = camera.read()
            if not success:
                continue
            frame = detect_drowsiness(frame)
            # Tambahkan timestamp di frame
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def get_log():
    try:
        with open("drowsiness_log.txt", "r") as f:
            return Response(f.read(), mimetype='text/plain')
    except FileNotFoundError:
        return Response("Log belum tersedia.", mimetype='text/plain')

@app.route('/stats')
def stats():
    return jsonify({'total_drowsy': drowsy_count})

@app.route('/snapshots/<filename>')
def get_snapshot(filename):
    return send_from_directory(LOG_IMAGE_DIR, filename)

@app.route('/history')
def history():
    try:
        with open("drowsiness_log.txt", "r") as f:
            rows = ""
            for line in f:
                if "|" in line:
                    timestamp, img = map(str.strip, line.strip().split("|"))
                    rows += f"<tr><td>{timestamp}</td><td><img src='/snapshots/{img}' height='100'></td><td>Mengantuk</td></tr>"
            if not rows:
                rows = "<tr><td colspan='3'>Belum ada data deteksi.</td></tr>"
    except FileNotFoundError:
        rows = "<tr><td colspan='3'>Log tidak ditemukan.</td></tr>"

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Riwayat Deteksi Kantuk</title>
        <style>
            body {{ font-family: 'Segoe UI'; background: #f0f8ff; color: #333; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: center; border: 1px solid #ccc; }}
            img {{ border-radius: 8px; }}
            a {{ text-decoration: none; color: #007BFF; }}
            button {{ padding: 5px 10px; font-size: 14px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Riwayat Deteksi Kantuk</h1>
            <a href="/">&#8592; Kembali ke Halaman Utama</a><br><br>
            <form method="POST" action="/delete_history" onsubmit="return confirm('Yakin ingin menghapus seluruh riwayat?');">
                <button type="submit">Hapus Semua Riwayat</button>
            </form>
            <br>
            <table>
                <tr><th>Timestamp</th><th>Snapshot</th><th>Status</th></tr>
                {rows}
            </table>
        </div>
    </body>
    </html>
    '''

@app.route('/delete_history', methods=['POST'])
def delete_history():
    # Hapus file log
    try:
        if os.path.exists("drowsiness_log.txt"):
            os.remove("drowsiness_log.txt")
        # Hapus semua file di folder snapshot
        for f in os.listdir(LOG_IMAGE_DIR):
            file_path = os.path.join(LOG_IMAGE_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        global drowsy_count
        drowsy_count = 0
    except Exception as e:
        print("Error deleting history:", e)
    return redirect(url_for('history'))

# === THREAD BLYNK LOOP ===
def run_blynk():
    while True:
        try:
            blynk.run()
        except Exception:
            time.sleep(1)

if __name__ == '__main__':
    threading.Thread(target=run_blynk, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)