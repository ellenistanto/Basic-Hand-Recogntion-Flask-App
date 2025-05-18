from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Buka webcam
stream = cv2.VideoCapture(0)

previous_time = 0  # Untuk FPS

def gen_frames():
    global previous_time
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        # Proses frame untuk deteksi tangan
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Gambar landmark dan titik
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Hitung FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode dan kirim ke browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routing Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
