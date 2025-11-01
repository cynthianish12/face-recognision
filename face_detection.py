import cv2
import time
import serial


PORT = 'COM11'         # change if needed
BAUD = 9600
DEADZONE_PIX = 20      # no movement near center
MAX_STEPS = 200        # cap per update
CAM_FOV_DEG = 62.0     # adjust if known

arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_cx, prev_cy = None, None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    current_time = time.time()
    time_diff = current_time - prev_time if prev_time else 1.0

    direction = ""

    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compute offset from frame center for proportional control
        frame_h, frame_w = gray.shape[:2]
        frame_cx = frame_w // 2
        offset_px = cx - frame_cx

        # deadzone
        if abs(offset_px) <= DEADZONE_PIX:
            direction = "Center"
        else:
            # steps per pixel via FOV estimate
            deg_per_px = CAM_FOV_DEG / float(frame_w)
            steps_per_deg = 2048.0 / 360.0  # 28BYJ-48
            steps = int(abs(offset_px) * deg_per_px * steps_per_deg)
            if steps > MAX_STEPS:
                steps = MAX_STEPS

            if offset_px > 0:
                direction = "Right"
                cmd = f"R{steps}\n".encode('ascii')
                arduino.write(cmd)
            else:
                direction = "Left"
                cmd = f"L{steps}\n".encode('ascii')
                arduino.write(cmd)

        prev_cx, prev_cy = cx, cy
        prev_time = current_time
        break

    cv2.putText(frame, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()







