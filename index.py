import cv2
import time

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Previous center and time
prev_cx, prev_cy = None, None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute center
        cx = x + w // 2
        cy = y + h // 2

        direction = ""
        speed = 0

        # Compare with previous center
        if prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy
            dt = time.time() - prev_time
            if dt > 0:
                speed = ((dx**2 + dy**2)**0.5) / dt  # px/s

            # Determine movement direction
            if abs(dx) > abs(dy):
                if dx > 0:
                    direction = "Right"
                elif dx < 0:
                    direction = "Left"
            else:
                if dy > 0:
                    direction = "Down"
                elif dy < 0:
                    direction = "Up"

        prev_cx, prev_cy = cx, cy
        prev_time = time.time()

        # Overlay direction and speed
        cv2.putText(frame, f"Direction: {direction}", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed:.2f} px/s", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Movement Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
