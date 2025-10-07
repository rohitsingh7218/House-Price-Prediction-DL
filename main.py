import cv2
import time
from playsound import playsound

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_cap = cv2.VideoCapture(0)

eyes_closed_time = None  # Track when eyes first close
alarm_playing = False    # To prevent alarm from playing repeatedly

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

        if len(eyes) > 0:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # If eyes are detected → reset timer
    if eyes_detected:
        eyes_closed_time = None
        alarm_playing = False
        cv2.putText(frame, "Eyes Open", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # Eyes closed
        if eyes_closed_time is None:
            eyes_closed_time = time.time()
        else:
            elapsed = time.time() - eyes_closed_time
            if elapsed >= 3 and not alarm_playing:  # Closed for 3 seconds
                cv2.putText(frame, "⚠ EMERGENCY! OPEN YOUR EYES!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                playsound('alarm.mp3')
                alarm_playing = True
            else:
                cv2.putText(frame, "Eyes Closed...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Emergency Eye Detection System", frame)

    if cv2.waitKey(10) == ord('a'):
        break

video_cap.release()
cv2.destroyAllWindows()
