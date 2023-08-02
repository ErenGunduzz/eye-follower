import cv2

def detect_eyes(eye_cascade, frame_gray):
    eyes = eye_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (ex, ey, ew, eh) in eyes:
        center = (ex + ew // 2, ey + eh // 2)
        cv2.ellipse(frame, center, (ew // 2, eh // 2), 0, 0, 360, (0, 255, 0), 2)

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open the camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_gray = frame_gray[y:y + h, x:x + w]
            detect_eyes(eye_cascade, face_gray)

        cv2.imshow('Eye Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
