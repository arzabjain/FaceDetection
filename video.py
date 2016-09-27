import cv2

face = cv2.CascadeClassifier(
    'C:\\Users\\Arzab-Nehal\\PycharmProjects\\machine_learning\\haarcascade_frontalface_alt.xml')
eye = cv2.CascadeClassifier('C:\\Users\\Arzab-Nehal\\PycharmProjects\\machine_learning\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + h]
        roi_color = frame[y:y + h, x:x + h]
        eyes = eye.detectMultiScale(roi)
        for (ex, ey, eh, ew) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
