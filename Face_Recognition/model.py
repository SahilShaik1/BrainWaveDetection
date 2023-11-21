import cv2

cam = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("Trainer.yml")

name_list = ["Sahil",""]
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        ROI = gray[y: y + h, x: x + w]
        cv2.imshow("ROI", ROI)
        serial, conf = recognizer.predict(ROI)
        conf = 100 - conf
        if conf < 0:
            conf = conf * -1
        if conf>50:
            cv2.putText(frame, name_list[serial], (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (50, 50, 255), 1)
        else :
            cv2.putText(frame, "Unknown Person Detected", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()    
