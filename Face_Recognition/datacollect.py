import cv2

cam = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


id = 0
count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        ROI = gray[y: y + h, x: x + w]
        count = count + 1
        cv2.imwrite('datasets/.' + str(id) + "." + str(count) + ".jpg", ROI)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50, 50, 255), 1)
    if count > 1000:
        break
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
print("Dataset Collection Finished")
