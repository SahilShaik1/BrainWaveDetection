# imports
import cv2
import numpy as np
import random
import os
import time
import tensorflow as tf
from tensorflow import keras
# define a video capture object


ID = {
    0 : "Person A",
    1 : "Person B",
    2 : "Person C"
}

limit = 10

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def creation(id):
    name = ID[id]
    os.mkdir(os.path.join(os.getcwd(), name))
    filepath = os.getcwd() + "\\" + name
    return filepath

def control(path, id):
    end = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            temp = np.array(id)
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            nd = img.ravel()
            temp = np.append(temp, nd)
            end.append(temp.tolist())
    return end



face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

dims = None
img_dims = (28, 28)

vid = cv2.VideoCapture(0)
# Training period
trainXA = np.array([])
path = creation(0)
counter = 0
train_period = True
while True and train_period is True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        ROI = frame[y: y+ h, x: x+ h]
        ROI = cv2.resize(ROI, img_dims, cv2.INTER_LINEAR)
        ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        ROI = sp_noise(ROI, .03)
        if counter < limit:
            cv2.imwrite(path + "\\" + str(counter) + ".jpg", ROI)
            counter = counter + 1
        else:
            dims = frame.shape
            train_period = False

path2 = creation(1)
limit_reached = False
print("Person A Finished.")
time.sleep(3)
counter = 0
while True and limit_reached is False:
    ret, frame = vid.read()
    height = dims[0]
    width = dims[1]
    x = random.randint(0, width - img_dims[0])
    y = random.randint(0, height - img_dims[1])
    ROI = frame[y: y + img_dims[1], x: x + img_dims[0]]
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI = sp_noise(ROI, .03)
    cv2.imwrite(path2 + "\\" + str(counter) + ".jpg", ROI)
    counter = counter + 1
    if counter >= limit:
        limit_reached = True

print("Entity B Finished")

# After the loop release    the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


training = control(path, 0)
np.append(training, control(path2, 1))

training = np.array(training)

#Data Recieved
#Starting machine learning
train_ratio = 0.95

train_size = int(limit * train_ratio)
dev_size = limit - train_size



np.random.shuffle(training)
m, n = training.shape

data_train = training[dev_size: m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

data_dev = training[0: dev_size].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=img_dims),
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(16, activation=tf.nn.leaky_relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(X_train.shape)
print(X_dev.shape)

X_train = X_train.reshape(train_size, img_dims[0], img_dims[1])

model.fit(x=X_train, y=Y_train, epochs=200)

X_dev = X_dev.reshape(dev_size, img_dims[0], img_dims[1])

model.evaluate(x=X_dev,y=Y_dev)

