import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


webcam = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

imgCount = 5000

dev_size = 200

total = []

for i in range(int(imgCount / 2)):
    imglist = np.array([0])
    name = "C:/Users/sahil/PycharmProjects/ISEF/imgs/not_sahil/N" + str(i) + ".jpg"
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nd = gray.ravel()
    imglist = np.append(imglist, nd)
    total.append(imglist.tolist())

for i in range(int(imgCount / 2)):
    imglist = np.array([1])
    name = "C:/Users/sahil/PycharmProjects/ISEF/imgs/sahil/S" + str(i) + ".jpg"
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nd = gray.ravel()
    imglist = np.append(imglist, nd)
    total.append(imglist.tolist())

total = np.array(total)

np.random.shuffle(total)

m, n = total.shape

data_train = total[dev_size: m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

data_dev = total[0:dev_size].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

#1000, 2501
print(X_train.shape)
#2500, 900


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(26,26)),
    keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation=tf.nn.sigmoid),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation=tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer = tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        print(f"\n acc: {logs.get('accuracy')}")
        acc = float(logs.get('accuracy'))
        if(acc > .9):
            print(f"Finished, {logs.get('accuracy')}")
            self.model.stop_training = True

callbacker=Callback()



X_train = X_train.reshape((imgCount - dev_size), 26, 26)

model.fit(x=X_train, y=Y_train, epochs=200,callbacks=[callbacker])

X_dev = X_dev.reshape(dev_size, 26, 26)

model.evaluate(x=X_dev,y=Y_dev)

