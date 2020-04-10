import pandas as pd
import numpy as np
import cv2
import glob
import imutils
from imutils import paths
import os
import os.path

data = []
labels = []

operatorDict = {
        "add": 0,
        "sub": 1,
        "mul": 2
    }

for operator in ['add','sub','mul']:
    CAPTCHA_FOLDER = 'data/' + operator
    cnt = 0
    for image in paths.list_images(CAPTCHA_FOLDER):
        cnt = cnt + 1
        if cnt > 20:
            break
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30, 30))

        # adding a 3rd dimension to the image
        img = np.expand_dims(img, axis=2)

        # grabing the name of the letter based on the folder it is present in
        label = image.split(os.path.sep)[-2]

        # appending to the empty lists
        data.append(img)
        labels.append(operatorDict[label])

# converting data and labels to np array
data = np.array(data, dtype="float")
labels = np.array(labels)

print(data.shape, labels.shape)

data = data/255.0

from sklearn.model_selection import train_test_split
(train_x, val_x, train_y, val_y) = train_test_split(data, labels, test_size=0.2, random_state=0)
print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

from sklearn.preprocessing import LabelBinarizer
import pickle
lb = LabelBinarizer().fit(train_y)
train_y = lb.transform(train_y)
val_y = lb.transform(val_y)

bin = pickle.dumps(lb)
with open("captcha_labels.pickle", "wb") as f:
    pickle.dump(lb, f)

print(train_y.shape, val_y.shape)

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(30, 30, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
estop = EarlyStopping(patience=10, mode='min', min_delta=0.01, monitor='val_loss')
model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=10, epochs=50, verbose=1, callbacks = [estop])


# test
operator_image = cv2.imread('data/mul/1586508146362.png')
operator_image = cv2.cvtColor(operator_image, cv2.COLOR_BGR2GRAY)
operator_image = cv2.resize(operator_image, (30, 30))
operator_image = np.expand_dims(operator_image, axis=2)
operator_image = np.expand_dims(operator_image, axis=0)
pred = model.predict(operator_image)
operator = lb.inverse_transform(pred)[0]
print("CAPTCHA operator is: {}".format(operator))