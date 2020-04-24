import numpy as np
import cv2
from imutils import paths
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle

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

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.1, random_state=0)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

lb = LabelBinarizer().fit(train_y)
train_y = lb.transform(train_y)
test_y = lb.transform(test_y)

# bin = pickle.dumps(lb)
with open("captcha_labels.pickle", "wb") as f:
    pickle.dump(lb, f)

print(train_y.shape, test_y.shape)

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
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=5, epochs=30, verbose=1, callbacks = [estop])

scores = model.evaluate(train_x, train_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")