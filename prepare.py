import pandas as pd
import numpy as np
import cv2
import glob
import imutils
from imutils import paths
import os
import os.path

ADD_CAPTCHA_FOLDER = "data/add"

data = []
labels = []
for image in paths.list_images(ADD_CAPTCHA_FOLDER):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (30, 30))

    # adding a 3rd dimension to the image
    img = np.expand_dims(img, axis=2)

    # grabing the name of the letter based on the folder it is present in
    label = image.split(os.path.sep)[-2]

    # appending to the empty lists
    data.append(img)
    labels.append(label)

# converting data and labels to np array
data = np.array(data, dtype="float")
labels = np.array(labels)

print(data.shape, labels.shape)