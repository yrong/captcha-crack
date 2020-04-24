from keras.models import model_from_json
import cv2
import numpy as np
import pickle

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

pickle_file = open("captcha_labels.pickle",'rb')
lb = pickle.load(pickle_file)

# evaluate loaded model on test data
loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
operator_image = cv2.imread('data/add/1586508139672.png')
operator_image = cv2.cvtColor(operator_image, cv2.COLOR_BGR2GRAY)
operator_image = cv2.resize(operator_image, (30, 30))
operator_image = np.expand_dims(operator_image, axis=2)
operator_image = np.expand_dims(operator_image, axis=0)
pred = loaded_model.predict(operator_image)
operator = lb.inverse_transform(pred)[0]
print("CAPTCHA operator is: {}".format(operator))
