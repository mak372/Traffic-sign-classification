import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential,load_model,save_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

data = []
labels = []
classes = 43
train_path = os.getcwd()
for i in range(classes):
  path = os.path.join(train_path,'train',str(i))
  images = os.listdir(path)
  for j in images:
    image = Image.open(path + '//'+ j)
    image = image.resize((70,70))
    image = np.array(image)
    data.append(image)
    labels.append(i)
data = np.array(data)
labels = np.array(labels)
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=43)
y_train = to_categorical(y_train,43)
y_test = to_categorical(y_test,43)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=64,epochs=5,validation_data=(X_test,y_test))
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
test_imgs = y_test["Path"].values
test_images = []
for images in test_imgs:
  image = Image.open(images)
  image = image.resize((30,30))
  test_images.append(np.array(image))
X_test = np.array(test_images)
prediction1 = model.predict(X_test)
prediction = np.argmax(prediction1,axis=1)
model.save("sign.h5")

