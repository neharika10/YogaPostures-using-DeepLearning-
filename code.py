import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
DATADIR="E:/1. Neharika/mini project/dataset/"
CATEGORIES=["headstand","plough","fish","shoulderstand"]

#LOADING DATASET
for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break
    
 IMAGE_SIZE=50

#RESIZING PHOTOS
new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()


#TRAINING FUNCTION
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array= cv2.resize(img_array,(IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
import random

#SHUFFLING FOR BETTER TRAINING
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
 x=[]
y=[]
for features, label in training_data:
    x.append(features)
    y.append(label)
    
x= np.array(x).reshape(-1, IMAGE_SIZE,IMAGE_SIZE, 1)
import pickle
pickle_out= open("X.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out= open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in= open("x.pickle","rb")
x= pickle.load(pickle_in)

#TRAINING DATA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import normalize
import pickle
import time

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
y=np.array(y)
x= x/255.0


dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(4))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(x, y,
                      batch_size=20,
                      epochs=10,
                      validation_split=0.1,
                      callbacks=[tensorboard])
 

#TRAINING MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("x.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
y=np.array(y)
x = x/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(4))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'],
                          )

            model.fit(x, y,
                      batch_size=20,
                      epochs=10,
                      validation_split=0.1,
                      callbacks=[tensorboard])

model.save('64x3-CNN.model')

#PREDICTIONS
import cv2
import tensorflow as tf

CATEGORIES = ["headstand","plough","fish","shoulderstand"]


def prepare(filepath):
    IMAGE_SIZE = 50 # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    return new_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")
prediction = model.predict([prepare('plough1.jfif')])
#print(prediction[0])

ans= prediction.tolist()#converting numpy array to list
ans= ans[0]
print (ans)
for i in range(0, len(ans)):
    if (j==1.0):
        ind= ans.index(j)
print(CATEGORIES[ind])
        
