


import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



data = pd.read_csv(r"D:\Downloads_HDD\A_ZHandwritten Data.csv").astype('float32')
#Διαχωριζουμε τις εικόνες απο τα labels του
X = data.drop('0',axis = 1)
y = data['0']
#Splitting the training and test data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)

#All labels are present in the form of a float , we convert them to int and assign each letter a num
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


#Making train and test data too an np array so they can be modified later
train_x = np.array(train_x)
test_x = np.array(test_x)

#We reshape  the train and test data to fit to the model

train_X = train_x.reshape(train_x.shape[0],28,28,1)
print("New shape of train data: ", train_X.shape)
test_X = test_x.reshape(test_x.shape[0], 28, 28,1)
print("New shape of train data: ", test_X.shape)
#Convert the float values to categorical values . so the model takes inputs of labels and outputs a vector of porbabilities
train_yOHE = to_categorical(train_y, num_classes = 26, dtype='int')
print("New shape of train labels: ", train_yOHE.shape)
test_yOHE = to_categorical(test_y, num_classes = 26, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)
#Creating the model 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(26,activation ="softmax"))
#Compiling it , using the optimazition function adam
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X,test_yOHE))

#Printing the result of valisation accuracy
print("The validation accuracy is :", history.history['val_accuracy'])
#Printing the result of training accuracy 
print("The training accuracy is :", history.history['accuracy'])
#Printing the result of validation 
print("The validation loss is :", history.history['val_loss'])
#Printing the result of training loss
print("The training loss is :", history.history['loss'])




#Prediction on external image
img = cv2.imread('C:\\Users\\Nikos\\Desktop\\test.jpg')
img = cv2.GaussianBlur(img, (7,7), 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))

img_pred = word_dict[np.argmax(model.predict(img_final))]
print("The image probably is the letter : ", img_pred)


