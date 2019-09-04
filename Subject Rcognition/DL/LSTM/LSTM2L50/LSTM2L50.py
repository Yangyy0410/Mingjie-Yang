import h5py
import numpy as np
import tensorflow as tf 
import keras
#import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.utils import *
from keras import backend as K
from keras.callbacks import *

def convert_to_one_hot(label, classes):
    label = np.eye(classes)[label.reshape(-1)].T
    return label

# Load data.
file = h5py.File('DeepLearning_data.h5','r')
Dldata = file['DeepLearningdata'][:]
Dldata = Dldata*2000
Dllabel = file['DeepLearninglabel'][:]  
file.close()

#Randomly shuffle the dataset.
data_size = Dldata.shape[0]
'''
shuffle_list = np.random.permutation(data_size)
np.save('shuffle_list',shuffle_list)
'''
shuffle_list = np.load('shuffle_list.npy')
data = Dldata[shuffle_list,:,:]
label = Dllabel[shuffle_list]

#Convert labels into one-hot vectors.
label = convert_to_one_hot(label,4).T

#Split dataset into Training set, Validation set and Test set
X_train = data[0:(data_size-1000),:,:]
Y_train = label[0:(data_size-1000),:]

X_val = data[(data_size-1000):(data_size-500),:,:]
Y_val = label[(data_size-1000):(data_size-500),:]

X_test = data[(data_size-500):data_size,:,:]
Y_test = label[(data_size-500):data_size,:]
#Model
#Model architecture
inputdata = Input(name='the_inputs',shape=(X_train.shape[1],X_train.shape[2]))
h1 = Bidirectional(LSTM(50, return_sequences=True))(inputdata)
h2 = Bidirectional(LSTM(50))(h1)
outputs = Dense(4, activation='softmax')(h2)

model = Model(inputs=inputdata, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

filepath = 'ep{epoch:03d}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#Fit Training set.
model.fit(X_train, Y_train, epochs=200, validation_data=(X_val, Y_val),batch_size=64,callbacks=[checkpoint])
#Evaluate on the Test set.
Trainingset_evl = model.evaluate(X_train, Y_train)
#Print the performance
print("Training set Loss = " + str(Trainingset_evl[0]))
print("Training set Accuracy = " + str(Trainingset_evl[1]))

Testset_evl = model.evaluate(X_test, Y_test)
print("Test set Loss = " + str(Testset_evl[0]))
print("Test set Accuracy = " + str(Testset_evl[1]))


