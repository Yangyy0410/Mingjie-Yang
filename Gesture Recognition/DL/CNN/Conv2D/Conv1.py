import h5py
import numpy as np
import tensorflow as tf 
import keras
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.utils import *
from keras import backend as K
from keras.callbacks import *

def convert_to_one_hot(label, classes):
    label = np.eye(classes)[label.reshape(-1)].T
    return label
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #plt.figure()
        plt.figure(figsize=(6,10))
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
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

#Convert labels into one-hot vectors and expand dimensions
data  = np.expand_dims(data, axis=3)
label = convert_to_one_hot(label,16).T

#Split dataset into Training set, Validation set and Test set
X_train = data[0:(data_size-2000),:,:]
Y_train = label[0:(data_size-2000),:]

X_val = data[(data_size-2000):(data_size-1000),:,:]
Y_val = label[(data_size-2000):(data_size-1000),:]

X_test = data[(data_size-1000):data_size,:,:]
Y_test = label[(data_size-1000):data_size,:]

#Model
#Model architecture
def CNN(input_shape=(200,6,1), classes=16): 
    X_input = Input(input_shape)
    
    X = Conv2D(filters=32, kernel_size=(20,3), strides=(1,1), activation='relu', padding='same')(X_input)
    X = MaxPooling2D((20,1))(X)

    X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(X)
    X = MaxPooling2D((2,1))(X)
    
    X = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding='valid')(X)
    
    X = Flatten(name='flatten')(X)
    X = Dropout(0.5)(X)
    X = Dense(128,activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X)
    return model
    
model = CNN()
model.summary()
import time
start = time.time()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = LossHistory() 
#Fit Training set.
model.fit(X_train, Y_train, epochs=100, validation_data=(X_val, Y_val),batch_size=64,callbacks=[history])
end = time.time()

#Evaluate on the Test set.
Testset_evl = model.evaluate(X_test, Y_test)
#Print the performance
print("Test set Loss = " + str(Testset_evl[0]))
print("Test set Accuracy = " + str(Testset_evl[1]))
print("time:",end-start)

history.loss_plot('epoch')
