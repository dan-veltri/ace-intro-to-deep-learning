#!/usr/bin/env python
'''
 keras_hello_world.R
 By: Dan Veltri (dan.veltri@gmail.com)
 Created: 12.04.2018
 Last Updated: 01.16.2020 - Updated code and plot_model_history for Tensorflow2
 
 A basic 'hello world' example to test your Keras+TensorFlow install.

 Throughout this course - we'll save features as 'X', and classes
 as 'Y'. So x_train are features of training samples and y_train are
 matching 'ground truth labels for the species of Iris the sample is.

 Iris Data Set: Iris data set is built into R
 Comprised of 3 different Iris species (50 samples each as rows)
 Each flower has 4 features, width and height of sepals and petals
 Features are in the first for cols, flower "class" label is final col
 Challenge: Prediction performance is not so great - can you modify
 the model to get better performance? 
'''

from __future__ import print_function # python3 print support

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets # scikit-learn provides iris data
from sklearn.utils import shuffle

import tensorflow as tf # we can now access tf.keras as needed
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model # needed to load saved models
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import to_categorical

# These are some basic settings for how to train our model
num_epochs = 10  # Rounds of training
num_batches = 16 # No. of samples per patch to train at a time

# Function to enable us to plot our training history
def plot_model_history(model_history, save_plot=True, plot_filename='train_history_plot.png'):
    '''
    On entry: model_history is output of keras model.fit, save_plot is boolean, plot_filename is save location
    On exit: If save_plot is True, figure saved to plot_filename path, else matplot figure is shown.    
    '''
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy on left plot
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'validation'], loc='best')
    
    # summarize history for loss on right plot
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'test'], loc='best')
    
    if save_plot:
        plt.savefig(plot_filename) #if you prefer to save a local copy
        plt.close(fig)
    else:
        plt.show()
    return

# Lets load in the Iris data
# Randomly split data into even train/test (set too small for validation)
iris = datasets.load_iris()
iris.x, iris.y = shuffle(iris.data,iris.target)

x_train = iris.x[:75,0:4]
y_train = to_categorical(iris.y[:75],3) # convert our responses to categories
        
x_test = iris.x[74:,0:4]
y_test = to_categorical(iris.y[74:],3)


# Build the structure of our model (no training yet!)
# We're going to establish a sequential model
# The final 'softmax' activation squishes predictions between 0-1.
# Q: Why is the number of units in the last layer set to 3?
model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(3,activation='softmax'))

# Compile our model- need to do this before using it
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print(model.summary())

print("\nTraining now...")
model_info = model.fit(x_train, y_train, 
                       epochs=num_epochs,
                       batch_size=num_batches,
                       validation_data=(x_test,y_test),
                       verbose=1)

print(model_info.history) # see whats inside model history
plot_model_history(model_info, save_plot=False) # use history to plot

# Example of saving and reloading your model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model
model = load_model('my_model.h5') #reload model

print("\nMaking predictions...")
loss, acc = model.evaluate(x_test,y_test,verbose=0)

print("\nModel Testing Accuracy: {}%".format(np.round(acc,2)))

# END PROGRAM
