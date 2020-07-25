#!/usr/bin/env python
# coding: utf-8

# In[1]:


#modules
import keras, os, pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#dataset
#define train and validation batch size
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data",target_size=(28,28),color_mode='grayscale',classes=['00','01','02','03','04','05','06','07','08','09'],batch_size=100)
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(28,28),color_mode='grayscale',classes=['00','01','02','03','04','05','06','07','08','09'],batch_size=20)


# In[3]:


#model definition

input = keras.layers.Input(shape=(28,28,1))
#stage 0
x0 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(input)

#stage 1
x1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x0)
added = keras.layers.Add()([x0, x1])
x0 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(added)

x1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x0)
added = keras.layers.Add()([x0, x1])
x0 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(added)

#stage 2
x0 = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu")(x0)
x0 = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu")(x0)

x1 = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu")(x0)
added = keras.layers.Add()([x0, x1])
x0 = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu")(added)

x0 = MaxPool2D(pool_size=(2,2),strides=(2,2))(x0)

#stage 3
x0 = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu")(x0)
x0 = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu")(x0)

x1 = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu")(x0)
added = keras.layers.Add()([x0, x1])
x0 = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu")(added)

#stage 3
x0 = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu")(x0)
x0 = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu")(x0)

x1 = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu")(x0)
added = keras.layers.Add()([x0, x1])
x0 = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu")(added)

x0 = MaxPool2D(pool_size=(2,2),strides=(2,2))(x0)
x0 = Flatten()(x0)
out = Dense(units=10, activation="softmax")(x0)
model = keras.models.Model(inputs=[input], outputs=out)
#


# In[ ]:


#summary of the model
model.summary()


# In[4]:


#define optimiser and leraning rate
from keras.optimizers import Adam
opt = Adam(lr=0.00003)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# In[5]:


#define the number of epochs and steps
steps=10
epoch=100

#choose what parameter to monitor for early stopping and model checkpoint
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=99, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=steps,generator=traindata, validation_data= testdata, validation_steps=1,epochs=epoch)


# In[8]:


#visualise accuracy

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.ylim(0.8, 1)
plt.show()


# In[12]:


#visualise loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.show()


# In[ ]:




