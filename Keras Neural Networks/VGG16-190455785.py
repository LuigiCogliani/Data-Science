
#modules
import keras, os, pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt

#dataset
#define train and validation batch size

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data",target_size=(28,28),color_mode='grayscale',classes=['00','01','02','03','04','05','06','07','08','09'],batch_size=2)
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(28,28),color_mode='grayscale',classes=['00','01','02','03','04','05','06','07','08','09'],batch_size=10)


# In[3]:

#model definition
model = Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))


# In[4]:


model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax"))


# In[5]:

#define optimiser and leraning rate
from keras.optimizers import Adam
opt = Adam(lr=0.0003)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# In[ ]:


#RECAP OF THE MODEL

model.summary()


# In[ ]:


#define the number of epochs and stepssteps=10
ep=100
#choose what parameter to monitor for early stopping and model checkpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='accuracy', min_delta=0, patience=99, verbose=1, mode='auto')
#hist = model.fit_generator(steps_per_epoch=steps,generator=traindata, validation_data= testdata, validation_steps=1,epochs=ep,callbacks=[checkpoint])
hist = model.fit_generator(steps_per_epoch=steps,generator=traindata, validation_data= testdata, validation_steps=1,epochs=ep)


# In[ ]:


#visualise accuracy

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()


# In[ ]:


#visualise loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.show()


# In[ ]:


from keras.preprocessing import image
img = image.load_img("prediction/010.png",target_size=(224,224))
plt.imshow(img, cmap='gray')
plt.show()

img = np.asarray(img)

img = np.expand_dims(img, axis=0)

from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
print(shape.output)


# In[ ]:




