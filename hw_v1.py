#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds # to load training data


# In[ ]:


# Model configuration
batch_size = 500
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = CategoricalCrossentropy()
metrics_function = CategoricalAccuracy()
no_classes = 100
no_epochs = 50
optimizer = Adam()
validation_split = 0.2
verbosity = 1
call_back = EarlyStopping(monitor='loss')

input_shape = (img_width, img_height, img_num_channels)
# ## load data

# In[ ]:


train_ds = tfds.load('cifar100',split='train[:-10%]', batch_size=-1) # this loads a dict with the datasets\
test_ds = tfds.load('cifar100',split='train[-10%:]', batch_size=-1) # this loads a dict with the datasets\

# In[ ]:

train_x= tf.cast(train_ds['image'], tf.float32)/255
train_y= train_ds['label']
test_x=tf.cast(test_ds['image'], tf.float32)/255
test_y=test_ds['label']
train_labels = to_categorical(train_y)
test_labels = to_categorical(test_y)
print(train_x.shape)
print(train_labels.shape)


# ## create model

# In[ ]:


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))



# ## save model
checkpoint_path = "./training_checkpoints"
# In[ ]:
# create a check point for callback function
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

# restore weights
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# ## train model

# In[ ]:
# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=metrics_function)


history = model.fit(
    train_x,train_labels,
    batch_size=batch_size,
    epochs=no_epochs,
    verbose=verbosity,
    callbacks=[call_back,cp_callback],
    validation_split=validation_split
)


# Visualize history
# Plot history: Loss
plt.figure()
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()
plt.savefig("./Val_loss.jpg")
numpy.savetxt("val_loss.txt", history.history['val_loss']);
# Plot history: Accuracy
plt.figure()
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
plt.savefig("./Val_accuracy.jpg")
numpy.savetxt("val_accuracy.txt", history.history['val_categorical_accuracy']);
# Plot history: Loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('loss history')
plt.ylabel('Loss')
plt.xlabel('No. epoch')
plt.show()
plt.savefig("./loss.jpg")
numpy.savetxt("loss.txt", history.history['loss']);
# Plot history: Accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.title('Accuracy history')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.show()
plt.savefig("./accuracy.jpg")
numpy.savetxt("categorical_accuracy.txt", history.history['categorical_accuracy']);

#Confusion matrix
y_pred=model.predict(test_x)
pred_label = tf.argmax(y_pred, axis = 1)
con_mat = tf.math.confusion_matrix(labels=test_y, predictions=pred_label).numpy()
print(con_mat)
numpy.savetxt("confusion_matrix.txt", con_mat)

#wilson confidence interval
#95% 1.96
#error +/- const * sqrt( (error * (1 - error)) / n)
score = model.evaluate(test_x, test_labels, verbose=0)
error = 1-score[1]
print(error)
n = test_x.shape[0]
print(n)
low = error -1.96 *numpy.sqrt((error * (1-error)) / n)
high = error + 1.96 *numpy.sqrt((error * (1-error)) / n)
print("95%% confidence interval:[%.1f%%, %.1f%%]:" % (low*100, high*100))
# ## save model and restore

# In[ ]:


tf.saved_model.save(model, './saved_model')


# In[ ]:


#restored_dense = tf.saved_model.load('/tmp/saved_model')

