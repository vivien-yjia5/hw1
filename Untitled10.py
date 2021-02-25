#!/usr/bin/env python
# coding: utf-8

# In[128]:


from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# In[140]:


# Model configuration
batch_size = 500
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = CategoricalCrossentropy()
no_classes = 100
no_epochs = 8
optimizer = Adam()
validation_split = 0.2
verbosity = 1
call_back = EarlyStopping()


# ## load data

# # Load CIFAR-10 data
# (input_train, target_train), (input_test, target_test) = cifar100.load_data()

# print(input_train.shape)

# # Determine shape of the data
# input_shape = (img_width, img_height, img_num_channels)
# print(input_shape)

# # Parse numbers as floats
# input_train = input_train.astype('float32')
# input_test = input_test.astype('float32')
# 
# # Normalize data
# input_train = input_train / 255
# input_test = input_test / 255

# In[118]:


train_ds = tfds.load('cifar100',split='train', batch_size=-1) # this loads a dict with the datasets\


# In[132]:


train_x = tf.cast(train_ds['image'], tf.float32)/255
train_y = train_ds['label']
train_labels = to_categorical(train_y)
print(train_x.shape)
print(train_labels.shape)


# ## create model

# In[137]:


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))


# In[141]:


# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])


# ## save model

# In[24]:


# we have to pass in the model (and anything else we want to save) as a kwarg
checkpoint = tf.train.Checkpoint(model=model)

# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
# checkpoint.save is called, the save counter is increased.
save_dir = checkpoint.save('/tmp/training_checkpoints')

# Restore the checkpointed values to the `model` object.
print("The save path is", save_dir)
status = checkpoint.restore(save_dir)
# we can check that everything loaded correctly, this is silent if all is well
status.assert_consumed()


# In[143]:


restored_dense = tf.saved_model.load('/tmp/saved_model')


# In[142]:


history = model.fit(
    train_x,train_labels,
    batch_size=batch_size,
    epochs=no_epochs,
    verbose=verbosity,
    callbacks=call_back,
    validation_split=validation_split
)


# # Fit data to model
# 
# history = model.fit(ds_train,
#             epochs=no_epochs,
#             verbose=verbosity,
#             validation_split=validation_split,
#             callbacks=call_back)

# In[122]:


# Generate generalization metrics
#score = model.evaluate(input_test, target_test, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()


# ## save model

# In[145]:


tf.saved_model.save(model, '/tmp/saved_model')


# In[ ]:




