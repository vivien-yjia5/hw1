#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy
import util_CIFAR as util
import model_CIFAR as md
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Model configuration
batch_size = 500
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = CategoricalCrossentropy()
metrics_function = CategoricalAccuracy()
no_classes = 100
no_epochs = 5
optimizer = Adam()
validation_split = 0.2
verbosity = 1
call_back = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0001)
input_shape = (img_width, img_height, img_num_channels)


# ## load data
train_x, train_labels, train_y, test_x, test_labels, test_y = util.load_data()

#normalize image
train_x = tf.image.per_image_standardization(train_x)
text_x = tf.image.per_image_standardization(test_x)


# ## create model

model = md.model_create(no_classes,input_shape)
model.summary()

# ## save model
checkpoint_path = "./training_checkpoints"
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
util.show_history(history,'val_loss')
util.show_history(history,'val_categorical_accuracy')
util.show_history(history,'loss')
util.show_history(history,'categorical_accuracy')
util.plot_diagnostics(history)

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
print("The loss of test data:%.2f; the accuracy of test data:%.2f:" % (score[0], score[1]))
n = test_x.shape[0]

low, high = util.confidence_interval(score[1],n)
print("95%% confidence interval:[%.1f%%, %.1f%%]:" % (low*100, high*100))
# ## save model and restore

# In[ ]:


tf.saved_model.save(model, './saved_model')


# In[ ]:


#restored_dense = tf.saved_model.load('/tmp/saved_model')

