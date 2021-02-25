#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy
from tensorflow.keras.datasets import cifar100
import tensorflow_datasets as tfds # to load training data
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_data():
	train_ds = tfds.load('cifar100',split='train[:-10%]', batch_size=-1) # this loads a dict with the datasets\
	test_ds = tfds.load('cifar100',split='train[-10%:]', batch_size=-1) # this loads a dict with the datasets\


	train_x= tf.cast(train_ds['image'], tf.float32)/255
	train_y= train_ds['label']
	test_x=tf.cast(test_ds['image'], tf.float32)/255
	test_y=test_ds['label']
	train_labels = to_categorical(train_y)
	test_labels = to_categorical(test_y)
	return train_x, train_labels, train_y, test_x, test_labels, test_y

def show_history(history,key):
	plt.figure()
#	plt.plot(history.history['val_loss'])
	plt.plot(history.history[key])
	plt.title(key)
	plt.ylabel(key)
	plt.xlabel('No. epoch')
	plt.show()
	plt.savefig("./"+key+".jpg")
	numpy.savetxt(key+".txt", history.history[key])
def plot_diagnostics(history):
#plot accuracy curve
	plt.figure()
	line_up = plt.plot(history.history['categorical_accuracy'], color='blue', label='train')
	line_down = plt.plot(history.history['val_categorical_accuracy'], color='red', label='validation')
	#plt.legend([line_up, line_down],['train', 'validation'])
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
			           ncol=2, mode="expand", borderaxespad=0.)
	plt.xlabel('No. epoch')
	plt.ylabel('Accuracy')
	plt.savefig("./Learning_curve.jpg")

def confidence_interval(error,n):
	low = error -1.96 *numpy.sqrt((error * (1-error)) / n)
	high = error + 1.96 *numpy.sqrt((error * (1-error)) / n)
	return low, high

