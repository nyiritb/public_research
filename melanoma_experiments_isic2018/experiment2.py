######################################################################################
from __future__ import print_function
import os
#os.environ['PYTHONHASHSEED'] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
#np.random.seed(0)
import random
#random.seed(0)
import tensorflow as tf
#tf.set_random_seed(0)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
######################################################################################
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import VGG16, VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
import zipfile
import pickle
import PIL
from PIL import Image
from PIL import ImageOps
from sklearn.utils import shuffle
from os import path
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from scipy import interp
import itertools
from itertools import cycle
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import sys

def save_model1(new_model_path, conv_model_path):
	model = VGG16(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = VGG16(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	transfer_layer = model.get_layer('block5_pool')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)

def save_model2(new_model_path, conv_model_path):
	model = VGG19(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = VGG19(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	transfer_layer = model.get_layer('block5_pool')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model3(new_model_path, conv_model_path):
	model = Xception(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = Xception(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	transfer_layer = model.get_layer('block14_sepconv2_act')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)

def save_model4(new_model_path, conv_model_path):
	model = ResNet50(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = ResNet50(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	transfer_layer = model.get_layer('avg_pool')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model5(new_model_path, conv_model_path):
	model = InceptionV3(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = InceptionV3(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	transfer_layer = model.get_layer('mixed10')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model6(new_model_path, conv_model_path):
	model = DenseNet121(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = DenseNet121(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('bn')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model7(new_model_path, conv_model_path):
	model = DenseNet169(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = DenseNet169(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('bn')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model8(new_model_path, conv_model_path):
	model = DenseNet201(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = DenseNet201(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('bn')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model9(new_model_path, conv_model_path):
	model = InceptionResNetV2(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = InceptionResNetV2(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('conv_7b_ac')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	
def save_model10(new_model_path, conv_model_path):
	model = MobileNet(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = MobileNet(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('conv_pw_13_relu')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	return
	
def save_model11(new_model_path, conv_model_path):
	model = NASNetMobile(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = NASNetMobile(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('?')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	return
	
def save_model12(new_model_path, conv_model_path):
	model = NASNetLarge(
		input_shape=(img_width, img_height, 3),
		include_top=False,
		weights=None
	)
	if pretrained:
		model = NASNetLarge(
			input_shape=(img_width, img_height, 3),
			include_top=False,
			weights='imagenet'
		)
	model.summary()
	transfer_layer = model.get_layer('?')
	conv_model = Model(inputs=model.input,
					   outputs=transfer_layer.output)
	new_model = Sequential()
	new_model.add(conv_model)
	new_model.add(GlobalAveragePooling2D())
	if num_fc_layers>=1:
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=2:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	if num_fc_layers>=3:
		new_model.add(Dropout(dropout))
		new_model.add(Dense(num_fc_neurons, activation='relu'))
	new_model.add(Dense(num_classes, activation='softmax'))

	print(new_model.summary())

	new_model.save(new_model_path)
	conv_model.save(conv_model_path)
	return
	
def load_images():

	categories = class_names

	X_train = []
	y_train = []
	
	file_root = 'original_images/'
	if crop:
		file_root = 'cropped_images/'
		
	i = 0
	for category in categories:
		for myfile in os.listdir(file_root+category+'/'):
			img = Image.open(file_root+category+'/'+myfile)
			img = img.resize((img_width,img_height))
			X_train.append(np.asarray(img))
			y_train.append(i)
		i += 1

	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	
	X_valid = X_train[int(8*len(X_train)/10):int(9*len(X_train)/10)]
	y_valid = y_train[int(8*len(y_train)/10):int(9*len(y_train)/10)]
	X_test = X_train[int(9*len(X_train)/10):]
	y_test = y_train[int(9*len(y_train)/10):]
	X_train = X_train[:int(8*len(X_train)/10)]
	y_train = y_train[:int(8*len(y_train)/10)]
	
	#DEBUG
	print(y_valid)
	print(y_test)
	sys.exit(0)
	#DEBUG

	num_classes = len(categories)

	if K.image_data_format() == 'channels_first':
		X_train = np.array(X_train).reshape(np.array(X_train).shape[0], 3, img_width, img_height)
		X_valid = np.array(X_valid).reshape(np.array(X_valid).shape[0], 3, img_width, img_height)
		X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 3, img_width, img_height)
		input_shape = (3, img_width, img_height)
	else:
		X_train = np.array(X_train).reshape(np.array(X_train).shape[0], img_width, img_height, 3)
		X_valid = np.array(X_valid).reshape(np.array(X_valid).shape[0], img_width, img_height, 3)
		X_test = np.array(X_test).reshape(np.array(X_test).shape[0], img_width, img_height, 3)
		input_shape = (img_width, img_height, 3)

	X_train = X_train.astype('float32')
	X_valid = X_valid.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255.
	X_valid /= 255.
	X_test /= 255.

	print('x_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_valid.shape[0], 'valid samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_valid = keras.utils.to_categorical(y_valid, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return X_train,y_train,X_valid,y_valid,X_test,y_test

def train_model(data, new_model, save_weights_path):

	X_train, y_train, X_valid, y_valid, X_test, y_test = data

	checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True)

	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')

	def train_prep(img):
		return img

	def valid_prep(img):
		return img

	if rotate and flip:
		train_datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True, fill_mode='constant', preprocessing_function=train_prep)
	elif flip:
		train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, preprocessing_function=train_prep)
	elif rotate:
		train_datagen = ImageDataGenerator(rotation_range=180, fill_mode='constant', preprocessing_function=train_prep)
	else:
		train_datagen = ImageDataGenerator(preprocessing_function=train_prep)
	valid_datagen = ImageDataGenerator(preprocessing_function=valid_prep)

	train_datagen.fit(X_train)
	valid_datagen.fit(X_valid)

	history = new_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
							  epochs=epochs,
							  verbose=1,
							  callbacks=[earlyStopping, checkpointer],
							  validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size))
	score = new_model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return history

def do_transfer_learning(num_frozen_layers, optimizer):

	new_model = load_model(new_model_path)
	conv_model = load_model(conv_model_path)

	if not load_weights_path==None:
		new_model.load_weights(load_weights_path)
		print('loaded weights from '+load_weights_path)

	i = 0
	for layer in conv_model.layers:
		i += 1
		if i <= num_frozen_layers:
			layer.trainable = False
		else:
			layer.trainable = True

	for layer in conv_model.layers:
		print('{0}:\t{1}'.format(layer.trainable, layer.name))

	new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	
	X_train,y_train,X_valid,y_valid,X_test,y_test = load_images()

	history = train_model(data=(X_train, y_train, X_valid, y_valid, X_test, y_test), new_model=new_model, save_weights_path=save_weights_path)

	new_model.load_weights(save_weights_path)
	
	y_valid_pred = new_model.predict(X_valid)
	y_test_pred = new_model.predict(X_test)
	
	filename = valid_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_valid_pred, f)
	f.close()
	
	filename = test_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_test_pred, f)
	f.close()

def compute_categorical_accuracy(y_test, y_test_pred):
	y_test_pred_cat = []
	y_test_cat = []
	for i in range(len(y_test)):
		y_test_cat.append(np.argmax(y_test[i]))
	for i in range(len(y_test)):
		y_test_pred_cat.append(np.argmax(y_test_pred[i]))
	print(y_test_cat)
	print(y_test_pred_cat)
	categorical_accuracy = 0.0
	for i in range(len(y_test_cat)):
		if y_test_cat[i]==y_test_pred_cat[i]:
			categorical_accuracy += 1
	categorical_accuracy /= len(y_test_cat)
	print(categorical_accuracy)
	print(len(y_test_cat))
	return y_test_cat, y_test_pred_cat

def create_model():
	if not (os.path.isfile(new_model_path) and os.path.isfile(conv_model_path)):
		if model_id==1:
			save_model1(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==2:
			save_model2(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==3:
			save_model3(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==4:
			save_model4(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==5:
			save_model5(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==6:
			save_model6(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==7:
			save_model7(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==8:
			save_model8(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==9:
			save_model9(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==10:
			save_model10(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==11:
			save_model11(new_model_path=new_model_path, conv_model_path=conv_model_path)
		elif model_id==12:
			save_model12(new_model_path=new_model_path, conv_model_path=conv_model_path)
			
def learn():

	if level == 0:
		num_frozen_layers = 9999
		optimizer = Adam(lr=learning_rate)
	elif level == 1:
		num_frozen_layers = 1
		optimizer = Adam(lr=learning_rate)

	do_transfer_learning(num_frozen_layers=num_frozen_layers, optimizer=optimizer)
		

if __name__ == "__main__":

	######################################################################################
	
	crop = int(sys.argv[1])
	print('crop: '+str(crop))
	
	img_size_level = int(sys.argv[2])
	print('img_size_level: '+str(img_size_level))
	
	model_id = int(sys.argv[3])
	print('model_id: '+str(model_id))
	
	experiment_number = int(sys.argv[4])
	print('experiment_number: '+str(experiment_number))
	
	######################################################################################
	
	load_weight_from_last_experiment = False
	num_fc_neurons = 1024
	dropout = 0.5
	learning_rate = 1e-05
	flip = 1
	rotate = 1
	level = 1
	pretrained = 1
	
	num_fc_layers = 2
	if model_id >= 3:
		num_fc_layers = 0
	
	######################################################################################
	
	loss = 'categorical_crossentropy'
	metrics = ['categorical_accuracy']
	class_names = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
	num_classes = len(class_names)
	
	######################################################################################
	
	if img_size_level == 1:
		img_width = 128
		img_height = img_width
	
	elif img_size_level == 2:
		img_width = 256
		img_height = img_width
			
	elif img_size_level == 3:
		img_width = 384
		img_height = img_width
		
	elif img_size_level == 4:
		img_width = 512
		img_height = img_width
		
	batch_size = 10
	if model_id >= 3:
		batch_size = 5
	
	epochs = 9999
	patience = 5
	
	######################################################################################
	
	root = 'experiment2/'
	load_weights_path = root+'weights'+str(experiment_number-1)+'.hdf5'
	if load_weight_from_last_experiment == 0:
		load_weights_path = None
	save_weights_path = root+'weights'+str(experiment_number)+'.hdf5'
	new_model_path = 'new_model_'+str(model_id)+'_'+str(img_width)+'x'+str(img_height)+'_pretrained='+str(pretrained)+'.h5'
	conv_model_path = 'conv_model_'+str(model_id)+'_'+str(img_width)+'x'+str(img_height)+'_pretrained='+str(pretrained)+'.h5'
	valid_pred_path = root+'valid_pred_'+str(experiment_number)+'.pckl'
	test_pred_path = root+'test_pred_'+str(experiment_number)+'.pckl'
	
	######################################################################################
	
	create_model()
	learn()