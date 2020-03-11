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
from keras.applications.nasnet import NASNetLarge
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
	
def pickle_images(filename, crop):

	X_train = []
	y_train = []
	X_valid = []
	y_valid = []
	X_test = []
	y_test = []
	
	if crop:
		trm_path = 'ISIC2017/train_bounding_box_cropped/m/'
		trs_path = 'ISIC2017/train_bounding_box_cropped/s/'
		trn_path = 'ISIC2017/train_bounding_box_cropped/n/'

		vm_path = 'ISIC2017/valid_bounding_box_cropped/m/'
		vs_path = 'ISIC2017/valid_bounding_box_cropped/s/'
		vn_path = 'ISIC2017/valid_bounding_box_cropped/n/'

		tem_path = 'ISIC2017/test_bounding_box_cropped/m/'
		tes_path = 'ISIC2017/test_bounding_box_cropped/s/'
		ten_path = 'ISIC2017/test_bounding_box_cropped/n/'
		
	else:
		trm_path = 'ISIC2017/train/m/'
		trs_path = 'ISIC2017/train/s/'
		trn_path = 'ISIC2017/train/n/'

		vm_path = 'ISIC2017/valid/m/'
		vs_path = 'ISIC2017/valid/s/'
		vn_path = 'ISIC2017/valid/n/'

		tem_path = 'ISIC2017/test/m/'
		tes_path = 'ISIC2017/test/s/'
		ten_path = 'ISIC2017/test/n/'

	for myfile in os.listdir(trm_path):
		img = Image.open(trm_path+myfile)
		img = img.resize((img_width,img_height))
		X_train.append(np.asarray(img))
		y_train.append(0)
	for myfile in os.listdir(trs_path):
		img = Image.open(trs_path+myfile)
		img = img.resize((img_width,img_height))
		X_train.append(np.asarray(img))
		y_train.append(1)
	for myfile in os.listdir(trn_path):
		img = Image.open(trn_path+myfile)
		img = img.resize((img_width,img_height))
		X_train.append(np.asarray(img))
		y_train.append(2)

	for myfile in os.listdir(vm_path):
		img = Image.open(vm_path+myfile)
		img = img.resize((img_width,img_height))
		X_valid.append(np.asarray(img))
		y_valid.append(0)
	for myfile in os.listdir(vs_path):
		img = Image.open(vs_path+myfile)
		img = img.resize((img_width,img_height))
		X_valid.append(np.asarray(img))
		y_valid.append(1)
	for myfile in os.listdir(vn_path):
		img = Image.open(vn_path+myfile)
		img = img.resize((img_width,img_height))
		X_valid.append(np.asarray(img))
		y_valid.append(2)

	for myfile in os.listdir(tem_path):
		img = Image.open(tem_path+myfile)
		img = img.resize((img_width,img_height))
		X_test.append(np.asarray(img))
		y_test.append(0)
	for myfile in os.listdir(tes_path):
		img = Image.open(tes_path+myfile)
		img = img.resize((img_width,img_height))
		X_test.append(np.asarray(img))
		y_test.append(1)
	for myfile in os.listdir(ten_path):
		img = Image.open(ten_path+myfile)
		img = img.resize((img_width,img_height))
		X_test.append(np.asarray(img))
		y_test.append(2)

	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
	X_test, y_test = shuffle(X_test, y_test, random_state=0)

	num_classes = 3

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

	f = open(filename, 'wb')
	pickle.dump((X_train,y_train,X_valid,y_valid,X_test,y_test), f)
	f.close()

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


	num_cols = 3
	num_rows = 3

	fig, ax = plt.subplots(num_cols, num_rows)
	fig.tight_layout()
	for X_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=num_cols*num_rows):
		for i in range(0, num_cols):
			for j in range(0, num_rows):
				ax[i,j].imshow(X_batch[i*3+j])
				if y_batch[i*3+j][0]==1:
					title = ax[i,j].set_title('melanoma')
				elif y_batch[i*3+j][1]==1:
					title = ax[i,j].set_title('seborrheic keratosis')
				elif y_batch[i*3+j][2]==1:
					title = ax[i,j].set_title('naevus')
				ax[i,j].axis('off')
		plt.savefig(root+'random_train_images_'+str(experiment_number)+'.png')
		break

	fig, ax = plt.subplots(num_cols, num_rows)
	fig.tight_layout()
	for X_batch, y_batch in valid_datagen.flow(X_valid, y_valid, batch_size=num_cols*num_rows):
		for i in range(0, num_cols):
			for j in range(0, num_rows):
				ax[i,j].imshow(X_batch[i*3+j])
				if y_batch[i*3+j][0]==1:
					title = ax[i,j].set_title('melanoma')
				elif y_batch[i*3+j][1]==1:
					title = ax[i,j].set_title('seborrheic keratosis')
				elif y_batch[i*3+j][2]==1:
					title = ax[i,j].set_title('naevus')
				ax[i,j].axis('off')
		plt.savefig(root+'random_valid_images_'+str(experiment_number)+'.png')
		break


	history = new_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
							  epochs=epochs,
							  verbose=1,
							  callbacks=[earlyStopping, checkpointer],
							  validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size))
	score = new_model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return history

def plot_history(history):
	# summarize history for accuracy
	plt.clf()
	plt.plot(history.history['categorical_accuracy'])
	plt.plot(history.history['val_categorical_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.savefig(root+'accuracy_plot_'+str(experiment_number)+'.png')
	# summarize history for loss
	plt.clf()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.savefig(root+'loss_plot_'+str(experiment_number)+'.png')

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
	
	filename = original_data_path
	if crop:
		filename = crop_data_path
	f = open(filename, 'rb')
	X_train,y_train,X_valid,y_valid,X_test,y_test = pickle.load(f)
	f.close()

	history = train_model(data=(X_train, y_train, X_valid, y_valid, X_test, y_test), new_model=new_model, save_weights_path=save_weights_path)

	plot_history(history=history)

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

def save_roc(y_test, y_test_pred):
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(num_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_test_pred.ravel())
	roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

	# Compute macro-average ROC curve and ROC area

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(num_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= num_classes

	fpr['macro'] = all_fpr
	tpr['macro'] = mean_tpr
	roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

	# Plot all ROC curves
	plt.clf()
	plt.figure()
	plt.plot(fpr['micro'], tpr['micro'],
			 label='micro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc['micro']),
			 color='blue', linestyle=':', linewidth=4)

	plt.plot(fpr['macro'], tpr['macro'],
			 label='macro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc['macro']),
			 color='navy', linestyle=':', linewidth=4)

	lw=2

	colors = cycle(['red', 'yellow', 'green'])
	class_names = cycle(['melanoma','seborrheic keratosis','naevus'])
	for i, color, class_name in zip(range(num_classes), colors, class_names):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw,
				 label='ROC curve for {0}'''.format(class_name))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating characteristic curve for skin lesion prediction')
	plt.legend(loc='lower right')
	plt.savefig(root+'roc_curve_'+str(experiment_number)+'.png')

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment='center',
				 color='white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def save_confusion_matrix(y_test_cat, y_test_pred_cat):
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test_cat, y_test_pred_cat)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.clf()
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
						  title='Confusion matrix, without normalization')
	plt.savefig(root+'confusion_matrix_'+str(experiment_number)+'.png')
	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
						  title='Normalized confusion matrix')

	plt.savefig(root+'confusion_matrix_normalized_'+str(experiment_number)+'.png')

def cat_num_to_name(cat_num):
	if (int(cat_num) == 0 or int(cat_num) == 667):
		return 'M'
	elif (int(cat_num) == 1 or int(cat_num) == 1179):
		return 'SK'
	else:
		return 'N'

def save_random_prediction_images(X_test, y_test_cat, y_test_pred_cat):
	num_cols = 4
	num_rows = 4
	fig, ax = plt.subplots(num_cols, num_rows)
	fig.tight_layout()
	for i in range(0, num_rows):
		for j in range(0, num_cols):
			ax[i,j].imshow(X_test[i*num_rows+j])
			title = ax[i,j].set_title('GT: '+cat_num_to_name(y_test_cat[i*num_rows+j])+', Pred: '+cat_num_to_name(y_test_pred_cat[i*num_rows+j]))
			ax[i,j].axis('off')
	plt.savefig(root+'random_predictions_'+str(experiment_number)+'.png')

def print_valid_auc(y_valid, y_valid_pred):
	two_class_y_valid = [value[0] for value in y_valid]

	two_class_y_valid_pred = [value[0] for value in y_valid_pred]

	two_class_y_valid = np.array(two_class_y_valid)
	two_class_y_valid_pred = np.array(two_class_y_valid_pred)

	mm_vs_rest_score = roc_auc_score(two_class_y_valid, two_class_y_valid_pred)

	print('valid mm vs rest auc: '+str(mm_vs_rest_score))

	two_class_y_valid = [value[1] for value in y_valid]

	two_class_y_valid_pred = [value[1] for value in y_valid_pred]

	two_class_y_valid = np.array(two_class_y_valid)
	two_class_y_valid_pred = np.array(two_class_y_valid_pred)

	sk_vs_rest_score = roc_auc_score(two_class_y_valid, two_class_y_valid_pred)

	print('valid sk vs rest auc: '+str(sk_vs_rest_score))

	print('valid average auc: '+str((mm_vs_rest_score+sk_vs_rest_score)/2.))

def print_test_auc(y_test, y_test_pred):
	two_class_y_test = [value[0] for value in y_test]

	two_class_y_test_pred = [value[0] for value in y_test_pred]

	two_class_y_test = np.array(two_class_y_test)
	two_class_y_test_pred = np.array(two_class_y_test_pred)

	mm_vs_rest_score = roc_auc_score(two_class_y_test, two_class_y_test_pred)

	print('test mm vs rest auc: '+str(mm_vs_rest_score))

	two_class_y_test = [value[1] for value in y_test]

	two_class_y_test_pred = [value[1] for value in y_test_pred]

	two_class_y_test = np.array(two_class_y_test)
	two_class_y_test_pred = np.array(two_class_y_test_pred)

	sk_vs_rest_score = roc_auc_score(two_class_y_test, two_class_y_test_pred)

	print('test sk vs rest auc: '+str(sk_vs_rest_score))

	print('test average auc: '+str((mm_vs_rest_score+sk_vs_rest_score)/2.))

def create_data_and_model():
	if crop:
		if not os.path.isfile(crop_data_path):
			pickle_images(filename=crop_data_path, crop=True)
	else:
		if not os.path.isfile(original_data_path):
			pickle_images(filename=original_data_path, crop=False)
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
			
def learn():

	data_path = original_data_path
	if crop:
		data_path = crop_data_path
	
	f = open(data_path, 'rb')
	X_train,y_train,X_valid,y_valid,X_test,y_test = pickle.load(f)
	f.close()
	
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

	level = int(sys.argv[3])
	print('level: '+str(level))
	
	pretrained = int(sys.argv[4])
	if pretrained == 1:
		pretrained = True
	else:
		pretrained = False
	
	model_id = int(sys.argv[5])
	print('model_id: '+str(model_id))
	
	experiment_number = int(sys.argv[6])
	print('experiment_number: '+str(experiment_number))
	
	######################################################################################
	
	load_weight_from_last_experiment = False
	num_fc_neurons = 1024
	dropout = 0.5
	learning_rate = 1e-05
	flip = 1
	rotate = 1
	
	num_fc_layers = 2
	if model_id >= 3:
		num_fc_layers = 0
	
	######################################################################################
	
	loss = 'categorical_crossentropy'
	metrics = ['categorical_accuracy']
	class_names = ['melanoma','seborrheic keratosis','naevus']
	num_classes = len(class_names)
	
	######################################################################################
	
	#POSSIBLE CONFIGURATIONS
	#model 1: 64,96,128,160          [64,160]  4
	#model 2: 64,96,128,160          [64,160]  4
	#model 3: 96,128,160,192         [96,192]  4
	#model 4: 224                    [224,224] 1
	#model 5: 160,192,224,256        [160,256] 4
	#model 6: 224                    [224,224] 1
	
	#STAGE1
	#model 1: 128
	#model 2: 128
	#model 3: 160
	#model 4: 224
	#model 5: 224
	#model 6: 224
	
	#STAGE2
	#model 1: 160
	#model 2: 160
	#model 3: 192
	#model 5: 256
	
	#STAGE3
	#model 1: 96
	#model 2: 96
	#model 3: 128
	#model 5: 192
	
	#STAGE4
	#model 1: 64
	#model 2: 64
	#model 3: 96
	#model 5: 160
	
	if img_size_level == 1:
		img_width = 128
		img_height = img_width
		if model_id==3:
			img_width = 160
			img_height = img_width
		elif model_id == 4 or model_id==5 or model_id==6:
			img_width = 224
			img_height = img_width
	
	elif img_size_level == 2:
		img_width = 160
		img_height = img_width
		if model_id==3:
			img_width = 192
			img_height = img_width
		elif model_id==5:
			img_width = 256
			img_height = img_width
			
	elif img_size_level == 3:
		img_width = 96
		img_height = img_width
		if model_id==3:
			img_width = 128
			img_height = img_width
		elif model_id==5:
			img_width = 192
			img_height = img_width
			
	elif img_size_level == 4:
		img_width = 64
		img_height = img_width
		if model_id==3:
			img_width = 96
			img_height = img_width
		elif model_id==5:
			img_width = 160
			img_height = img_width
		
	batch_size = 10
	if model_id >= 3:
		batch_size = 5
	
	epochs = 100
	patience = 10
	
	######################################################################################
	
	root = 'experiment2/'
	load_weights_path = root+'weights'+str(experiment_number-1)+'.hdf5'
	if load_weight_from_last_experiment == 0:
		load_weights_path = None
	save_weights_path = root+'weights'+str(experiment_number)+'.hdf5'
	original_data_path = 'loaded_data_original_'+str(img_width)+'x'+str(img_height)+'.pckl'
	crop_data_path = 'loaded_data_crop_'+str(img_width)+'x'+str(img_height)+'.pckl'
	new_model_path = 'new_model_'+str(model_id)+'_'+str(img_width)+'x'+str(img_height)+'_pretrained='+str(pretrained)+'.h5'
	conv_model_path = 'conv_model_'+str(model_id)+'_'+str(img_width)+'x'+str(img_height)+'_pretrained='+str(pretrained)+'.h5'
	valid_pred_path = root+'valid_pred_'+str(experiment_number)+'.pckl'
	test_pred_path = root+'test_pred_'+str(experiment_number)+'.pckl'
	
	######################################################################################
	
	create_data_and_model()
	learn()
	
	######################################################################################
	
	filename = original_data_path
	if crop:
		filename = crop_data_path
	f = open(filename, 'rb')
	_,_,_,y_valid,X_test,y_test = pickle.load(f)
	f.close()
	
	filename = valid_pred_path
	f = open(filename, 'rb')
	y_valid_pred = pickle.load(f)
	f.close()
	
	filename = test_pred_path
	f = open(filename, 'rb')
	y_test_pred = pickle.load(f)
	f.close()
	
	######################################################################################
	
	y_test_cat, y_test_pred_cat = compute_categorical_accuracy(y_test=y_test, y_test_pred=y_test_pred)

	######################################################################################
	
	save_roc(y_test=y_test, y_test_pred=y_test_pred)
	save_confusion_matrix(y_test_cat=y_test_cat, y_test_pred_cat=y_test_pred_cat)
	save_random_prediction_images(X_test=X_test, y_test_cat=y_test_cat, y_test_pred_cat=y_test_pred_cat)
	
	######################################################################################

	print_valid_auc(y_valid=y_valid, y_valid_pred=y_valid_pred)
	print_test_auc(y_test=y_test, y_test_pred=y_test_pred)
	
	######################################################################################