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
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.applications import VGG16, VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
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

if __name__ == "__main__":

	######################################################################################
	
	data_path = 'loaded_data_crop_128x128.pckl'
	
	valid_pred1_path = sys.argv[1]
	print('valid_pred1_path: '+str(valid_pred1_path))
	
	valid_pred1_weighing = float(sys.argv[2])
	print('valid_pred1_weighing: '+str(valid_pred1_weighing))
	
	valid_pred2_path = sys.argv[3]
	print('valid_pred2_path: '+str(valid_pred2_path))
	
	valid_pred2_weighing = float(sys.argv[4])
	print('valid_pred2_weighing: '+str(valid_pred2_weighing))
	
	test_pred1_path = sys.argv[5]
	print('test_pred1_path: '+str(test_pred1_path))
	
	test_pred1_weighing = float(sys.argv[6])
	print('test_pred1_weighing: '+str(test_pred1_weighing))
	
	test_pred2_path = sys.argv[7]
	print('test_pred2_path: '+str(test_pred2_path))
	
	test_pred2_weighing = float(sys.argv[8])
	print('test_pred2_weighing: '+str(test_pred2_weighing))

	experiment_number = int(sys.argv[9])
	print('experiment_number: '+str(experiment_number))
	
	######################################################################################
	
	class_names = ['melanoma','seborrheic keratosis','naevus']
	num_classes = len(class_names)
	
	######################################################################################
	
	root = 'experiment4/'
	valid_pred_path = root+'valid_pred_'+str(experiment_number)+'.pckl'
	test_pred_path = root+'test_pred_'+str(experiment_number)+'.pckl'
	
	######################################################################################
	
	filename = data_path
		
	f = open(filename, 'rb')
	_,_,_,y_valid,_,y_test = pickle.load(f)
	f.close()
	
	filename = valid_pred1_path
	f = open(filename, 'rb')
	y_valid_pred1 = pickle.load(f)
	f.close()
	
	filename = valid_pred2_path
	f = open(filename, 'rb')
	y_valid_pred2 = pickle.load(f)
	f.close()
	
	filename = test_pred1_path
	f = open(filename, 'rb')
	y_test_pred1 = pickle.load(f)
	f.close()
	
	filename = test_pred2_path
	f = open(filename, 'rb')
	y_test_pred2 = pickle.load(f)
	f.close()
	
	y_valid_list = [y_valid_pred1*valid_pred1_weighing, y_valid_pred2*valid_pred2_weighing]
	y_test_list = [y_test_pred1*test_pred1_weighing, y_test_pred2*test_pred2_weighing]
	valid_weighing_list = [valid_pred1_weighing, valid_pred2_weighing]
	test_weighing_list = [test_pred1_weighing, test_pred2_weighing]
	y_valid_pred = sum(y_valid_list) / sum(valid_weighing_list)
	y_test_pred = sum(y_test_list) / sum(test_weighing_list)
	
	######################################################################################
	
	filename = valid_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_valid_pred, f)
	f.close()
	
	filename = test_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_test_pred, f)
	f.close()
	
	######################################################################################
	
	y_test_cat, y_test_pred_cat = compute_categorical_accuracy(y_test=y_test, y_test_pred=y_test_pred)

	######################################################################################
	
	save_roc(y_test=y_test, y_test_pred=y_test_pred)
	save_confusion_matrix(y_test_cat=y_test_cat, y_test_pred_cat=y_test_pred_cat)
	
	######################################################################################

	print_valid_auc(y_valid=y_valid, y_valid_pred=y_valid_pred)
	print_test_auc(y_test=y_test, y_test_pred=y_test_pred)
	
	######################################################################################