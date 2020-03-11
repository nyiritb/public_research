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
	
def augment_test(filename, crop):

	categories = class_names

	file_root = 'original_images/'
	if crop:
		file_root = 'cropped_images/'

	for i in range(num_aug):
	
		if not os.path.isfile(filename[:-5]+'_'+str(i)+'.pckl'):

			X_train = []
			y_train = []
			
			j = 0
			for category in categories:
				for myfile in os.listdir(file_root+category+'/'):
					X_train.append(file_root+category+'/'+myfile)
					y_train.append(j)
				j += 1

			X_train, y_train = shuffle(X_train, y_train, random_state=0)
			
			X_test = X_train[int(len(X_train)*(1-test_amount)):]
			y_test = y_train[int(len(y_train)*(1-test_amount)):]
			
			for j in range(len(X_test)):
				img = Image.open(X_test[j])
				img = img.resize((img_width,img_height))
				rand_rot = random.randint(1, 360)
				rand_flip = random.choice([True, False])
				rand_mirror = random.choice([True, False])
				img = img.rotate(rand_rot)
				if rand_flip:
					img = ImageOps.flip(img)
				if rand_mirror:
					img = ImageOps.mirror(img)
				X_test[j] = np.asarray(img)

			num_classes = len(categories)

			if K.image_data_format() == 'channels_first':
				X_test = np.array(X_test).reshape(np.array(X_test).shape[0], 3, img_width, img_height)
				input_shape = (3, img_width, img_height)
			else:
				X_test = np.array(X_test).reshape(np.array(X_test).shape[0], img_width, img_height, 3)
				input_shape = (img_width, img_height, 3)

			X_test = X_test.astype('float32')

			X_test /= 255.

			print(X_test.shape[0], 'test samples')

			# convert class vectors to binary class matrices
			y_test = keras.utils.to_categorical(y_test, num_classes)

			f = open(filename[:-5]+'_'+str(i)+'.pckl', 'wb')
			pickle.dump((X_test,y_test), f)
			f.close()

def predict():

	for i in range(num_aug):
		if not os.path.isfile(test_pred_path[:-5]+'_'+str(i)+'.pckl'):
	
			if crop:
				filename=crop_data_path
			else:
				filename=original_data_path
		
			f = open(filename[:-5]+'_'+str(i)+'.pckl', 'rb')
			X_test,y_test = pickle.load(f)
			f.close()

			new_model = load_model(new_model_path)
			
			new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

			new_model.load_weights(load_weights_path)
			print('loaded weights from '+load_weights_path)
			
			y_test_pred = new_model.predict(X_test)
			
			filename = test_pred_path[:-5]+'_'+str(i)+'.pckl'
			f = open(filename, 'wb')
			pickle.dump(y_test_pred, f)
			f.close()

def create_aug_data():
	if crop:
		augment_test(filename=crop_data_path, crop=True)
	else:
		augment_test(filename=original_data_path, crop=False)
		
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

if __name__ == "__main__":

	######################################################################################
	
	crop = int(sys.argv[1])
	if crop==1:
		crop=True
	else:
		crop=False
	print('crop: '+str(crop))
	
	img_width = int(sys.argv[2])
	img_height = img_width
	print('img_width: '+str(img_width))
	
	new_model_path = sys.argv[3]
	print('new_model_path: '+new_model_path)
	
	load_weights_path = sys.argv[4]
	print('load_weights_path: '+load_weights_path)
	
	num_aug = int(sys.argv[5])
	print('num_aug: '+str(num_aug))

	experiment_number = int(sys.argv[6])
	print('experiment_number: '+str(experiment_number))
	
	######################################################################################
	
	optimizer = Adam(lr=1e-5)
	loss = 'categorical_crossentropy'
	metrics = ['categorical_accuracy']
	class_names = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
	num_classes = len(class_names)
	test_amount = 0.05
	
	######################################################################################
	
	root = 'experiment3/'
	original_data_path = 'loaded_data_original_'+str(img_width)+'x'+str(img_height)+'.pckl'
	crop_data_path = 'loaded_data_crop_'+str(img_width)+'x'+str(img_height)+'.pckl'
	test_pred_path = root+'test_pred_'+str(experiment_number)+'.pckl'
	
	######################################################################################
	
	create_aug_data()
	predict()
	
	######################################################################################
	
	filename = original_data_path
	if crop:
		filename = crop_data_path
		
	f = open(filename[:-5]+'_0.pckl', 'rb')
	_,y_test = pickle.load(f)
	f.close()
		
	y_test_list = []
		
	for i in range(num_aug):
		
		filename = test_pred_path[:-5]+'_'+str(i)+'.pckl'
		f = open(filename, 'rb')
		y_test_pred = pickle.load(f)
		y_test_list.append(y_test_pred)
		f.close()
		
	y_test_pred = sum(y_test_list) / float(len(y_test_list))
	
	######################################################################################
	
	filename = test_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_test_pred, f)
	f.close()
	
	######################################################################################
	
	compute_categorical_accuracy(y_test, y_test_pred)