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
	
	data_path = 'loaded_data_original_256x256_0.pckl'
	
	test_pred1_path = sys.argv[1]
	print('test_pred1_path: '+str(test_pred1_path))
	
	test_pred1_weighing = float(sys.argv[2])
	print('test_pred1_weighing: '+str(test_pred1_weighing))
	
	test_pred2_path = sys.argv[3]
	print('test_pred2_path: '+str(test_pred2_path))
	
	test_pred2_weighing = float(sys.argv[4])
	print('test_pred2_weighing: '+str(test_pred2_weighing))

	experiment_number = int(sys.argv[5])
	print('experiment_number: '+str(experiment_number))
	
	######################################################################################
	
	class_names = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
	num_classes = len(class_names)
	
	######################################################################################
	
	root = 'experiment4/'
	test_pred_path = root+'test_pred_'+str(experiment_number)+'.pckl'
	
	######################################################################################
	
	filename = data_path
		
	f = open(filename, 'rb')
	_,y_test = pickle.load(f)
	f.close()
	
	filename = test_pred1_path
	f = open(filename, 'rb')
	y_test_pred1 = pickle.load(f)
	f.close()
	
	filename = test_pred2_path
	f = open(filename, 'rb')
	y_test_pred2 = pickle.load(f)
	f.close()
	
	y_test_list = [y_test_pred1*test_pred1_weighing, y_test_pred2*test_pred2_weighing]
	test_weighing_list = [test_pred1_weighing, test_pred2_weighing]
	y_test_pred = sum(y_test_list) / sum(test_weighing_list)
	
	######################################################################################
	
	
	filename = test_pred_path
	f = open(filename, 'wb')
	pickle.dump(y_test_pred, f)
	f.close()
	
	######################################################################################
	
	compute_categorical_accuracy(y_test, y_test_pred)