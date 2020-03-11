import os
import re
import numpy as np
from numpy import sin, cos, radians, degrees, abs, sqrt
import theano
import theano.tensor as T
import lasagne
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image, ImageOps
import cPickle as pickle
from termcolor import colored

import sys

def prepare_data(project_folder, db_type, desired_iod):

	#sets x and y dimension of crop windows
	x_dim = (desired_iod*3)/2
	y_dim = desired_iod/2


	if (db_type.upper() == 'ELTE1'):
		data_folder = project_folder+'/dbs/elte1'
	elif (db_type.upper() == 'ELTE2'):
		data_folder = project_folder+'/dbs/elte2'
	elif (db_type.upper() == 'ELTE3'):
		data_folder = project_folder+'/dbs/elte3'
	elif (db_type.upper() == 'CAVE'):
		data_folder = project_folder+'/dbs/cave'
	else:
		print "DB type not supported."
		return

	paths = open(data_folder+'/contents','r').read().split('\n')

	labels_list_x = []
	labels_list_y = []
	data_list = []
	eyes_list = []
	headpose_list = []
	new_paths = []

	original_img_size_x = 0
	original_img_size_y = 0

	for path in paths:

		path = data_folder+path
		if (db_type.upper()=='ELTE1' or db_type.upper()=='ELTE2' or db_type.upper()=='ELTE3'):
			original_path = path.split("_processed_")[0]+'.png'
		elif (db_type.upper()=='CAVE'):
			original_path = path.split("_processed_")[0]+'.jpg'

		#tests for existence and emptiness of descriptor files
		if (not(os.path.isfile(original_path+'.pars')) or
			not(os.path.isfile(path+'_0.txt.eyescache')) or
			not(os.path.isfile(path+'_0.txt.headposecache')) or
			not(os.path.isfile(path+'_0.txt')) or
			not(os.stat(original_path+'.pars').st_size != 0) or
			not(os.stat(path+'_0.txt.eyescache').st_size != 0) or
			not(os.stat(path+'_0.txt.headposecache').st_size != 0) or
			not(os.stat(path+'_0.txt').st_size != 0)):

			continue

		#reads information from filenames, descriptors and cache files

		#original gaze is always 0 in ELTE databases
		if db_type.upper() == 'ELTE1' or db_type.upper() == 'ELTE2' or db_type.upper() == 'ELTE3':
			original_gaze_x = np.float32(0)
			original_gaze_y = np.float32(0)
		#original gaze is contained in the filename in CAVE databases
		elif db_type.upper() == 'CAVE':
			original_gaze_x = radians(int((path.split('_')[4])[:-1]))
			original_gaze_y = radians(int((path.split('_')[3])[:-1]))

		eyes_l = np.float32(np.loadtxt(path+'_0.txt.eyescache')[:6])
		pupil_l = np.float32(np.loadtxt(path+'_0.txt.eyescache')[-2:-1])
		eyes_l = np.append(eyes_l, pupil_l, axis=0)
		headpose = np.float32(np.loadtxt(path+'_0.txt.headposecache'))

		mesh_rotation_yaw = np.float32(re.search('_yaw_(.*)_pitch_', path).group(1))
		mesh_rotation_pitch = np.float32(re.search('_pitch_(.*)_roll_', path).group(1))

		zface_pitch = np.float32(np.loadtxt(original_path+'.pars')[3])
		zface_yaw = np.float32(np.loadtxt(original_path+'.pars')[4])
		zface_roll = np.float32(np.loadtxt(original_path+'.pars')[5])

		mesh2d = np.float32(np.loadtxt(path+'_0.txt'))

		#calculates middle of left and right eye
		eye_l = (mesh2d[22]+mesh2d[19])/2
		eye_r = (mesh2d[28]+mesh2d[25])/2

		#calculates interocular distance
		iod = abs(mesh2d[22][0]-mesh2d[25][0])

		#converts image to grayscale
		img = Image.open(path).convert('L')


		original_img_size_x = img.size[0]
		original_img_size_y = img.size[1]
		x_dim_before = original_img_size_x
		y_dim_before = original_img_size_y
		x_dim_after = int(x_dim_before*(desired_iod/iod))
		y_dim_after = int(y_dim_before*(desired_iod/iod))
		# resizes picture so actual iod becomes desired iod (the smaller the actual iod, the larger the produced pic)
		img = img.resize((x_dim_after,y_dim_after))

		eye_r[0] = int(eye_r[0]*(desired_iod/iod))
		eye_l[0] = int(eye_l[0]*(desired_iod/iod))
		eye_r[1] = int(eye_r[1]*(desired_iod/iod))
		eye_l[1] = int(eye_l[1]*(desired_iod/iod))
		new_iod = abs(mesh2d[22][0]-mesh2d[25][0])

		# crop to specified x and y dimensions around left pupil
		img = img.crop((eye_l[0]-int(x_dim/2),eye_l[1]-int(y_dim/2),eye_l[0]+int(x_dim/2),eye_l[1]+int(y_dim/2)))

		x_dim = img.size[0]
		y_dim = img.size[1]

		#calculates gaze
		gaze_x_norotation = original_gaze_x - zface_yaw
		gaze_y_norotation = original_gaze_y - zface_pitch

		gaze_x_norotation_noroll = gaze_x_norotation * cos(-zface_roll) - gaze_y_norotation * sin(-zface_roll)
		gaze_y_norotation_noroll = gaze_x_norotation * sin(-zface_roll) + gaze_y_norotation * cos(-zface_roll)

		gaze_x = gaze_x_norotation_noroll + mesh_rotation_yaw
		gaze_y = gaze_y_norotation_noroll + mesh_rotation_pitch

		gaze_x = np.float32(gaze_x)
		gaze_y = np.float32(gaze_y)

		pixels = img.getdata()

		data_list.append(np.float32(pixels))
		eyes_list.append(np.float32(eyes_l))
		headpose_list.append(np.float32(headpose))
		labels_list_x.append(gaze_x)
		labels_list_y.append(gaze_y)
		new_paths.append(path)

		# logs parameters
		with open("results.txt", "a") as myfile:
			myfile.write("**********"+'\n')
			myfile.write("desired_iod: "+str(desired_iod)+'\n')
			myfile.write("db_type: "+str(db_type)+'\n')
			myfile.write("**********"+'\n')

	return data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim

def get_data(db_type, data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim, test_group_id, train_imgs, test_imgs, pers_imgs, train_rots, test_rots, pers_rots, test, use_pretrained_model=False):

	new_data_list = []
	new_eyes_list = []
	new_headpose_list = []
	new_labels_list_x = []
	new_labels_list_y = []

	last_group_id = -1
	last_pic_id = -1
	sum_rots = 0
	sum_pics = 0
	reserved_for_personalization = 20
	test_imgs = test_imgs + reserved_for_personalization

	for i in range(len(new_paths)):
		if (db_type.upper() == 'ELTE1'):
			path = new_paths[i].split('/dbs/elte1')[1]
		elif (db_type.upper() == 'ELTE2'):
			path = new_paths[i].split('/dbs/elte2')[1]
		elif (db_type.upper() == 'ELTE3'):
			path = new_paths[i].split('/dbs/elte3')[1]
		elif (db_type.upper() == 'CAVE'):
			path = new_paths[i].split('/dbs/cave')[1]
		group_id = int(path.split('/')[1])
		if (db_type.upper() == 'ELTE1' or db_type.upper() == 'ELTE2' or db_type.upper() == 'ELTE3'):
			pic_id = int((path.split('/')[2]).split('_')[2])
		elif (db_type.upper() == 'CAVE'):
			pic_id = [int((path.split('/')[2]).split('_')[2][:-1]),int((path.split('/')[2]).split('_')[3][:-1]),int((path.split('/')[2]).split('_')[4][:-1])]

		if (group_id == last_group_id and pic_id == last_pic_id):
			sum_rots = sum_rots + 1
		else:
			sum_rots = 1

		if (group_id == last_group_id and sum_rots == 1):
			sum_pics = sum_pics + 1
		elif (group_id != last_group_id):
			sum_pics = 1

		last_group_id = group_id
		last_pic_id = pic_id

		#omits data if it shouldn't go to specified group according to parameters (training or validation group, amount of training, and personalization pictures per subject and rotations per pictures, etc...)
		if (test and group_id!=test_group_id) or ((not test) and group_id==test_group_id and (sum_pics > pers_imgs or sum_rots > pers_rots)) or ((not test) and group_id!=test_group_id and (sum_pics > train_imgs or sum_rots > train_rots)) or (test and group_id==test_group_id and (sum_pics > test_imgs or sum_rots > test_rots or sum_pics <= reserved_for_personalization)) :
			continue

		new_data_list.append(data_list[i])
		new_labels_list_x.append(labels_list_x[i])
		new_labels_list_y.append(labels_list_y[i])
		new_eyes_list.append(eyes_list[i])
		new_headpose_list.append(headpose_list[i])

	#normalizes data to help speed up convergence
	data = np.array(new_data_list)
	data = data/255.

	eyes = np.array(new_eyes_list)
	eyes = eyes/float(max(original_img_size_x,original_img_size_y))

	headpose = np.array(new_headpose_list)

	#reshapes data for the network's consumption
	labels = np.array([new_labels_list_x,new_labels_list_y])
	labels = labels.transpose()

	#parameter -1: unspecified amount (here: number of examples)
	data = data.reshape(-1, 1, x_dim, y_dim)
	print data.shape
	eyes = eyes.reshape(-1, 14)
	print eyes.shape
	headpose = headpose.reshape(-1, 3)
	print headpose.shape

	#shuffles training data
	if not test:
		data, eyes, headpose, labels = shuffle(data, eyes, headpose, labels, random_state=42)

	return data, eyes, headpose, labels

def train_cnn(lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, X_train, X_train_eyes, X_train_headpose, y_train, X_valid, X_valid_eyes, X_valid_headpose, y_valid, use_headpose, use_eyes, best_weights, use_pretrained_model=False):
	#describes network architecture
	dataset = {
		'train': {'X': X_train, 'eyes': X_train_eyes, 'headpose': X_train_headpose, 'y': y_train},
		'valid': {'X': X_valid, 'eyes': X_valid_eyes, 'headpose': X_valid_headpose, 'y': y_valid}
	}
	input_shape = dataset['train']['X'][0].shape
	l_in = lasagne.layers.InputLayer(
		shape=(None, input_shape[0], input_shape[1], input_shape[2]),
	)
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=16, filter_size=(3, 3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotNormal(gain='relu')
	)
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))
	l_conv2 = lasagne.layers.Conv2DLayer(
		l_pool1, num_filters=32, filter_size=(2, 2),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotNormal(gain='relu')
	)
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))
	l_pool2_dropout = lasagne.layers.DropoutLayer(l_pool2, p=dropout1)
	eyes_shape = dataset['train']['eyes'][0].shape
	l_in_eyes = lasagne.layers.InputLayer(
		shape=(None, eyes_shape[0])
	)
	headpose_shape = dataset['train']['headpose'][0].shape
	l_in_headpose = lasagne.layers.InputLayer(
		shape=(None, headpose_shape[0])
	)
	#concatenates eye and/or headpose information to the net
	if (use_eyes and use_headpose):
		l_pool2_dropout_reshaped = lasagne.layers.ReshapeLayer(
			l_pool2_dropout,
			(-1, (lasagne.layers.get_output_shape(l_pool2_dropout))[1]*(lasagne.layers.get_output_shape(l_pool2_dropout))[2]*(lasagne.layers.get_output_shape(l_pool2_dropout))[3])
		)
		l_concat = lasagne.layers.ConcatLayer(
			[l_pool2_dropout_reshaped, l_in_eyes, l_in_headpose],
			axis = 1
		)
	elif use_eyes:
		l_pool2_dropout_reshaped = lasagne.layers.ReshapeLayer(
			l_pool2_dropout,
			(-1, (lasagne.layers.get_output_shape(l_pool2_dropout))[1]*(lasagne.layers.get_output_shape(l_pool2_dropout))[2]*(lasagne.layers.get_output_shape(l_pool2_dropout))[3])
		)
		l_concat = lasagne.layers.ConcatLayer(
			[l_pool2_dropout_reshaped, l_in_eyes],
			axis = 1
		)
	elif use_headpose:
		l_pool2_dropout_reshaped = lasagne.layers.ReshapeLayer(
			l_pool2_dropout,
			(-1, (lasagne.layers.get_output_shape(l_pool2_dropout))[1]*(lasagne.layers.get_output_shape(l_pool2_dropout))[2]*(lasagne.layers.get_output_shape(l_pool2_dropout))[3])
		)
		l_concat = lasagne.layers.ConcatLayer(
			[l_pool2_dropout_reshaped, l_in_headpose],
			axis = 1
		)
	else:
		l_concat = l_pool2_dropout
	l_hidden1 = lasagne.layers.DenseLayer(
		l_concat, num_units=h1_neurons,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotNormal(gain='relu')
	)
	l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, dropout2)
	l_hidden2 = lasagne.layers.DenseLayer(
		l_hidden1_dropout, num_units=h2_neurons,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotNormal(gain='relu')
	)
	#output units are the x and y angles of the gaze
	l_output = lasagne.layers.DenseLayer(
		l_hidden2,
		num_units=2,
		nonlinearity=lasagne.nonlinearities.identity
	)
	#logs structure of the net
	with open("results.txt", "a") as myfile:
		myfile.write("**********"+'\n')
		myfile.write("net structure:\n"+str(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(l_output)))+'\n')
		myfile.write("**********"+'\n')
	#print out the shape of the net
	print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(l_output))
	#theano uses symbolic variables to store and process data
	net_output = lasagne.layers.get_output(l_output)
	true_output = (T.TensorType(theano.config.floatX, (False, False)))('true_output')
	loss = T.mean(lasagne.objectives.squared_error(net_output, true_output))
	#computes training and validation loss according as squared error of the net's output and the true output
	loss_train = T.mean(lasagne.objectives.squared_error(
		    lasagne.layers.get_output(l_output, deterministic=False), true_output)
	)
	loss_eval = T.mean(lasagne.objectives.squared_error(
		    lasagne.layers.get_output(l_output, deterministic=True), true_output)
	)
	#adds l2 regularization to the loss function
	loss_regularization = lasagne.regularization.regularize_network_params(l_output, lasagne.regularization.l2)
	params = lasagne.layers.get_all_params(l_output, trainable=True)
	loss_train = loss_train + lambda_l2 * loss_regularization
	#adamax updates weights of the network with the help of the loss function
	updates = lasagne.updates.adamax(loss_train, params)
	#warn instaed of giving error in case of unused import, because headpose and eye information can be omitted by design
	train = theano.function([l_in.input_var, l_in_eyes.input_var, l_in_headpose.input_var, true_output], loss_train, updates=updates, on_unused_input='warn')
	get_output = theano.function([l_in.input_var,l_in_eyes.input_var,l_in_headpose.input_var],
                             lasagne.layers.get_output(l_output, deterministic=True), on_unused_input='warn')
	BATCH_SIZE = batch_size
	N_EPOCHS = np.inf
	batch_idx = 0
	epoch = 0

	train_mean_errors = []
	train_rmses = []
	valid_mean_errors = []
	valid_rmses = []

	patience = es_patience
	best_valid_rmse = np.inf
	best_valid_mean_error = np.inf
	best_valid_epoch = 0

	#train model with batch gradient descent until early stopping decides to end it
	#print out progress during training
	while epoch < N_EPOCHS:
		train(dataset['train']['X'][batch_idx:batch_idx + BATCH_SIZE],
			  dataset['train']['eyes'][batch_idx:batch_idx + BATCH_SIZE],
			  dataset['train']['headpose'][batch_idx:batch_idx + BATCH_SIZE],
		      dataset['train']['y'][batch_idx:batch_idx + BATCH_SIZE])
		batch_idx += BATCH_SIZE
		if batch_idx >= dataset['train']['X'].shape[0]:
			batch_idx = 0
			epoch += 1

			if use_pretrained_model and epoch==1:
				lasagne.layers.set_all_param_values(l_output, best_weights)


			val_predictions = get_output(dataset['valid']['X'], dataset['valid']['eyes'], dataset['valid']['headpose'])
			train_predictions = get_output(dataset['train']['X'], dataset['train']['eyes'], dataset['train']['headpose'])

			train_mean_error = degrees(mean_absolute_error(dataset['train']['y'], train_predictions))
			print("Epoch {} training accuracy (mean error in degrees): {}".format(epoch, train_mean_error))
			valid_mean_error = degrees(mean_absolute_error(dataset['valid']['y'], val_predictions))
			print("Epoch {} validation accuracy (mean error in degrees): {}".format(epoch, valid_mean_error))
			train_mean_errors.append(train_mean_error)
			valid_mean_errors.append(valid_mean_error)
			train_rmse = degrees(sqrt(mean_squared_error(dataset['train']['y'], train_predictions)))
			print("Epoch {} training accuracy (RMSE in degrees): {}".format(epoch, train_rmse))
			valid_rmse = degrees(sqrt(mean_squared_error(dataset['valid']['y'], val_predictions)))
			if valid_rmse < best_valid_rmse:
				print("Epoch {} validation accuracy (RMSE in degrees): {}".format(epoch, colored(valid_rmse, 'green')))
			else:
				print("Epoch {} validation accuracy (RMSE in degrees): {}".format(epoch, colored(valid_rmse, 'red')))
			train_rmses.append(train_rmse)
			valid_rmses.append(valid_rmse)

			if valid_rmse < best_valid_rmse:
				best_valid_rmse = valid_rmse
				best_valid_mean_error = valid_mean_error
				best_valid_epoch = epoch
				best_weights = lasagne.layers.get_all_param_values(l_output)
				best_val_predictions = val_predictions
			elif best_valid_epoch + patience <= epoch:
				print(colored("Early stopping.", 'blue'))
				print(colored("Best valid rmse was " + str(best_valid_rmse) + " at epoch " + str(best_valid_epoch), 'blue'))
				print(colored("Best valid mean error was " + str(best_valid_mean_error) + " at epoch " + str(best_valid_epoch), 'blue'))
				lasagne.layers.set_all_param_values(l_output, best_weights)
				break

				train_losses.append(train_rmse)
				valid_losses.append(valid_rmse)
				best_valid_loss = best_valid_rmse

	return best_val_predictions, train_mean_errors, valid_mean_errors, train_rmses, valid_rmses, best_valid_rmse, best_valid_mean_error, best_weights

def crossvalidate(data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim, lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, project_folder, db_type, db_size, train_imgs, test_imgs, pers_imgs, train_rots, test_rots, pers_rots, use_headpose, use_eyes, best_weights_list=None, use_pretrained_model=False):
	#print and log some useful information
	if use_eyes:
		print "USING EYE INFORMATION"
	else:
		print "NOT USING EYE INFORMATION"
	if use_headpose:
		print "USING HEADPOSE INFORMATION"
	else:
		print "NOT USING HEADPOSE INFORMATION"
	#set numpy's threshold to infinity in order to log out all the relevant information, not just part of it
	np.set_printoptions(threshold='nan')
	with open("results.txt", "a") as myfile:
		myfile.write("**********"+'\n')
		myfile.write("db_type: "+str(db_type)+'\n')
		myfile.write("lambda_l2: "+str(lambda_l2)+'\n')
		myfile.write("dropout1: "+str(dropout1)+'\n')
		myfile.write("dropout2: "+str(dropout2)+'\n')
		myfile.write("h1_neurons: "+str(h1_neurons)+'\n')
		myfile.write("h2_neurons: "+str(h2_neurons)+'\n')
		myfile.write("es_patience: "+str(es_patience)+'\n')
		myfile.write("batch_size: "+str(batch_size)+'\n')
		myfile.write("train_imgs: "+str(train_imgs)+'\n')
		myfile.write("test_imgs: "+str(test_imgs)+'\n')
		myfile.write("pers_imgs: "+str(pers_imgs)+'\n')
		myfile.write("train_rots: "+str(train_rots)+'\n')
		myfile.write("test_rots: "+str(test_rots)+'\n')
		myfile.write("pers_rots: "+str(pers_rots)+'\n')
		myfile.write("use_eyes: "+str(use_eyes)+'\n')
		myfile.write("use_headpose: "+str(use_headpose)+'\n')
		myfile.write("**********"+'\n')
	sum_val_mean_errors = 0
	sum_val_rmses = 0
	skipped = 0
	#can load weights of a previous trained net
	if not use_pretrained_model:
		best_weights_list = [0]*(db_size+1)
	#crossvalidate for every subject
	for i in range(1,db_size+1):
		X_train, X_train_eyes, X_train_headpose, y_train = get_data(db_type, data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim, i, train_imgs, test_imgs, pers_imgs, train_rots, test_rots, pers_rots, False, use_pretrained_model)
		X_valid, X_valid_eyes, X_valid_headpose, y_valid = get_data(db_type, data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim, i, train_imgs, test_imgs, pers_imgs, train_rots, test_rots, pers_rots, True, use_pretrained_model)
		#if no test subjects present, skip subject
		if (X_valid.shape[0] == 0):
			skipped = skipped + 1
			print "No image for subject. Skipping subject."
			continue
		#print and log out some useful information
		print "number of training pics: " + str(X_train.shape)
		print "number of test pics: " + str(X_valid.shape)
		print colored("Testing on " + str(i) + ". subject:", attrs=['bold', 'dark', 'blink'])
		with open("results.txt", "a") as myfile:
			myfile.write("**********"+'\n')
			myfile.write("number of training pics: " + str(X_train.shape)+'\n')
			myfile.write("number of test pics: " + str(X_valid.shape)+'\n')
			if use_pretrained_model==True:
				myfile.write("***FURTHER TRAINING PRETRAINED MODEL***")
			myfile.write("Testing on " + str(i) + ". subject:"+'\n')
			myfile.write("**********"+'\n')
		best_val_predictions, train_mean_errors, valid_mean_errors, train_rmses, valid_rmses, best_valid_rmse, best_valid_mean_error, best_weights = train_cnn(lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, X_train, X_train_eyes, X_train_headpose, y_train, X_valid, X_valid_eyes, X_valid_headpose, y_valid, use_headpose, use_eyes, best_weights_list[i], use_pretrained_model)
		best_weights_list[i] = best_weights
		sum_val_rmses += best_valid_rmse
		sum_val_mean_errors += best_valid_mean_error
		with open("results.txt", "a") as myfile:
			myfile.write("train mean errors: " + str(train_mean_errors)+'\n')
			myfile.write("valid mean errors: " + str(valid_mean_errors)+'\n')
			myfile.write("best valid mean error: " + str(best_valid_mean_error)+'\n')
			myfile.write("train rmse's: " + str(train_rmses)+'\n')
			myfile.write("valid rmse's: " + str(valid_rmses)+'\n')
			myfile.write("best valid rmse: " + str(best_valid_rmse)+'\n')
			myfile.write("valid ground truth:\n" + str(y_valid)+'\n')
			myfile.write("valid prediction:\n" + str(best_val_predictions)+'\n')
		with open(project_folder+"/logs/"+"last_net_convergence.log", "w") as myfile:
			myfile.write("train rmses: " + str(train_rmses)+'\n')
			myfile.write("valid rmses: " + str(valid_rmses)+'\n')
	#calculate average loss
	try:
		average_valid_mean_error = sum_val_mean_errors/(db_size-skipped)
		average_valid_rmse = sum_val_rmses/(db_size-skipped)
	except ZeroDivisionError as err:
		print "All subjects skipped. Not enough validation data to do crossvalidation."
		print
		return
	#print and log out some useful information
	with open("results.txt", "a") as myfile:
		myfile.write("**********"+'\n')
		myfile.write("AVERAGE VALID MEAN ERROR: "+str(average_valid_mean_error)+'\n')
		myfile.write("AVERAGE VALID RMSE: "+str(average_valid_rmse)+'\n')
		myfile.write("**********"+'\n')
	print colored(("AVERAGE VALID MEAN ERROR: " + str(average_valid_mean_error)), 'yellow')
	print colored(("AVERAGE VALID RMSE: " + str(average_valid_rmse)), 'yellow')

	return best_weights_list, average_valid_mean_error, average_valid_rmse

def get_results(project_folder, db_type, db_size, desired_iod, use_headpose, use_eyes, num_train_imgs, num_pers_imgs, lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, best_weights_list=None, use_pretrained_model=False):
	"""Read and process data, then perform leave-one-out crossvalidation on it.
	Project folder should exits.
	Database type should be one of the following: 'elte1'/'elte2'/'elte3'/'cave'.
	Database size should match the number of different subjects in the database given.
	Desired interocular distance should be a multiple of 32.
	Number of training images should be a natural number.
	Number of personalization images should be a natural number.
	Either number of training images or number of personalization images should be positive.
	Lambda l2 should be 0 or positive.
	Dropout probability should be between 0 and 1.
	Hidden layer neurons should be a positive integer.
	Early Stopping patience should be a positive integer.
	Batch size should be a positive integer.
	Best weights list should be a list of weights compatible with this neural net.

	Parameters
	----------
	project_folder : String
		Main folder of the project.
	db_type: String
		Name of database to use.
	db_size: Integer
		Number of subjects in the database.
	desired_iod: Integer
		Interocular distance. of the subjects in the resized images.
	use_headpose: Boolean
		Whether or not to concatenate headpose information to the net.
	use_eyes: Boolean
		Whether or not to concatenate eye marker information to the net.
	num_train_imgs: Integer
		Number of training images per subject.
	num_pers_imgs: Integer
		Number of personalization images per subject.
	lambda_l2: Float
		Amount of l2 regularization to use
	dropout1: Float
		Probability for dropout in the first dropout layer.
	dropout2: Float
		Probability for dropout in the second dropout layer.
	h1_neurons: Integer
		Number of neurons in the first hidden layer.
	h2_neurons: Integer
		Number of neurons in the second hidden layer.
	es_patience: Integer
		Number of epochs where the error increases before Early Stopping kicks in.
	batch_size: Integer
		Size of minibatch to use for the descent.
	best_weights_list: List
		List of weights the neural net obtained in a previous training.
	use_pretrained_model: Boolean
		Whether or no to use the weights of a previous model as initial weights.

	Returns
	-------
	best_weights_list : list
		List of weights produced by the net when validation error was the lowest.
	"""

	#load data
	data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim = load_dumped_data(project_folder, db_type, desired_iod)

	#crossvalidate
	try:
		best_weights_list, average_valid_mean_error, average_valid_rmse = crossvalidate(data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim, lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, project_folder, db_type, db_size, num_train_imgs, 999, num_pers_imgs, 10, 1, 10, use_headpose, use_eyes, best_weights_list, use_pretrained_model)
	except (ValueError, TypeError):
		return

	#return weights of model to be able to save it for later use
	return best_weights_list

def dump_prep_data(project_folder, db_type, desired_iod):

	prep_data_path = project_folder+"/data/"+"data_"+str(db_type)+"_"+str(desired_iod)+".p"

	data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim = prepare_data(project_folder, db_type, desired_iod)

	pickle.dump( [data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim], open( prep_data_path, "wb" ) )

def dump_model(project_folder, db_type, desired_iod, best_weights_list):

	model_path = project_folder+"/models/"+"model_"+str(db_type)+"_"+str(desired_iod)+".p"

	pickle.dump( best_weights_list, open( model_path, "wb" ) )

def load_dumped_data(project_folder, db_type, desired_iod):

	prep_data_path = project_folder+"/data/"+"data_"+str(db_type)+"_"+str(desired_iod)+".p"

	if not(os.path.isfile(prep_data_path)):
		print "Dumping model..."
		dump_prep_data(project_folder, db_type, desired_iod)

	data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim = pickle.load( open( prep_data_path, "rb" ) )

	return data_list, eyes_list, original_img_size_x, original_img_size_y, headpose_list, labels_list_x, labels_list_y, new_paths, x_dim, y_dim

def load_dumped_model(project_folder, db_type, desired_iod, best_weights_list):

	model_path = project_folder+"/models/"+"model_"+str(db_type)+"_"+str(desired_iod)+".p"

	if not(os.path.isfile(model_path)):
		print "Dumping model..."
		dump_model(project_folder, db_type, desired_iod, best_weights_list)

	best_weights_list = pickle.load( open( model_path, "rb" ) )

	return best_weights_list
