import re
import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import data, color, exposure, feature


def hyperparam(db_path):
	"""Logs loss as a function of hyperparameters for every hyperparameter one by one.
	Path of logfile should be an existing file and should contain log in a particular format.
	Parameters
	----------
	db_path : String
		Path of logfile.
	"""

	print "Generating plot..."
	print

	#extracts information from db_path
	try:
		f = open(db_path,'r')
		iod = re.findall('iod: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		feature_id = re.findall('feature_id: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		feature_param = re.findall('feature_param: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_imgs_per_train_subj = re.findall('num_imgs_per_train_subj: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_imgs_per_test_subj = re.findall('num_imgs_per_test_subj: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_imgs_per_pers_subj = re.findall('num_imgs_per_pers_subj: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_rots_per_train_imgs = re.findall('num_rots_per_train_imgs: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_rots_per_test_imgs = re.findall('num_rots_per_test_imgs: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		num_rots_per_pers_imgs = re.findall('num_rots_per_pers_imgs: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		svr_eps = re.findall('svr_eps: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		svr_c = re.findall('svr_c: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		svr_p = re.findall('svr_p: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		rmse = re.findall('rmse: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		mean_error = re.findall('mean_error: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		use_eyes = re.findall('use_eyes: (.+?)\n',f.read())
		f.close()
		f = open(db_path,'r')
		use_headpose = re.findall('use_headpose: (.+?)\n',f.read())
		f.close()
	except IOError as err:
		print "Wrong path."
		print
		return err

	#stores information in numpy arrays
	try:
		iod = np.array([float(i) for i in iod])
		feature_id = np.array([float(i) for i in feature_id])
		feature_param = np.array([float(i) for i in feature_param])
		num_imgs_per_train_subj = np.array([float(i) for i in num_imgs_per_train_subj])
		num_imgs_per_test_subj = np.array([float(i) for i in num_imgs_per_test_subj])
		num_imgs_per_pers_subj = np.array([float(i) for i in num_imgs_per_pers_subj])
		num_rots_per_train_imgs = np.array([float(i) for i in num_rots_per_train_imgs])
		num_rots_per_test_imgs = np.array([float(i) for i in num_rots_per_test_imgs])
		num_rots_per_pers_imgs = np.array([float(i) for i in num_rots_per_pers_imgs])
		svr_eps = np.array([float(i) for i in svr_eps])
		svr_c = np.array([float(i) for i in svr_c])
		svr_p = np.array([float(i) for i in svr_p])
		rmse = np.array([float(i) for i in rmse])
		mean_error = np.array([float(i) for i in mean_error])
		use_headpose = np.array([float(i) for i in use_headpose])
		use_eyes = np.array([float(i) for i in use_eyes])
	except ValueError as err:
		print "Inconsistent log file."
		print
		return err

	# determines which error it should display
	if (rmse.size > 0 and mean_error.size > 0):
		print "Log can't contain both error types."
		print
		return
	elif rmse.size <= 0 and mean_error.size <= 0:
		print "Inconsistent log file."
		print
		return
	elif rmse.size > 0:
		y = rmse
	elif mean_error.size > 0:
		y = mean_error

	def make_ylabels():
		if rmse.size > 0:
			plt.ylabel('RMSE')
		elif mean_error.size > 0:
			plt.ylabel('Mean error')

	#makes labels, displays plots
	try:
		x = iod
		plt.plot(x, y, 'ro')
		plt.xlabel("Interocular distance")
		make_ylabels()
		plt.show()

		x = feature_id
		plt.plot(x, y, 'ro')
		plt.xlabel("1: HOG, 2: LBP")
		make_ylabels()
		plt.show()

		x = feature_param
		plt.plot(x, y, 'ro')
		plt.xlabel("Parameter of feature detector")
		make_ylabels()
		plt.show()

		x = num_imgs_per_train_subj
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of images per training subject")
		make_ylabels()
		plt.show()

		x = num_imgs_per_test_subj
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of images per test subjectr")
		make_ylabels()
		plt.show()

		x = num_imgs_per_pers_subj
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of images per personalization subject")
		make_ylabels()
		plt.show()

		x = num_rots_per_train_imgs
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of rotations per training images")
		make_ylabels()
		plt.show()

		x = num_rots_per_test_imgs
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of rotations per test images")
		make_ylabels()
		plt.show()

		x = num_rots_per_pers_imgs
		plt.plot(x, y, 'ro')
		plt.xlabel("Number of rotations per personalization images")
		make_ylabels()
		plt.show()

		x = use_headpose
		plt.plot(x, y, 'ro')
		plt.xlabel("0: not using headpose information, 1: using headpose information")
		make_ylabels()
		plt.show()

		x = use_eyes
		plt.plot(x, y, 'ro')
		plt.xlabel("0: not using eye markers, 1: using eye markers")
		make_ylabels()
		plt.show()

		plt.xscale('log')
		x = svr_eps
		plt.plot(x, y, 'ro')
		plt.xlabel("SVR epsilon")
		make_ylabels()
		plt.show()

		plt.xscale('log')
		x = svr_c
		plt.plot(x, y, 'ro')
		plt.xlabel("SVR c")
		make_ylabels()
		plt.show()

		plt.xscale('log')
		x = svr_p
		plt.plot(x, y, 'ro')
		plt.xlabel("SVR p")
		make_ylabels()
		plt.show()

	except ValueError as err:
		print "Inconsistent log file."
		print
		return err

def train_valid_convergence(db_path):
	"""Logs training and validation convergence from a log file containing the training and validation rmse of a neural net epoch by epoch (CNN).
	Path of logfile should be an existing file and should contain log in a particular format.
	Parameters
	----------
	db_path : String
		Path of logfile.
	"""

	print "Generating plot..."
	print

	#extracts information from db_path
	try:
		f = open(db_path,'r')
		train_rmses = re.split(',', re.findall('train rmses: \[(.+?)\]\n',f.read())[0])
		f.close()
		f = open(db_path,'r')
		valid_rmses = re.split(',', re.findall('valid rmses: \[(.+?)\]\n',f.read())[0])
		f.close()
	except IOError as err:
		print "Wrong path."
		print
		return err
	except IndexError as err:
		print "Inconsistent log file."
		print
		return err

	#plots convergence of train and valid loss, labels then displays plot
	try:
		plt.plot(train_rmses, linewidth=2, label="train")
		plt.plot(valid_rmses, linewidth=2, label="valid")
		plt.grid()
		plt.legend()
		plt.xlabel("epoch")
		plt.ylabel("rmse")
		plt.ylim(0, 20)
		plt.show()
	except ValueError as err:
		print "Inconsistent log file."
		print
		return err

def heatmap(db_path):
	"""Makes a heatmap from the yaw/pitch degrees of the rotated images contained in input path.
	Path of database should be an existing path.
	Parameters
	----------
	db_path : String
		Path of database.
	"""

	print "Generating plot..."
	print

	#looks for processed images iteratively
	paths = []
	for folder, subs, files in os.walk(db_path):
		for filename in files:
			if filename.endswith('.png') or filename.endswith('.jpg'):
				if "processed" in filename:
					paths.append(os.path.join(folder, filename))

	x = []
	y = []

	#extracts yaw and pitch rotation values from input
	for path in paths:
		x.append(np.degrees(np.float(re.search('_yaw_(.+?)_pitch_', path).group(1))))
		y.append(np.degrees(np.float(re.search('_pitch_(.+?)_roll_', path).group(1))))

	#makes histogram with 120 bins from -60 to +60 degrees in both yaw and pitch (x and y) dimensions, then converts histogram into a heatmap
	heatmap, xedges, yedges = np.histogram2d(y, x, bins=120, range=[[-60,60],[-60,60]])
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	#labels then displays plot
	plt.clf()
	plt.imshow(heatmap, extent=extent, interpolation='bicubic')
	plt.xlabel('Yaw (degrees)')
	plt.ylabel('Pitch (degrees)')
	plt.title('Training data - headposes')
	plt.show()

