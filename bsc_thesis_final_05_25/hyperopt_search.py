#"pip install pymongo==2.1.1" because recent version won't work

import os
import subprocess
import re
import random
from math import log
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials

def svr(project_folder, logfile, db_type, err_type, evals, mongo, port):
	"""Uses bayesian optimization to search for optimal hyperparameters.
	Logs the results in a log file.
	Project folder should exits.
	Database type should be one of the following: 'elte1'/'elte2'/'elte3'/'cave'.
	Error type should be one of the following: 'rmse'/'me'.
	Number of evaluations should be a positive integer.
	Mongo should be one of the following: 'y'/'n'.
	Port should be a free valid port number.
	Parameters
	----------
	project_folder : String
		Main folder of the project.
	logfile : String
		Desired name of file to log into.
	db_type: String
		Name of database to use.
	err_type: String
		Type of error to minimize.
	evals: Integer
		Number of evaluations.
	mongo: String
		Whether or not to run search on cluster (currently not implemented).
	port: Integer
		Port on which mongodb will run the cluster search (currently not implemented).
	"""

	#define search space
	iod = hp.choice('iod', [15])
	feature_id = hp.choice('feature_id', [1])
	feature_param = hp.choice('feature_param', [24])
	num_imgs_per_train_subj = hp.choice('num_imgs_per_train_subj', [999])
	num_imgs_per_test_subj = hp.choice('num_imgs_per_test_subj', [999])
	num_imgs_per_pers_subj = hp.quniform('num_imgs_per_pers_subj', 0, 20, 1)
	num_rots_per_train_imgs = hp.choice('num_rots_per_train_imgs', [10])
	num_rots_per_test_imgs = hp.choice('num_rots_per_test_imgs', [1])
	num_rots_per_pers_imgs = hp.choice('num_rots_per_pers_imgs', [10])
	svr_eps = hp.choice('svr_eps', [0.01])
	svr_c = hp.choice('svr_c', [0.1])
	svr_p = hp.choice('svr_p', [0.0001])
	use_headpose = hp.choice('use_headpose', [0,1])
	use_eyes = hp.choice('use_eyes', [0,1])

	space = (
			iod,
			feature_id,
			feature_param,
			num_imgs_per_train_subj,
			num_imgs_per_test_subj,
			num_imgs_per_pers_subj,
			num_rots_per_train_imgs,
			num_rots_per_test_imgs,
			num_rots_per_pers_imgs,
			svr_eps,
			svr_c,
			svr_p,
			use_headpose,
			use_eyes
			)

	#define objective function
	def objective(args):
		iod, feature_id, feature_param, num_imgs_per_train_subj, num_imgs_per_test_subj, num_imgs_per_pers_subj, num_rots_per_train_imgs, num_rots_per_test_imgs, num_rots_per_pers_imgs, svr_eps, svr_c, svr_p, use_headpose, use_eyes = args
		path = [str(project_folder)+"/gaze_svm/half-face-tracker", str(iod), str(feature_id), str(feature_param), str(num_imgs_per_train_subj), str(num_imgs_per_test_subj), str(num_imgs_per_pers_subj), str(num_rots_per_train_imgs), str(num_rots_per_test_imgs), str(num_rots_per_pers_imgs), str(svr_eps), str(svr_c), str(svr_p), str(db_type), str(use_headpose), str(use_eyes)]
		print path
		output = subprocess.check_output(path)
		print output
		rmse = float(re.search('Gaze RMSE: (.+?) degrees',output).group(1))
		mean_error = float(re.search('Gaze mean error: (.+?) degrees',output).group(1))
		#log results during search
		subprocess.check_output(["python", str(project_folder)+"/hyperopt_svr_logger.py", str(iod), str(feature_id), str(feature_param), str(num_imgs_per_train_subj), str(num_imgs_per_test_subj), str(num_imgs_per_pers_subj), str(num_rots_per_train_imgs), str(num_rots_per_test_imgs), str(num_rots_per_pers_imgs), str(svr_eps), str(svr_c), str(svr_p), str(rmse), str(mean_error), str(project_folder), str(logfile), str(db_type), str(err_type), str(use_headpose), str(use_eyes)])
		if err_type=='rmse':
			return rmse
		elif err_type=='me':
			return mean_error

	if mongo=='n':
		trials = Trials()
	elif mongo=='y':
		print "Currently not implemented."
		return

	#start search
	best = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=int(evals))
	
