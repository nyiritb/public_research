import os
import sys
import subprocess

import preprocess
import hyperopt_search
import dcnn
import plot

def printMainMenu():
	print('Options:')
	print('--------')
	print('-\'prep\': Preprocess raw databases.')
	print('-\'ml\': Estimate gazes.')
	print('-\'plot\': Plot results.')
	print('-[any other string]: Back to the main menu.')
	print('-\'exit\': Back to the terminal.')

def printMlMenu():
	print('Options:')
	print('--------')
	print('-\'prep\': Preprocess raw databases.')
	print('--\'svr\': Search for the optimal hyperparameters of our Support Vector Regressor.')
	print('--\'dcnn\': Use a Deep Convolutional Neural Network.')
	print('-\'plot\': Make some plots.')
	print('-[any other string]: Back to the main menu.')
	print('-\'exit\': Back to the terminal.')


def printPlotMenu():
	print('Options:')
	print('--------')
	print('-\'prep\': Preprocess raw databases.')
	print('-\'ml\': Estimate gazes.')
	print('--\'heatmap\': Make a heatmap of the headposes on a database.')
	print('--\'cnn\': Make plot of training and validation convergence of convolutional neural network.')
	print('--\'svr\': Make plots of logged hyperparameters of the support vector machine.')
	print('-[any other string]: Back to the main menu.')
	print('-\'exit\': Back to the terminal.')

def printDBOptions():
	print('Options:')
	print('--------')
	print('\'elte1\': Use ELTE1 database for crossvalidation')
	print('\'elte2\': Use ELTE2 database for crossvalidation')
	print('\'elte3\': Use ELTE3 database for crossvalidation')
	print('\'cave\': Use CAVE database for crossvalidation')

if __name__ == "__main__":

	print
	project_folder = raw_input("Input project path (leave it blank if you are there): ")
	print
	while ( not(os.path.isdir(project_folder) or project_folder=="") ):
		print "What you entered does not exist as a directory."
		print
		project_folder = raw_input("Input project path (leave it blank if you are there): ")
		print
	if project_folder == "":
		project_folder = subprocess.check_output('pwd').rstrip()
	print('What can I do for you?')
	input = ""

	while(True):

		if (input=='prep'):
			print ""
			db_path = project_folder+'/dbs/'+raw_input("Enter name of database: ")
			print
			while (not(os.path.isdir(db_path))):
				print "What you entered does not exist as a directory."
				print
				db_path = project_folder+'/dbs/'+raw_input("Enter name of database: ")
				print
			rots_per_pic = raw_input("How many times do you want to rotate each image? ")
			print
			while (not(rots_per_pic.isdigit()) or rots_per_pic<=0):
				print "Wrong input. It should be a natural number."
				print
				rots_per_pic = raw_input("How many times do you want to rotate each image? ")
				print
			preprocess.rotate_images(db_path, rots_per_pic)
			print("Done.")
			print("What else would you like me to do?")

		elif (input=='ml'):
			print
			printMlMenu()
			print
			input = raw_input()
			print

			if (input=='svr'):
				err_type = raw_input("What do you want to minimize (RMSE: 'rmse', Mean Error: 'me')?: ")
				print
				while (err_type!='rmse' and err_type!='me'):
					print "Wrong input. It should be either 'rmse' or 'me'."
					print
					err_type = raw_input("What do you want to minimize (RMSE: 'rmse', Mean Error: 'me')?: ")
					print
				logfile = raw_input("Specify a filename for the log file generated (if it already exists, it will be appended to): ")
				print
				evals = raw_input("How many evaluations do you want it to do?: ")
				print
				while (not(evals.isdigit()) or evals<=0):
					print "Wrong input. It should be a natural number."
					print
					evals = raw_input("How many evaluations do you want it to do?: ")
					print
				mongo = raw_input("Do you want to run it with mongodb on a cluster (y/n)?: ")
				print
				while (mongo!='n'):
					if (mongo=='y'):
						print "Sorry. Not implemented in this version. See: \"Developer's Guide: Further Improvements\""
					else:
						print "Wrong input. It should be either 'y' or 'n'."
					print
					mongo = raw_input("Do you want to run it with mongodb on a cluster (y/n)?: ")
					print
				printDBOptions()
				print
				db_type = raw_input()
				print
				while (db_type.upper()!='ELTE1' and db_type.upper()!='ELTE2' and db_type.upper()!='ELTE3' and db_type.upper()!='CAVE'):
					print "What you entered is not one of the databases recognized."
					print
					printDBOptions()
					print
					db_type = raw_input()
					print
				if mongo=='y':
					port = raw_input("What port do you want to run mongodb on? ")
					print
					while (port<=0 or not(port.isdigit())):
						print "Wrong input. It should be a natural number."
						print
						port = raw_input("What port do you want to run mongodb on? ")
						print
				elif mongo=='n':
					port = 0
				hyperopt_search.svr(project_folder, logfile, db_type, err_type, evals, mongo, port)
				print
				print("Done.")
				print("What else would you like me to do?")
			elif (input=='dcnn'):
				printDBOptions()
				print
				db_type = raw_input()
				print
				while (db_type.upper()!='ELTE1' and db_type.upper()!='ELTE2' and db_type.upper()!='ELTE3' and db_type.upper()!='CAVE'):
					print "What you entered is not one of the databases recognized."
					print
					printDBOptions()
					print
					db_type = raw_input()
					print
				print
				if db_type.upper()=='ELTE1':
					db_size = 21
				elif db_type.upper()=='ELTE2':
					db_size = 12
				elif db_type.upper()=='ELTE3':
					db_size = 19
				elif db_type.upper()=='CAVE':
					db_size = 56

				desired_iod=32
				use_headpose=True
				use_eyes=True
				num_train_imgs=999
				num_pers_imgs=0
				lambda_l2=0
				dropout1=0.1
				dropout2=0.1
				h1_neurons=1024
				h2_neurons=1024
				es_patience=10
				batch_size=100
				best_weights_list=None
				use_pretrained_model=False

				print "Showcasing deep convolutional neural network with the following parameters:"
				print "***************************************************************************"
				print "Interocular distance in pixels: "+str(desired_iod)
				print "Using headpose information: "+str(use_headpose)
				print "Using eye marker information: "+str(use_eyes)
				print "Teaching with "+str(num_train_imgs)+" images per subject (10 rotations per images)"
				print "Personalizing with "+str(num_pers_imgs)+" images per subject (10 rotations per images)"
				print "Lambda parameter of l2 regularization: "+str(lambda_l2)
				print "Number of neurons in first hidden layer: "+str(h1_neurons)
				print "Number of neurons in second hidden layer: "+str(h2_neurons)
				print "Early Stopping patience: "+str(es_patience)
				print "Minibatch size: "+str(batch_size)
				print "Pretraining an earlier model: "+str(use_pretrained_model)
				print "***************************************************************************"
				dcnn.get_results(project_folder, db_type, db_size, desired_iod, use_headpose, use_eyes, num_train_imgs, num_pers_imgs, lambda_l2, dropout1, dropout2, h1_neurons, h2_neurons, es_patience, batch_size, best_weights_list, use_pretrained_model)
			else:
				continue

		elif (input=='plot'):
			print
			printPlotMenu()
			input = raw_input()
			print

			if (input=='heatmap'):
				db_path = project_folder+'/dbs/'+raw_input("Enter name of database (leave blank if you want to use all databases available): ")
				print
				while (not(os.path.isdir(db_path))):
					print "What you entered does not exist as a directory."
					print
					db_path = project_folder+'/dbs/'+raw_input("Enter name of database (leave blank if you want to use all databases available): ")
					print
				plot.heatmap(db_path)
			elif (input=='cnn'):
				db_path = project_folder+'/logs/'+raw_input("Enter name of log file: ")
				print
				while (not(os.path.isfile(db_path))):
					print "What you entered does not exist as a file."
					print
					db_path = project_folder+'/logs/'+raw_input("Enter name of log file: ")
					print
				plot.train_valid_convergence(db_path)
			elif (input=='svr'):
				db_path = project_folder+'/logs/'+raw_input("Enter name of log file: ")
				print
				while (not(os.path.isfile(db_path))):
					print "What you entered does not exist as a file."
					print
					db_path = project_folder+'/logs/'+raw_input("Enter name of log file: ")
					print
				plot.hyperparam(db_path)

			else:
				continue

		elif (input=='exit'):
			print
			print("Bye.")
			print
			sys.exit()

		print
		printMainMenu()
		print
		input = raw_input()
