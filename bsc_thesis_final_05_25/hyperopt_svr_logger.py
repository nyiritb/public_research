import sys

logfile = open(sys.argv[15]+'/logs/' + sys.argv[16],'a')

if (sys.argv[18]=='rmse'):
	logfile.write('iod: ' + sys.argv[1] + '\n' + 'feature_id: ' + sys.argv[2] + '\n' + 'feature_param: ' + sys.argv[3] + '\n' + 'num_imgs_per_train_subj: ' + sys.argv[4] + '\n' + 'num_imgs_per_test_subj: ' + sys.argv[5] + '\n' + 'num_imgs_per_pers_subj: ' + sys.argv[6] + '\n' 'num_rots_per_train_imgs: ' + sys.argv[7] + '\n' + 'num_rots_per_test_imgs: ' + sys.argv[8] + '\n' + 'num_rots_per_pers_imgs: ' + sys.argv[9] + '\n' + 'svr_eps: ' + sys.argv[10] + '\n' + 'svr_c: ' + sys.argv[11] + '\n' + 'svr_p: ' + sys.argv[12] + "\n" + 'db_type: ' + sys.argv[17] + '\n' + 'rmse: ' + sys.argv[13] + '\n' + 'use_headpose: ' + sys.argv[19] + '\n' + 'use_eyes: ' + sys.argv[20] + "\n\n")
elif (sys.argv[18]=='me'):
	logfile.write('iod: ' + sys.argv[1] + '\n' + 'feature_id: ' + sys.argv[2] + '\n' + 'feature_param: ' + sys.argv[3] + '\n' + 'num_imgs_per_train_subj: ' + sys.argv[4] + '\n' + 'num_imgs_per_test_subj: ' + sys.argv[5] + '\n' + 'num_imgs_per_pers_subj: ' + sys.argv[6] + '\n' 'num_rots_per_train_imgs: ' + sys.argv[7] + '\n' + 'num_rots_per_test_imgs: ' + sys.argv[8] + '\n' + 'num_rots_per_pers_imgs: ' + sys.argv[9] + '\n' + 'svr_eps: ' + sys.argv[10] + '\n' + 'svr_c: ' + sys.argv[11] + '\n' + 'svr_p: ' + sys.argv[12] + "\n" + 'db_type: ' + sys.argv[17] + '\n' + 'mean_error: ' + sys.argv[14] + '\n' + 'use_headpose: ' + sys.argv[19] + '\n' + 'use_eyes: ' + sys.argv[20] + "\n\n")

