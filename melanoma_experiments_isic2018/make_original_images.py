import csv
import os
from shutil import copyfile

categories = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']

with open('ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		for category in categories:
			if row[category]=='1.0':
				src = 'ISIC2018_Task3_Training_Input/'+row['image']+'.jpg'
				dst = 'original_images/'+category+'/'+row['image']+'.jpg'
				copyfile(src, dst)