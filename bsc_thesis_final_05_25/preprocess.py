import subprocess
import os
import random
import sys
import re
import numpy as np
from numpy import sin, cos, radians, abs
from scipy import spatial
from skimage import data
from skimage.transform import PiecewiseAffineTransform, warp, AffineTransform
from skimage.io import imsave
from skimage.viewer import ImageViewer
from matplotlib import path as polygon
from sklearn.utils import shuffle

def estimate(self, src, dst):
	"""Extended original function from: https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py#L439
	Set the control points with which to perform the piecewise mapping.
	Return triangles generated from source and destination coordinates.
	Number of source and destination coordinates must match.
	Parameters
	----------
	src : (N, 2) array
		Source coordinates.
	dst : (N, 2) array
		Destination coordinates.
	Returns
	-------
	src_triangles : list
		List of triangles produced by Delaunay Triangulation on the source coordinates
	dst_triangles : list
		List of triangles produced by Delaunay Triangulation on the destination coordinates
	"""

	# forward piecewise affine
	# triangulate input positions into mesh
	self._tesselation = spatial.Delaunay(src)
	# find affine mapping from source positions to destination
	self.affines = []
	src_triangles = []
	dst_triangles = []

	for tri in self._tesselation.vertices:
		affine = AffineTransform()
		affine.estimate(src[tri, :], dst[tri, :])
		self.affines.append(affine)
		src_triangles.append(src[tri, :])
		dst_triangles.append(dst[tri, :])
	# inverse piecewise affine
	# triangulate input positions into mesh
	self._inverse_tesselation = spatial.Delaunay(dst)
	# find affine mapping from source positions to destination
	self.inverse_affines = []
	for tri in self._inverse_tesselation.vertices:
		affine = AffineTransform()
		affine.estimate(dst[tri, :], src[tri, :])
		self.inverse_affines.append(affine)

	return src_triangles, dst_triangles

PiecewiseAffineTransform.estimate = estimate

def shuffle_contents(path):
	"""Shuffles images associated with the same subject, and rotations associated with the same image, but leaves all images that are associated with the same subject next to one another, and all roations associated with the same image next to one another.
	Overwrites the file given as parameter with the new ordering.
	Input file should exist.
	Parameters
	----------
	path : String
		File containing the paths of succesfully processed images in arbitrary order.
	"""

	#decides whether or not pictures are part of CAVE or ELTE databases in order to be able to extract information from them the right way
	if 'dbs/cave' in path:
		db_type = 'cave'
	elif 'dbs/elte' in path:
		db_type = 'elte'


	lines = open(path,'r').read().split('\n')

	#randomizes the order of the paths
	shuffle(lines)

	filename_list = [] 
	subject_name_list = [] 
	pic_of_subject_list = []

	#extracts name of subject and number of picture per subject
	for line in lines:
		filename = line
		subject_name = filename.split('_')[0:2]
		if (db_type == 'elte'):
			pic_of_subject = filename.split('_')[2]
		elif (db_type == 'cave'):
			pic_of_subject = filename.split('_')[2:5]
		filename_list.append(filename)
		if subject_name not in subject_name_list:
			subject_name_list.append(subject_name)
		if pic_of_subject not in pic_of_subject_list:
			pic_of_subject_list.append(pic_of_subject)

	new_lines = []

	#reorders paths so that paths with same same subject but different image and paths with same same image but different rotation will be next to each other 
	for subject_name in subject_name_list:
		for pic_of_subject in pic_of_subject_list:
			for filename in filename_list:
				if (db_type == 'elte'):
					if subject_name == filename.split('_')[0:2] and pic_of_subject == filename.split('_')[2]:
						new_lines.append(filename)
				elif (db_type == 'cave'):
					if subject_name == filename.split('_')[0:2] and pic_of_subject == filename.split('_')[2:5]:
						new_lines.append(filename)

	#rewrites input file with the results
	with open(path, 'w') as f:
		f.write('\n'.join(new_lines))


def rotate_images(data_folder, rots_per_pic):
	"""Rotates images and produces the new coordinates for their facial keypoint markers.
	Creates rotated files and new markers in the same folder where the input files came from.
	Input path should exist.
	Number of rotations should be a natural number.
	Parameters
	----------
	data_folder : String
		Folder containing raw data to be processed.
	rots_per_pic : Integer
		Number of rotations to be produced per image.
	"""

	print "Rotating images..."

	#search for images in folder iteratively
	old_paths = []
	for folder, subs, files in os.walk(data_folder):
		for filename in files:
			if filename.endswith('.png') or filename.endswith('.jpg'):
				old_paths.append(os.path.join(folder, filename))
	#sorts the paths obtained
	old_paths.sort()

	old_paths_with_sums = {}

	for filename in old_paths:
		old_paths_with_sums[filename] = 0

	#counts how many times the images were already processed 
	new_paths = []
	all_files_sum = 0
	already_processed_sum = 0
	for filename in old_paths:
		if "processed" not in filename:
			all_files_sum = all_files_sum + 1
			new_paths.append(filename)
			print('File found:')
			print filename
		else:
			already_processed_sum = already_processed_sum + 1
			matching = [s for s in new_paths if ((filename.partition("_processed_")[0]+".png")==s or (filename.partition("_processed_")[0]+".jpg")==s)]
			for i in matching:
				old_paths_with_sums[i] = old_paths_with_sums[i] + 1
				if old_paths_with_sums[i] >= rots_per_pic:
					new_paths.remove(i)
					print('File already processed '+str(old_paths_with_sums[i])+' time(s):')
					print(i)
				else:
					print('File processed '+str(old_paths_with_sums[i])+' time(s):')
					print(i)

	processed_sum = 0
	too_big_angles_sum = 0
	no_desc_found_sum = 0
	markers_out_of_mesh = 0

	for current_path in new_paths:
		#rotates image as many times as needed to achieve the desired number of rotations
		for i in range(int(rots_per_pic) - old_paths_with_sums[current_path]):
			path = current_path
			
			#loads files generated by Zface if they exist and are not empty
			if (os.path.isfile(path+'.mesh3D') and
				os.path.isfile(path+'.mesh2D') and
				os.path.isfile(path+'.ctrl2D') and
				os.path.isfile(path+'.pars') and
				os.stat(path+'.mesh3D').st_size != 0 and
				os.stat(path+'.mesh2D').st_size != 0 and
				os.stat(path+'.ctrl2D').st_size != 0 and
				os.stat(path+'.pars').st_size != 0):
				src3 = np.loadtxt(path+'.mesh3D')
				src2 = np.loadtxt(path+'.mesh2D')
				ctrl2 = np.loadtxt(path+'.ctrl2D')
				scale = np.loadtxt(path+'.pars')[0]
				translx = np.loadtxt(path+'.pars')[1]
				transly = np.loadtxt(path+'.pars')[2]
				pitch = np.loadtxt(path+'.pars')[3]
				yaw = np.loadtxt(path+'.pars')[4]
				roll = np.loadtxt(path+'.pars')[5]

				#tests wether or not initial rotation is too large
				if (abs(yaw)<radians(30) and abs(pitch)<radians(15)):

					image = data.load(path)
					rows, cols = image.shape[0], image.shape[1]

					x = src3[:,0]
					y = src3[:,1]
					z = src3[:,2]

					#transform 3D mesh from normalized space and rotation to actual space and rotation
					x = x*cos(roll)+y*-sin(roll)
					y = x*sin(roll)+y*cos(roll)
					z = z

					x = x*cos(yaw)+z*sin(yaw)
					y = y
					z = x*-sin(yaw)+z*cos(yaw)

					x = x
					y = y*cos(pitch)+z*-sin(pitch)
					z = y*sin(pitch)+z*cos(pitch)

					x = x*scale+translx
					y = y*scale+transly

					#ortographically projects the 3D mesh to 2D (this will be our source for the Piecewise Affine Transform)
					src_cols = x
					src_rows = y

					src_rows, src_cols = np.meshgrid(src_rows, src_cols, sparse=True)
					src = np.dstack([src_cols.flat, src_rows.flat])[0]

					#transforms it back to normalized space
					x = (x-translx)/scale
					y = (y-transly)/scale

					#rotates it back to 0 rotation
					yaw = -yaw
					pitch = -pitch
					roll = -roll

					#adds random rotation
					real_yaw = radians(random.uniform(-30, 30))
					real_pitch = radians(random.uniform(-15, 15))
					real_roll = 0

					yaw = yaw + real_yaw
					pitch = pitch + real_pitch
					roll = roll + real_roll

					x = x*cos(roll)+y*-sin(roll)
					y = x*sin(roll)+y*cos(roll)
					z = z

					x = x*cos(yaw)+z*sin(yaw)
					y = y
					z = x*-sin(yaw)+z*cos(yaw)

					x = x
					y = y*cos(pitch)+z*-sin(pitch)
					z = y*sin(pitch)+z*cos(pitch)

					#transforms it back to real space
					x = x*scale+translx
					y = y*scale+transly

					#orthographic projection of new coordinates will be the destination for PiecewiseAffineTransform
					dst_cols = x
					dst_rows = y
					dst = np.vstack([dst_cols, dst_rows]).T

					out_rows = rows
					out_cols = cols

					#looks for triangles formed by Delaunay triangularion, extracts the ones associated with each facial keypoint marker
					tform = PiecewiseAffineTransform()
					src_triangles, dst_triangles = tform.estimate(src[:,0:2], dst)
					ctrl2_transforms = []
					for current_ctrl2 in ctrl2:
						for i in range(len(src_triangles)):
							triangle = polygon.Path(src_triangles[i])
							if triangle.contains_point(current_ctrl2):
								ctrl2_transforms.append(tform.affines[i])
								break
					if len(ctrl2_transforms)!=49:
						markers_out_of_mesh = markers_out_of_mesh + 1
						print "didn't process image, because can't find all shape parameters:"
						print path
						continue
					out_ctrl2 = []
					for i in range(len(ctrl2_transforms)):
							#performs transformation on marker
							out_ctrl2.append(ctrl2_transforms[i](ctrl2[i]))
					out_ctrl2 = np.transpose((np.transpose(out_ctrl2)[0],np.transpose(out_ctrl2)[1]))
					out_ctrl2 = np.squeeze(out_ctrl2)

					#transforms image to the new surface triangle by triangle using Delaunay triangulation, then interpolation to smooth it out
					tform = PiecewiseAffineTransform()
					tform.estimate(dst, src[:,0:2])
					out_image = warp(image, tform, output_shape=(out_rows, out_cols))

					out_path = path[:-4]+'_processed'+'_yaw_'+str(real_yaw)+'_pitch_'+str(real_pitch)+'_roll_'+str(real_roll)+path[-4:]

					#saves image and marker points
					imsave(out_path, out_image)

					np.savetxt(out_path+'_0.txt', out_ctrl2)

					processed_sum = processed_sum + 1
					print(str(processed_sum)+'. file processed:')
					print(path)
				else:
					too_big_angles_sum = too_big_angles_sum + 1
					print("didn't process image, because of too big original rotation:")
					print(path)
			else:
				no_desc_found_sum = no_desc_found_sum + 1
				print("didn't process image, beacuse descriptor documents not found:")
				print(path)

	out_paths = []
	for folder, subs, files in os.walk(data_folder):
		for filename in files:
			if filename.endswith('.png') or filename.endswith('.jpg'):
				if "processed" in filename:
					out_path = os.path.join(folder, filename).replace(data_folder, "")
					out_paths.append(out_path)

	#writes paths of generated images into contents
	filename = data_folder+'/contents'

	with open(filename, 'w') as f:
		f.write('\n'.join(out_paths))

	print "Shuffling contents..."
	#shuffles contents
	shuffle_contents(filename)


	#prints some statistics about the process on the screen
	print
	print("Statistics:")
	print("-----------")
	print("Files found: "+str(all_files_sum))
	if all_files_sum != 0:
		print("Already processed: "+str(already_processed_sum))
		print("Got processed now: "+str(processed_sum))
		print("All processed: "+str((processed_sum+already_processed_sum)*100/all_files_sum)+"%")
		print("Can't be processed because of too big angles: "+str(too_big_angles_sum*100/all_files_sum)+"%")
		print("Can't be processed because of no decriptors: "+str(no_desc_found_sum*100/all_files_sum)+"%")
		print("Can't be processed because of markers outside of mesh: "+str(markers_out_of_mesh*100/all_files_sum)+"%")
