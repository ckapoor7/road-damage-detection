import os 
import random
import numpy as np
import cv2 
import scipy.io as io
import torch
from torch.utils.data import Dataset

import utils

class RDD_Dataset(Dataset):
	def __init__(self, opt):				# root: list; transform: torch transform
		self.opt = opt
		# root path
		self.baseroot = os.path.join(opt.baseroot, '')
		namelist = self.get_names(self.baseroot)
		#print(namelist)
		# build image list
		self.imglist_A, self.imglist_B = self.build_imglist(namelist, opt)
		#print(self.imglist_A)
		#print(self.imglist_B)
 	
	def get_names(self, path):
		# read a folder and return the image name
		ret = os.listdir(path)
		for i in range(len(ret)):
			file_name = ret[i]
			ret[i] = file_name[:-4]		# remove file extension
		return ret
	
	def build_imglist(self, namelist, opt):
		# build an image list
		imglist_A = []
		imglist_B = []
		for i in range(len(namelist)):
			imglist_A.append(namelist[i] + '.mat')
			imglist_B.append(namelist[i] + '.png')
		return imglist_A, imglist_B
	"""
	def __getitem__(self, index):
		# read an image
		imgpath_A = os.path.join(self.baseroot, self.imglist_A[index])
		#img_A = io.loadmat(imgpath_A)['cube']		# (482, 512, 31), in range [0, 1], float64
		#print(img_A)
		imgpath_B = os.path.join(self.baseroot, self.imglist_B[index])
		img_B =	cv2.imread(imgpath_B, -1)		# (482, 512, 3), in range [0, 255], uint8
		
		# normalization
		img_B = img_B.astype(np.float64) / 255.0

		# crop
		if self.opt.crop_size > 0:
			h, w = img_A.shape[:2]
			rand_h = random.randint(0, h - self.opt.crop_size)
			rand_w = random.randint(0, w - self.opt.crop.size)
			img_A = img_A[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
			img_B = img_B[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]

		# to tensor
		img_A = torch.from_numpy(img_A.astype(np.float32).transpose(2, 0, 1)).contiguous()
		img_B = torch.from_numpy(img_B.astype(np.float32).transpose(2, 0, 1)).contiguous()

		return img_B, img_A
"""	

	def __len__(self):
		return len(self.imglist_A)
