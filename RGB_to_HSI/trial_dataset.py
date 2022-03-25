import os 
import random
import numpy as np
import cv2
import scipy.io as io
import torch
from torch.utils.data import Dataset

import utils

def get_names(path):
	# read a folder and return the image name
	ret = os.listdir(path)
	for i in range (len(ret)):
		file_name = ret[i]
		ret[i] = file_name[:-4]  # without file extension

	return ret

baseroot = '/chaitanya/datasets/rdd_train'
files = get_names(baseroot)
print(files)
