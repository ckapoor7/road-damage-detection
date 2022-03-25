import torch
import scipy.io as io
import numpy as np
import cv2

img = cv2.imread('/home/pratik_narang/Chaitanya/HSI/1586853722_kurt-cotoaga-cqblg3lzepk-unsplash-modded.jpg', -1)
print(img.shape)

cv2.imshow('1', img)
cv2.waitKey(5000)
