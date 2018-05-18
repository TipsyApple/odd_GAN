import os
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 4

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# 存疑，这里是直接复制别人的，做图像均值处理，需要考证一下
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Data Path
__C.DATA_PATH = '/Users/martin/Documents/code/DataSet/VOC2007'
# Image Set Path
__C.IMAGE_SET_PATH = os.path.join(__C.DATA_PATH, 'ImageSets', 'Main')
# Annotations Path
__C.ANNOTATION_PATH = os.path.join(__C.DATA_PATH, 'Annotations')

__C.JPEG_IMAGE_PATH = os.path.join(__C.DATA_PATH, 'JPEGImages')
