import os
import numpy as np
import cv2

from config import cfg

def get_data_list(image_set_path, image_path):
    with open(os.path.join(image_set_path, 'test.txt')) as f:
        return [os.path.join(image_path ,line.rstrip('\n')+'.jpg') for line in f]

def get_image(image_path):
    return cv2.imread(image_path)
    
    
# 对像VOC这样的数据文件夹结构适用
class Data():
    def __init__(self):
        self._data_path = cfg.DATA_PATH
        self._data_image_set_path = cfg.IMAGE_SET_PATH
        self._data_JPEG_image_path = cfg.JPEG_IMAGE_PATH
        # self._data_annotation_path = ANNOTATION_PATH
        self._data_list = get_data_list(self._data_image_set_path, self._data_JPEG_image_path)
        self._batch_count = 0
        
    def __call__(self, batch_size=cfg.TRAIN.BATCH_SIZE):
        batch_number = len(self._data_list)/batch_size
        if self._batch_count < batch_number-1:
            self._batch_count += 1
        else:
            self._batch_count = 0

        path_list = self._data_list[self._batch_count*batch_size:(self._batch_count+1)*batch_size]
        batch = [get_image(path).astype(np.float32) for path in path_list]
        return np.array(batch)


if __name__ == '__main__':
    data = Data()
    print(data().shape)
    print('\n')
    print(data()[0].shape)