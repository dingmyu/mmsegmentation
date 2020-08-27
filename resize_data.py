import os
import glob
import cv2


label_list = glob.glob('data/*/*/*/*/*_labelTrainIds.png')
file_list = glob.glob('data/*/*/*/*/*_leftImg8bit.png')

for index, item in enumerate(file_list):
    img = cv2.resize(cv2.imread(item), (512, 256), interpolation=cv2.INTER_LINEAR)
    result = cv2.imwrite(item, img)
    if index % 100 == 0:
        print('img', index, img.shape)

for index, item in enumerate(label_list):
    img = cv2.resize(cv2.imread(item, flags=0), (512, 256), interpolation=cv2.INTER_NEAREST)
    result = cv2.imwrite(item, img)
    if index % 100 == 0:
        print('label', index, img.shape)
