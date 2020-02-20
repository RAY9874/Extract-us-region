#detection
from utils.inference import detect
from utils.inference import plot_detections
from SSD300 import SSD300
from utils.training import MultiboxLoss, scheduler
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler,TensorBoard
from keras.optimizers import SGD
import os
import skimage
import cv2
import us_image_utils
import os,json
import random
import numpy as np
from skimage.draw import rectangle
from utils.boxes import assign_prior_boxes
from utils.boxes import create_prior_boxes, to_point_form
from utils.boxes import denormalize_box
import matplotlib.pyplot as plt
import copy
import keras.backend as K
#屏蔽tensorflow 打印日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def extract_us_region(test_img_path,model):
   
    IMAGE_MIN_DIM =300
    IMAGE_MAX_DIM = 300

    image = cv2.imread(test_img_path)
    ori_image =copy.deepcopy(image)
    # # 归一化、
    image = image.astype(np.float64)
    mean = np.mean(image)
    image = image / np.max(image)
    image, window, scale, padding, crop = us_image_utils.resize_image(
                                                        image,
                                                        min_dim=IMAGE_MIN_DIM,
                                                        min_scale=0,
                                                        max_dim=IMAGE_MAX_DIM,
                                                        )
    if crop is not None:
        print('CAUTION : crop is applied to image')

    image = np.expand_dims(image, 0)
    predictions = model.predict(image)

    # print(predictions)
    prior_boxes = to_point_form(create_prior_boxes())
    detections = detect(predictions, prior_boxes,conf_thresh=0.9, nms_thresh=0,top_k=2)
    confidence,x1,x2,y1,y2 = detections[0][1][0]
    detections = denormalize_box([x1,x2,y1,y2], (300,300))
  
    #未还原为原图时的裁剪
    # x_min, y_min, x_max, y_max = detections
    # us_region = image[0,y_min:y_max, x_min:x_max,:]
    # us_region = us_region*255


    #还原为原图 再裁剪
    x_min, y_min, x_max, y_max = detections
    (top, bottom) = padding[0]
    (left, right) = padding[1]
    x_min, y_min, x_max, y_max = x_min-left ,y_min-top,x_max-left,y_max-top
    x_min, y_min, x_max, y_max = int(x_min/scale), int(y_min/scale), int(x_max/scale), int(y_max/scale)
    us_region = ori_image[y_min:y_max, x_min:x_max,:]
#     del prior_boxes
#     del detections
#     K.clear_session()
      
    return us_region,confidence
if __name__ == '__main__':
	# 使用说明
	# 1. 拷贝US_SSD至本地
	# 2. 读取权重
	# 3. 设置类别为2（背景+超声区）
	# 4. 调用extract_us_region
	####
	####
	# tips
	# 1.一般情况下，预测概率>0.99 （confidence值就是预测概率）
	# 
	# 
	####
	####
	#样例代码如下，假设路径为---./
	#							---data
	#							---US_SSD
	#							---yourcode.py
	#则按照如下方式import
	import sys
	sys.path.append('./US_SSD')
	from US_SSD.detect import extract_us_region,extract_us_region_new
	from US_SSD.SSD300 import SSD300
    
    weights_path = './trained_models/SSD10k/weights.15-0.72.hdf5'
    num_classes =2
    model = SSD300(weights_path=weights_path,num_classes=num_classes)
    test_img_path = './sample_data/sample_ori_data/'#自己定义的一些测试样例，可以试一下
    
    for path in os.listdir(test_img_path):
        image_path = test_img_path +path
        us_region,confidence = extract_us_region(image_path,model)
        
        print(us_region.shape,confidence)