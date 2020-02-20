
# coding: utf-8

# In[1]:


import os
import json



import skimage
import cv2
import us_image_utils
import os,json
import random
import numpy as np
from skimage.draw import rectangle
from utils.boxes import assign_prior_boxes
from utils.boxes import create_prior_boxes, to_point_form

class DataGenerator:
    def __init__(self,batch_size):
        self.annotation_dict = self.load_annotation()
        self.batch_size = batch_size
        self.train_list = self.get_list('train_list.txt')
        self.val_list = self.get_list('val_list.txt')
        self.test_list = self.get_list('test_list.txt')
        self.IMAGE_MIN_DIM=200
        self.IMAGE_MAX_DIM=300
        
        #         prior box
        configuration = {'feature_map_sizes': [38, 19, 10, 5, 3, 1],
                         'image_size': self.IMAGE_MAX_DIM,
                         'steps': [8, 16, 32, 64, 100, 300],
                         'min_sizes': [30, 60, 111, 162, 213, 264],
                         'max_sizes': [60, 111, 162, 213, 264, 315],
                         'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                         'variance': [0.1, 0.2]}
        self.prior_boxes = to_point_form(create_prior_boxes(configuration))
    def next_batch(self,mode):
        assert mode in ['train','val','test']
        b = 0
        image_index = -1
        while True:
            try:
                if mode == 'train':
                    image_index = random.randint(0,len(self.train_list)-1)
                    img_path = self.train_list[image_index]
                elif mode=='val':
                    image_index = random.randint(0,len(self.val_list)-1)
                    img_path = self.val_list[image_index]
                else:
                    image_index = (image_index + 1)
                    img_path = self.test_list[image_index]
                
                path = './data/'+img_path
                image,bbox  = self.load_image_bbox(path)
                
                if bbox is None:
                    print('error bbox')
                    continue
                if b == 0:
                    batch_images = []
                    batch_bbox = []
                    
                batch_images.append(image.astype(np.float32))
                batch_bbox.append(bbox)
                b = b+1
                if b >= self.batch_size:
                    batch_images = np.array(batch_images)
                    batch_bbox=np.array(batch_bbox)
                    yield batch_images, batch_bbox
                    
                    b=0
            except(GeneratorExit, KeyboardInterrupt):
                # print(GeneratorExit)
                raise

    def load_annotation(self):
        os.listdir('./')
        annotation_floder ='./annotation'
        annotation_dict = {}
        for annotation in os.listdir(annotation_floder):
            path = os.path.join(annotation_floder,annotation)
            with open(path,'r',encoding='utf-8') as f:
                temp_dict = json.load(f)
            annotation_dict.update(temp_dict)
        return annotation_dict
    def get_list(self,list_file):
        with open(list_file,'r',encoding='utf-8') as f:
            l = f.readlines()
        l = [ll.strip() for ll in l]
        return l
    def load_image_bbox(self,path):
        image = skimage.io.imread(path)
        # # 归一化、
        image = image.astype(np.float64)
        mean = np.mean(image)
        image = image / np.max(image)

        
#         bouning box
        img_name = path.split('/')[-1]
        key =[key for key in  self.annotation_dict.keys() if self.annotation_dict[key]['filename']==img_name][0]
        region = self.annotation_dict[key]['regions']['0']['shape_attributes']
        x=region['x']
        y = region['y']
        w = region['width']
        h = region['height']

#         mask
        rect_start = (y,x)
        rect_end = (y+h,x+w)
        mask = np.zeros(image.shape[:2],dtype=np.uint8)
        rr, cc = rectangle(rect_start, end=rect_end, shape=mask.shape)
        mask[rr, cc] = 1
        mask = np.expand_dims(mask,axis=2)
        image, window, scale, padding, crop = us_image_utils.resize_image(
                                                            image,
                                                            min_dim=self.IMAGE_MIN_DIM,
                                                            min_scale=0,
                                                            max_dim=self.IMAGE_MAX_DIM,
                                                            )
        mask = us_image_utils.resize_mask(mask, scale, padding, crop)

#         mask 2 bouding box
        bbox = us_image_utils.extract_bboxes(mask)
        bbox = bbox/mask.shape[0]
        if (bbox==np.array([0,0,0,0])).all():
            return image,None
        assert bbox.shape[0] == 1
        ymin,xmin,ymax,xmax = bbox[0]

        box_data = assign_prior_boxes(self.prior_boxes, np.array([[xmin, ymin, xmax,ymax,0,1]]), 2, [.1, .1, .2, .2])
        
        return image,box_data

from SSD300 import SSD300
from utils.training import MultiboxLoss, scheduler
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler,TensorBoard
from keras.optimizers import SGD,Adam
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
#设置gpu动态使用，不会沾满全部内存
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

num_classes =2
alpha_loss = 1.0
learning_rate = 1e-5
momentum = .9
weight_decay = 5e-4
gamma_decay = 0.1
negative_positive_ratio = 3
model_name='SSD10k'


model = SSD300(input_shape=(300, 300, 3), num_classes=num_classes,num_priors=[4,6,6,6,4,4], weights_path=None,return_base=False)

multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, alpha_loss)
optimizer = Adam()
model.compile(optimizer, loss=multibox_loss.compute_loss)

model_path = './trained_models/' + model_name + '/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True,save_best_only=True,monitor='val_loss')
tb =TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
callbacks = [checkpoint, tb]


train_g = DataGenerator(batch_size=4)
val_g =  DataGenerator(batch_size=1)


history = model.fit_generator(train_g.next_batch(mode ='train'),
                    steps_per_epoch=150,
                    epochs=250,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data = val_g.next_batch(mode = 'val'),
                    validation_steps=113,
                    use_multiprocessing=False,
                    workers=1)



import matplotlib.pyplot as plt


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")

plt.savefig('loss.jpg')

