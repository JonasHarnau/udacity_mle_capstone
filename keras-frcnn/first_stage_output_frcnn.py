from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.utils.generic_utils import Progbar
from sklearn.datasets import load_files
import pandas as pd

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="path", help="Path to data. Assumes subfolders correspond to labels.")
parser.add_option("--output_path", dest="output_path", help="Path to save the output data to.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=300)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("--logs_path", dest="logs_path", help="Where logs for the losses should be saved.", default='./first_stage_logs.csv')

(options, args) = parser.parse_args()

if not options.path:   # if filename is not given
    parser.error('Error: path to data must be specified. Pass --path to command line')

if not options.output_path:   # if filename is not given
    parser.error('Error: path to output data must be specified. Pass --output_path to command line')
    
    
with open(options.config_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'vgg':
    from keras_frcnn import vgg as nn
elif C.network == 'vgg_lite':
    from keras_frcnn import vgg_lite as nn
else:
    import keras_frcnn.resnet as nn
    
# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.path
output_path = options.output_path

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
        
    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio    

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

class_mapping = C.class_mapping

class_mapping = {v: k for k, v in class_mapping.items()}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg' or C.network == 'vgg_lite':
    num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

file_info = load_files(img_path, load_content=False, shuffle=False)
file_info['img_name'] = pd.Series(file_info['filenames']).str.split('/', expand=True).iloc[:,-1].str.split('.', expand=True).iloc[:,0].values.astype('str')

logs = pd.DataFrame(None, index = file_info['img_name'], columns=['prob'])

for label in file_info['target_names']:
    try:
        os.mkdir(os.path.join(output_path, label))
    except FileExistsError:
        continue
    
progbar = Progbar(len(file_info['filenames']))

for i, img_path in enumerate(file_info['filenames']):
    if not img_path.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue

    img = cv2.imread(img_path)

    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7,
                               max_boxes=C.num_rois)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]
    
    ROIs = np.expand_dims(R, axis=0)
    if ROIs.shape[1] == 0:
        break
    
    
    [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
    
    
    row, col = np.unravel_index(np.argmax(P_cls[0][:,:-1]), P_cls[0][:,:-1].shape)
    highest_prob = np.max(P_cls[0][:,:-1])    
    cls_num = col
    
    (x, y, w, h) = ROIs[0, row, :]
#    try:
    (tx, ty, tw, th) = P_regr[0, row, 4*cls_num:4*(cls_num+1)]
    tx /= C.classifier_regr_std[0]
    ty /= C.classifier_regr_std[1]
    tw /= C.classifier_regr_std[2]
    th /= C.classifier_regr_std[3]
    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
    if highest_prob > 0.5:
        crop_img = np.round(X[0][max(C.rpn_stride*y,0):C.rpn_stride*(y+h), 
                                 max(C.rpn_stride*x,0):C.rpn_stride*(x+w)]).astype(np.uint8)
    else: # don't crop
        crop_img = X[0].astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, file_info['target_names'][file_info['target'][i]],
                             file_info['img_name'][i] + '.png'), cv2.resize(crop_img, (224, 224), 
                                                                            interpolation=cv2.INTER_CUBIC))    
#    except:
#        pass
    
    logs.loc[file_info['img_name'][i]] = highest_prob

    progbar.update(i+1)
    
print('saving logs')
logs.to_csv(options.logs_path)
    