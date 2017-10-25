from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model

import pandas as pd

def first_stage_output(img_paths, num_rois, keras_frcnn_path,
                       config_path, input_weight_path):

    cwd = os.getcwd()
    with open(config_path, 'rb') as f_in:
        os.chdir(keras_frcnn_path) # need to be in the keras-frcnn folder
        C = pickle.load(f_in)

    from keras_frcnn import config
    from keras_frcnn import roi_helpers
    if C.network == 'vgg':
        from keras_frcnn import vgg as nn
    elif C.network == 'vgg_lite':
        from keras_frcnn import vgg_lite as nn
    else:
        import keras_frcnn.resnet as nn
    os.chdir(cwd) # change back to original directory
    
    C.model_path = input_weight_path
        
    # turn off any data augmentation
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    
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
        #img = img[:, :, (2, 1, 0)] # (see here: https://github.com/yhenon/keras-frcnn/issues/148)
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(ratio, x1, y1, x2, y2):
        real_x1 = int(round(x1 / ratio))
        real_y1 = int(round(y1 / ratio))
        real_x2 = int(round(x2 / ratio))
        real_y2 = int(round(y2 / ratio))
        return (real_x1, real_y1, real_x2 ,real_y2)
    
    class_mapping = C.class_mapping
    
    class_mapping = {v: k for k, v in class_mapping.items()}
    C.num_rois = int(num_rois)
    
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
    
    results = {}
    
    for img_path in img_paths:
        img, ratio = format_img_size(cv2.imread(img_path), C)
        
        X = format_img_channels(img, C)
        
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
        
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
        
        
        row, col = np.unravel_index(np.argmax(P_cls[0][:,:-1]), P_cls[0][:,:-1].shape)
        highest_prob = np.max(P_cls[0][:,:-1])    
        cls_num = col
        
        (x, y, w, h) = ROIs[0, row, :]

        (tx, ty, tw, th) = P_regr[0, row, 4*cls_num:4*(cls_num+1)]
        tx /= C.classifier_regr_std[0]
        ty /= C.classifier_regr_std[1]
        tw /= C.classifier_regr_std[2]
        th /= C.classifier_regr_std[3]
        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
        
        box_coords = get_real_coordinates(ratio, C.rpn_stride*max(x,0), C.rpn_stride*max(y,0), 
                                          C.rpn_stride*(x+w), C.rpn_stride*(y+h))
        
        results[img_path] = [box_coords, highest_prob]
    
    return results


