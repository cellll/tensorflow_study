import tensorflow as tf
import time
from PIL import Image
import os
import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.contrib.image import rotate
import argparse
from datetime import datetime
import hashlib
import random
import re
import sys
import tarfile
from six.moves import urllib

import math
import random
import time


def main(args):
    
    
    label = input_data(args.input_dir)
    sess = init(args)
    
    for i in range(len(label)):
        files = os.listdir(os.path.join(args.input_dir, label[i]))
        num_files = len(files)
 
        if not num_files == 0:
            if not os.path.exists(os.path.join(args.output_dir)):
                os.mkdir(args.output_dir)
                os.mkdir(os.path.join(args.output_dir, label[i]))
            
                    
            for f in files:
                for step in range(args.step):
                    print ("{}  :  Step {}".format(os.path.join(args.input_dir, label[i], f), step))
                    distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(args.random_crop, args.random_scale, args.random_brightness)
                    result = get_random_distorted_bottlenecks(sess, os.path.join(args.input_dir, label[i], f), distorted_jpeg_data_tensor, distorted_image_tensor)
                    imsave(os.path.join(args.output_dir, label[i], f.split('.')[0]+'_'+str(step)+'.png'), np.asarray(random_rotation(sess, result)))
                    
                    
                
                    
def init(args):
    
    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=args.gpu_memory_fraction
    
    return tf.Session(config=config)
    
    
def input_data(input_dir):
    label=[]
    
    for a in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, a)):
            label.append(a)

    return label



def add_input_distortions(random_crop, random_scale, random_brightness):
    jpeg_data = tf.placeholder(tf.string, name='JPGInput')
    decode_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    decode_image_as_float = tf.cast(decode_image, dtype=tf.float32)
    decode_image_4d = tf.expand_dims(decode_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
    
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, args.input_width)
    precrop_height = tf.multiply(scale_value, args.input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decode_image_4d, precrop_shape_as_int)
    #precropped_image = tf.image.resize_bilinear(decode_image_as_float, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [args.input_height, args.input_width, args.input_depth])
    
    flip = random.randrange(0,2)
    if (flip == 1) and (args.is_flip):
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    
    brightness_min = 1.0 - (args.random_brightness / 100.0)
    brightness_max = 1.0 + (args.random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    
    offset_image = tf.subtract(brightened_image, args.input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / args.input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result

def get_random_distorted_bottlenecks(sess, imgfile, distorted_jpeg, distorted_image):
    jpeg_data = gfile.FastGFile(imgfile, 'rb').read()
    distorted_image_data= sess.run(distorted_image, {distorted_jpeg:jpeg_data})
    
    return distorted_image_data


def random_rotation(sess, img):
    img = img.squeeze()
    isrotate = random.randrange(0,2)
    if isrotate and args.is_rotate:
        get_rotated_img = rotate(img, random.randrange(1, 80))
        rotated_img = sess.run(get_rotated_img)
    else : 
        rotated_img = img
        
    return rotated_img




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--input_width',
        type=int,
        default=299,
        help='input width'
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=299,
        help='input height'
    )
    parser.add_argument(
        '--input_depth',
        type=int,
        default=3,
        help='input depth'
    )
    parser.add_argument(
        '--input_mean',
        type=int,
        default=128,
        help='input mean'
    )
    parser.add_argument(
        '--input_std',
        type=int,
        default=128,
        help='input mean'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        default='',
        help='input dir'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        default='',
        help='output dir'
    )
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default='0.5',
        help='gpu memory fraction'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default='50',
        help='random crop'
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default='50',
        help='random scale'
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default='50',
        help='random brightness'
    )
    parser.add_argument(
        '--step',
        type=int,
        default='10',
        help='how many augmentation '
    )
    parser.add_argument(
        '--is_flip',
        default=True,
        help='random flip'
    )
    parser.add_argument(
        '--is_rotate',
        default=True,
        help='random rotate'
    )
    
    
    args = parser.parse_args()
    main(args)
    
