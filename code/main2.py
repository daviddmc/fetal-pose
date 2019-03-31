from options import Options
from data import read_nifti
from model import get_network
import tensorflow as tf
import os
import time
import numpy as np
import random
import scipy.io as sio
from math import ceil


np.random.seed(123)
random.seed(123)

# options
opts = Options().parse()

    
    
def test_fun(opts):
    v = read_nifti('../rawdata/022618/022618_000.nii.gz')
    joint_coord = sio.loadmat('../rawdata/022618/022618.mat')['joint_coord'].reshape((-1, 15, 3)).transpose((0,2,1))
    
    lc, rc = 2, 3
    
    x_l, y_l, z_l = joint_coord[0, 0, lc]-1, joint_coord[0, 1, lc]-1, joint_coord[0, 2, lc]-1
    x_r, y_r, z_r = joint_coord[0, 0, rc]-1, joint_coord[0, 1, rc]-1, joint_coord[0, 2, rc]-1
    xv, yv, zv = np.meshgrid(np.arange(v.shape[1]), np.arange(v.shape[0]), np.arange(v.shape[2]))
    
    val_mask_l = np.floor((xv-x_l)**2 + (yv-y_l)**2 + (zv-z_l)**2) <= 3**2
    val_mask_r = np.floor((xv-x_r)**2 + (yv-y_r)**2 + (zv-z_r)**2) <= 3**2
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    outputs, _ = get_network(inputs, opts)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(opts.output_path, opts.name, 'model%d.ckpt'%opts.epochs))
        for r in range(1, 50):
            mask_l = np.floor((xv-x_l)**2 + (yv-y_l)**2 + (zv-z_l)**2) <= r**2
            mask_r = np.floor((xv-x_r)**2 + (yv-y_r)**2 + (zv-z_r)**2) <= r**2
            output_val = sess.run(outputs[-1], feed_dict={inputs: np.expand_dims(np.expand_dims(v * mask_r, 0), -1)})
            output_val = np.squeeze(output_val)
            print(r, np.max(output_val[:,:,:,lc] * val_mask_r), np.max(output_val[:,:,:,rc] * val_mask_r))

    print(np.sqrt(sum((joint_coord[0, :, :] - np.reshape(np.array([x_r+1, y_r+1, z_r+1]), (3,1))) ** 2, 0)))
    
    
def test_fun2(opts):
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    outputs, _ = get_network(inputs, opts)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(opts.output_path, opts.name, 'model%d.ckpt'%opts.epochs))
        for folder in [os.path.join('../newdata', f) for f in sorted(os.listdir('../newdata'))]:
            folder_basename = os.path.basename(folder)
            print(folder_basename)
            predict_filename = os.path.join('../predict', folder_basename + '.mat')
            niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            joint_coord = np.zeros((len(niinames), 3, 15))
            confidence = np.zeros((len(niinames), 15))
            for nf, nii in enumerate(niinames):
                v = read_nifti(nii)
                pad_width = [(int((ceil(s/8.0)*8-s)/2), int(ceil((ceil(s/8.0)*8-s)/2))) for s in v.shape]
                v = np.pad(v, pad_width,  'reflect')
                v = np.expand_dims(np.expand_dims(v, 0), -1)
                output_val = sess.run(outputs[-1], feed_dict={inputs: v})
                output_val = np.squeeze(output_val)
                sss = output_val.shape
                volume = output_val[pad_width[0][0]:sss[0]-pad_width[0][1], pad_width[1][0]:sss[1]-pad_width[1][1], pad_width[2][0]:sss[2]-pad_width[2][1]]
                volume[volume < 0] = 0
                s = volume[:,:,:,0].shape
                for i in range(joint_coord.shape[-1]):
                    #print(i, np.max(volume[:,:,:,i]))
                    ind = np.unravel_index(np.argmax(volume[:,:,:,i]), s)
                    confidence[nf, i] = volume[ind[0],ind[1],ind[2],i]
                    weights = 0
                    x_p = y_p = z_p = 0
                    r = 2
                    for x in range(ind[1]-r,ind[1]+r+1):
                        for y in range(ind[0]-r,ind[0]+r+1):
                            for z in range(ind[2]-r,ind[2]+r+1):
                                if 0 <= x < s[1] and 0 <= y < s[0] and 0 <= z < s[2]:
                                    weights += volume[y, x, z, i]
                                    x_p += x * volume[y, x, z, i]
                                    y_p += y * volume[y, x, z, i]
                                    z_p += z * volume[y, x, z, i]
                    if weights == 0:
                        print('weight zero')
                        joint_coord[nf, :, i] = 0
                    else:
                        joint_coord[nf, 0, i] = x_p / weights + 1
                        joint_coord[nf, 1, i] = y_p / weights + 1
                        joint_coord[nf, 2, i] = z_p / weights + 1
            sio.savemat(predict_filename, {'joint_coord': joint_coord, 'confidence': confidence})                
    print('finish')
        

if __name__ == '__main__':
    tf.reset_default_graph()
    test_fun2(opts)
