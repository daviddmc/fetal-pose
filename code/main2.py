from options import Options
from data import read_nifti
from model import get_network
import tensorflow as tf
import os
import time
import numpy as np
import random
import scipy.io as sio
from math import ceil, gamma
from skimage.feature import peak_local_max
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation, Mplp
from pgmpy.factors.discrete import DiscreteFactor
import pandas as pd

np.random.seed(123)
random.seed(123)

# options
opts = Options().parse()

kp = ['ankle_l', 'ankle_r', 'knee_l', 'knee_r', 'bladder', 'elbow_l', 'elbow_r',
 'eye_l', 'eye_r', 'hip_l', 'hip_r', 'shoulder_l', 'shoulder_r', 'wrist_l', 'wrist_r']

bone = np.array([[9,10], [11, 12], [7, 8], [11, 5], [12, 6], [5, 13], [6, 14], [9, 2], [10, 3], [2, 0], [3, 1], [4, 9], [4, 10] ]);
#bone = np.array([[9,10], [11, 12], [7, 8], [11, 5], [12, 6], [5, 13], [6, 14], [9, 2], [10, 3], [2, 0], [3, 1], [4, 10], [7, 11], [4, 11]])

#num_peak = [5, 5, 5, 5, 3, 5, 5, 3, 3, 5, 5, 3, 3, 5, 5]
num_a = 1 # 7
num_b = 1 # 3
num_peak = [num_a, num_a, num_a, num_a, num_b, num_a, num_a, num_b, num_b, num_a, num_a, num_b, num_b, num_a, num_a]
use_MRF = num_a > 1 or num_b > 1

pfname = '../predict_'
pfname += opts.name
if not os.path.isdir(pfname):
    os.makedirs(pfname)
    
def trim_mean(a, proportiontocut):
    uppercut = a.shape[0] - int(proportiontocut * a.shape[0])
    atmp = np.partition(a, uppercut - 1, axis=0)
    return np.mean(atmp[:uppercut], axis=0)

#GA = {'022618.mat':31+4/7, '031615.mat':28, '031616.mat':28+5/7, '032318a.mat':36+5/7, '040218.mat':33, '040716.mat':33+6/7, '043015.mat':29+3/7, '061217.mat':32+5/7, '102617.mat':31+3/7}
    
def test_fun2(opts):
    '''
    def L_GA(ga):
        return ga * 0.67924 + 0.86298

    res = []
    for label in [os.path.join('../label', f) for f in sorted(os.listdir('../label'))]:
        ga = GA[os.path.basename(label)]
        joint_coord = sio.loadmat(label)['joint_coord'].reshape((-1, 15, 3)).transpose((0,2,1)) 
        bone_len = joint_coord[:, :, bone[:, 0]] - joint_coord[:, :, bone[:, 1]];
        bone_len = np.sqrt(np.sum(bone_len**2, 1));
        ri = np.random.randint(bone_len.shape[0], size=1000)
        res.append(bone_len[ri, :] * L_GA(32) / L_GA(ga));
    res =np.concatenate(res)
    mu = []
    sigma = []
    for i in range(bone.shape[0]):
        beta = 2; #8
        tmp = res[:, i];
        tmp = tmp[tmp < 51];
        mu.append(np.mean(tmp));
        sigma.append(np.sqrt(np.var(tmp) * gamma(1/beta) / gamma(3/beta)));
        
    GA_dict = {}
    info_ga = pd.read_excel('../info_saved.xlsx')[['fnames', 'W', 'D']]
    for index, row in info_ga.iterrows():
        GA_dict[str(row['fnames']).zfill(6) if type(row['fnames']) is int else row['fnames']] = (row['W'] if row['W'] else 32) + row['D'] / 7.0
    '''
    
    
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    outputs, _ = get_network(inputs, opts)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ep_load = max([int(''.join(filter(str.isdigit, x))) for x in os.listdir(os.path.join(opts.output_path, opts.name)) if x.endswith('.ckpt.meta')])
        print('load ep:%d' % ep_load)
        saver.restore(sess, os.path.join(opts.output_path, opts.name, 'model%d.ckpt'%ep_load))
        for folder in [os.path.join('../newdata', f) for f in sorted(os.listdir('../newdata'))]: # for each series
            folder_basename = os.path.basename(folder)
            print(folder_basename)
            #ga = GA_dict[folder_basename]
            #fac = L_GA(32) / L_GA(ga)
            predict_filename = os.path.join(pfname, folder_basename + '.mat')
            niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            joint_coord = np.zeros((len(niinames), 3, 15))
            confidence = np.zeros((len(niinames), 15))
            for nf, nii in enumerate(niinames): # for each volume
                print(nf)
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
                # build MM
                for i in range(joint_coord.shape[-1]): # for each keypoint
                    if use_MRF:
                        ind = locs[i][np.nonzero(locs_value[i] == config_max[kp[i]])[0][0], :]
                        #ind = locs[i][config_max[kp[i]], :]
                    else:
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
            #sio.savemat(predict_filename, {'joint_coord': joint_coord, 'confidence': confidence})   
            sio.savemat(predict_filename, {'joint_coord': joint_coord})               
    print('finish')
        

if __name__ == '__main__':
    tf.reset_default_graph()
    test_fun2(opts)
