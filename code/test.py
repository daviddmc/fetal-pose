import numpy as np
import os
from math import ceil
from scipy.ndimage import zoom
import scipy.io as sio

from skimage.feature import peak_local_max
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation, Mplp
from pgmpy.factors.discrete import DiscreteFactor
import pandas as pd
from math import gamma

from data import data_train, data_test

kp = ['ankle_l', 'ankle_r', 'knee_l', 'knee_r', 'bladder', 'elbow_l', 'elbow_r',
 'eye_l', 'eye_r', 'hip_l', 'hip_r', 'shoulder_l', 'shoulder_r', 'wrist_l', 'wrist_r']

#bone = np.array([[9,10], [11, 12], [7, 8], [11, 5], [12, 6], [5, 13], [6, 14], [9, 2], [10, 3], [2, 0], [3, 1], [4, 9], [4, 10] ])
bone = np.array([[9,10], [11, 12], [7, 8], [11, 5], [12, 6], [5, 13], [6, 14], [9, 2], [10, 3], [2, 0], [3, 1], [4, 10], [7, 11], [4, 11]])
num_a = 3 #7
num_b = 3
num_peak = [num_a, num_a, num_a, num_a, num_b, num_a, num_a, num_b, num_b, num_a, num_a, num_b, num_b, num_a, num_a]

name_dict = {}


def reset_dict(): 
    global name_dict 
    name_dict = {}


def test_result(outputs, joint_coord, s, dn, opts):

    outputs = np.squeeze(outputs)
    outputs = outputs[:, :, :, :opts.nJoint]
    if opts.test_arg[1] is not None:
        flip_order = [1, 0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
        outputs = np.flip(outputs, opts.test_arg[1])
        outputs = outputs[:, :, :, flip_order]
    if opts.test_arg[0] is not None:
        outputs = np.rot90(outputs, -1, opts.test_arg[0])
    joint_coord = np.squeeze(joint_coord)
    s = np.squeeze(s)
    dn = np.squeeze(dn)

    predict_mean_coord = np.zeros_like(joint_coord)* 0.0
    
    pad_width = [(int((ceil(ss/8.0)*8-ss)/2), int(ceil((ceil(ss/8.0)*8-ss)/2))) for ss in s]
    #pad_width = [(int((ceil(ss/16.0)*16-ss)/2), int(ceil((ceil(ss/16.0)*16-ss)/2))) for ss in s]
    sss = outputs.shape
    volume = outputs[pad_width[0][0]:sss[0]-pad_width[0][1], pad_width[1][0]:sss[1]-pad_width[1][1], pad_width[2][0]:sss[2]-pad_width[2][1]]
    
    volume[volume < 0] = 0
    #xv, yv, zv = np.meshgrid(np.arange(1, s[1]+1), np.arange(1, s[0]+1), np.arange(1, s[2]+1))
    
    if opts.use_MRF and len(mu) == 0:
        init_GA()
        
    if opts.use_MRF:
        inds = MRF(volume, joint_coord.shape[-1], GA_dict[data_test[dn]])
    
    for i in range(joint_coord.shape[-1]):
        if joint_coord[2, i] <= 0:
            joint_coord[:, i] = np.nan
            predict_mean_coord[:, i] = np.nan
        else:
        
            if opts.use_MRF:
                ind = inds[i]
            else:
                ind = np.unravel_index(np.argmax(volume[:,:,:,i]), s)
                
            weights = 0
            x_p = y_p = z_p = 0
            for x in range(ind[1]-1,ind[1]+2):
                for y in range(ind[0]-1,ind[0]+2):
                    for z in range(ind[2]-1,ind[2]+2):
                        if 0 <= x < volume.shape[1] and 0 <= y < volume.shape[0] and 0 <= z < volume.shape[2]:
                            weights += volume[y, x, z, i]
                            x_p += x * volume[y, x, z, i]
                            y_p += y * volume[y, x, z, i]
                            z_p += z * volume[y, x, z, i]
            #weights = np.exp( (-1.0/8.0)*((xv - ind[1]- 1)**2 + (yv - ind[0] - 1)**2 + (zv - ind[2] - 1)**2) ) * volume[:,:,:,i]
            predict_mean_coord[0, i] = x_p / weights + 1#np.average(xv, weights = weights)
            predict_mean_coord[1, i] = y_p / weights + 1#np.average(yv, weights = weights)
            predict_mean_coord[2, i] = z_p / weights + 1#np.average(zv, weights = weights)

    return np.hstack((dn, predict_mean_coord.ravel(), joint_coord.ravel()))


def save_test_result(res, opts):
    if res:
        headers = ['data_id']
        for prefix in ['predict_', 'label_']:
            for suffix in ['_x', '_y', '_z']:
                for j in opts.joint:
                    headers.append(prefix + str(j+1) + suffix)
        np.savetxt(os.path.join(opts.output_path, opts.name, opts.name + '.csv'), np.vstack(res), 
            fmt='%.3f', delimiter=',', header=','.join(headers), comments='')
            
            
            
mu = []
sigma = []
GA_dict = {}
beta = 2

def init_GA():

    info_ga = pd.read_excel('../info_saved.xlsx')[['fnames', 'W', 'D']]
    for index, row in info_ga.iterrows():
        GA_dict[str(row['fnames']).zfill(6) if type(row['fnames']) is int else row['fnames']] = (row['W'] if row['W'] else 32) + row['D'] / 7.0
    res = []
    for ds in data_train:
        ga = GA_dict[ds]
        joint_coord = sio.loadmat(os.path.join('../label', ds + '.mat'))['joint_coord']
        bone_len = joint_coord[:, :, bone[:, 0]] - joint_coord[:, :, bone[:, 1]];
        bone_len = np.sqrt(np.sum(bone_len**2, 1));
        res.append(bone_len[:, :] * L_GA(32) / L_GA(ga));
    res =np.concatenate(res)
    for i in range(bone.shape[0]):
        tmp = res[:, i];
        tmp = tmp[tmp < 51];
        mu.append(np.mean(tmp));
        sigma.append(np.sqrt(np.var(tmp) * gamma(1/beta) / gamma(3/beta)));
         
def L_GA(ga):
    return ga * 0.67924 + 0.86298

def MRF(volume, J, ga):
    
    MM = MarkovModel()
    MM.add_nodes_from(kp)
    locs = []
    locs_value = []
    for i in range(J):
        locs.append(peak_local_max(volume[:,:,:,i], min_distance=3, exclude_border=True, indices=True, num_peaks=num_peak[i]))
        locs_value.append(volume[locs[i][:, 0], locs[i][:, 1], locs[i][:, 2], i]- np.amin(volume[:,:,:,i])) 
        factor = DiscreteFactor([kp[i]], cardinality=[locs[i].shape[0]], values=locs_value[i])
        MM.add_factors(factor)
    fac = L_GA(32) / L_GA(ga)
    for i, b in enumerate(bone):
        MM.add_edge(kp[b[0]], kp[b[1]])
        dr = np.sqrt(np.sum((locs[b[0]].reshape((-1, 1, 3)) - locs[b[1]].reshape((1, -1, 3))) ** 2, 2))
        edgePot = np.exp(- ((dr * fac - mu[i]) / sigma[i]) ** beta);
        edgePot[dr > 40] = 0
        factor = DiscreteFactor([kp[b[0]], kp[b[1]]], cardinality=[locs[b[0]].shape[0], locs[b[1]].shape[0]], values=edgePot)
        MM.add_factors(factor)
    # inference
    #assert MM.check_model()
    inference_method = BeliefPropagation(MM)
    #inference_method = Mplp(MM)
    config_max = inference_method.map_query()
    
    inds = []
    for i in range(J):
        inds.append(locs[i][config_max[kp[i]], :])
        #inds.append(locs[i][np.nonzero(locs_value[i] == config_max[kp[i]])[0][0], :])
    return inds