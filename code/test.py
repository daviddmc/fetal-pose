import numpy as np
import os
from math import ceil
from scipy.ndimage import zoom
import scipy.io as sio

name_dict = {}

def reset_dict(): 
    global name_dict 
    name_dict = {}

def save_volume(volume, dn, opts):
    #if dn == 22618:
    name_dict[dn] = name_dict.get(dn, 0) + 1
    p = os.path.join(opts.output_path, opts.name, str(dn) + '_%03d.mat' % name_dict[dn])
    if os.path.isfile(p):
        volume += sio.loadmat(p)['heatmap']
    sio.savemat(p, {'heatmap': volume})
    #np.save(p, volume)
    return volume


def test_result(outputs, joint_coord, s, dn, opts):

    outputs = np.squeeze(outputs)
    if outputs.shape[-1] > opts.nJoint:
        #[[11,5],[12,6],[5,13],[6,14],[9,2],[10,3],[2,0],[3,1]]
        bones = outputs[:, :, :, opts.nJoint:]
        outputs = outputs[:, :, :, :opts.nJoint]
        for i, b in enumerate(opts.bone):
            outputs[:, :, :, b[0]] = outputs[:, :, :, b[0]] * bones[:, :, :, i]
            outputs[:, :, :, b[1]] = outputs[:, :, :, b[1]] * bones[:, :, :, i]
    else:
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
    
    #print('dn')
    if opts.fac > 1:
        outputs = zoom(outputs, (opts.fac, opts.fac, opts.fac, 1), order=1)
    pad_width = [(int((ceil(ss/8.0)*8-ss)/2), int(ceil((ceil(ss/8.0)*8-ss)/2))) for ss in s]
    #pad_width = [(int((ceil(ss/16.0)*16-ss)/2), int(ceil((ceil(ss/16.0)*16-ss)/2))) for ss in s]
    sss = outputs.shape
    volume = outputs[pad_width[0][0]:sss[0]-pad_width[0][1], pad_width[1][0]:sss[1]-pad_width[1][1], pad_width[2][0]:sss[2]-pad_width[2][1]]
    '''save volume'''
    #volume = save_volume(volume, int(dn), opts)
    
    volume[volume < 0] = 0
    #xv, yv, zv = np.meshgrid(np.arange(1, s[1]+1), np.arange(1, s[0]+1), np.arange(1, s[2]+1))
    
    for i in range(joint_coord.shape[-1]):
        if joint_coord[2, i] <= 0:
            joint_coord[:, i] = np.nan
            predict_mean_coord[:, i] = np.nan
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
    headers = ['data_id']
    for prefix in ['predict_', 'label_']:
        for suffix in ['_x', '_y', '_z']:
            for j in opts.joint:
                headers.append(prefix + str(j+1) + suffix)
    np.savetxt(os.path.join(opts.output_path, opts.name, opts.name + '.csv'), np.vstack(res), 
        fmt='%.3f', delimiter=',', header=','.join(headers), comments='')
        
        
def first_heatmap_p(joint_coord, shape, opts):
    shape = np.squeeze(shape)
    y_range = np.reshape(np.arange(1, 1+shape[0], dtype=np.float32), (-1,1,1,1))
    x_range = np.reshape(np.arange(1, 1+shape[1], dtype=np.float32), (1,-1,1,1))
    z_range = np.reshape(np.arange(1, 1+shape[2], dtype=np.float32), (1,1,-1,1))
    x_label, y_label, z_label = np.reshape(joint_coord, (3,1,1,1,-1))
    heatmap = opts.mag*np.exp((-1.0/2.0/opts.sigma**2)*((x_range - x_label)**2 + (y_range - y_label)**2 + (z_range - z_label)**2), dtype=np.float32)
    heatmap = np.expand_dims(heatmap, 0)
    pad_width = [(int((ceil(s/8.0)*8-s)/2), int(ceil((ceil(s/8.0)*8-s)/2))) for s in shape]
    heatmap = np.pad(heatmap, [(0,0)] + pad_width + [(0,0)],  'constant') #'constant', 'edge'
    return heatmap
    
    
def get_heatmap_p(heatmap, opts):
    heatmap = np.squeeze(heatmap)
    s = heatmap[:,:,:,0].shape
    xv, yv, zv = np.meshgrid(np.arange(1, 1+s[1]), np.arange(1, 1+s[0]), np.arange(1, 1+s[2]))

    for i in range(heatmap.shape[-1]):
        ind = np.unravel_index(np.argmax(heatmap[:,:,:,i]), s)
        heatmap[:,:,:,i] = opts.mag * np.exp((-1.0/2/opts.sigma**2)*((xv - ind[1] - 1)**2 + (yv - ind[0] - 1)**2 + (zv - ind[2] - 1)**2),dtype=np.float32)

    return np.expand_dims(heatmap, 0)
    
