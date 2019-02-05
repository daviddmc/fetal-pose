import numpy as np
import os
from math import ceil
from scipy.ndimage import zoom

name_dict = {}

def save_volume(volume, dn, opts):
    name_dict[dn] = name_dict.get(dn, 0) + 1
    np.save(os.path.join(opts.output_path, opts.name, str(dn) + '_%03d' % name_dict[dn]), volume)


def test_result(outputs, joint_coord, s, dn, opts):

    outputs = np.squeeze(outputs)[:, :, :, :opts.nJoint]
    joint_coord = np.squeeze(joint_coord)
    s = np.squeeze(s)
    dn = np.squeeze(dn)

    predict_mean_coord = np.zeros_like(joint_coord)* 0.0
    """
    if opts.fac > 1:
        volume = outputs
        pad_width = [(int((ceil(ss/16.0)*16-ss)/2), int(ceil((ceil(ss/16.0)*16-ss)/2))) for ss in s]
        sss = outputs.shape
        xv, yv, zv = np.meshgrid(np.arange(0,sss[1]), np.arange(0,sss[0]), np.arange(0,sss[2]))
        xv = xv * opts.fac - pad_width[1][0]
        yv = yv * opts.fac - pad_width[0][0]
        zv = zv * opts.fac - pad_width[2][0]
        
        for i in range(joint_coord.shape[-1]):
            if joint_coord[2, i] <= 0:
                joint_coord[:, i] = np.nan
                predict_mean_coord[:, i] = np.nan
            else:
                ind = np.unravel_index(np.argmax(volume[:,:,:,i]), s)
                weights = np.exp( (-1.0/8.0)*((xv - (ind[1] * 2 - pad_width[1][0]))**2 + (yv - (ind[0]*2 - pad_width[0][0]))**2 + (zv - (ind[2]*2-pad_width[2][0]))**2) ) * volume[:,:,:,i]
                predict_mean_coord[0, i] = np.average(xv, weights = weights)
                predict_mean_coord[1, i] = np.average(yv, weights = weights)
                predict_mean_coord[2, i] = np.average(zv, weights = weights)
    """
    
    #print('dn')
    if opts.fac > 1:
        outputs = zoom(outputs, (opts.fac, opts.fac, opts.fac, 1), order=1)
    #pad_width = [(int((ceil(ss/8.0)*8-ss)/2), int(ceil((ceil(ss/8.0)*8-ss)/2))) for ss in s]
    pad_width = [(int((ceil(ss/16.0)*16-ss)/2), int(ceil((ceil(ss/16.0)*16-ss)/2))) for ss in s]
    sss = outputs.shape
    volume = outputs[pad_width[0][0]:sss[0]-pad_width[0][1], pad_width[1][0]:sss[1]-pad_width[1][1], pad_width[2][0]:sss[2]-pad_width[2][1]]
    '''save volume'''
    #save_volume(volume, int(dn), opts)
    
    volume[volume < 0] = 0
    xv, yv, zv = np.meshgrid(np.arange(s[1]), np.arange(s[0]), np.arange(s[2]))
    
    for i in range(joint_coord.shape[-1]):
        if joint_coord[2, i] <= 0:
            joint_coord[:, i] = np.nan
            predict_mean_coord[:, i] = np.nan
        else:
            ind = np.unravel_index(np.argmax(volume[:,:,:,i]), s)
            weights = np.exp( (-1.0/8.0)*((xv - ind[1])**2 + (yv - ind[0])**2 + (zv - ind[2])**2) ) * volume[:,:,:,i]
            predict_mean_coord[0, i] = np.average(xv, weights = weights)
            predict_mean_coord[1, i] = np.average(yv, weights = weights)
            predict_mean_coord[2, i] = np.average(zv, weights = weights)

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
    y_range = np.reshape(np.arange(0, shape[0], dtype=np.float32), (-1,1,1,1))
    x_range = np.reshape(np.arange(0, shape[1], dtype=np.float32), (1,-1,1,1))
    z_range = np.reshape(np.arange(0, shape[2], dtype=np.float32), (1,1,-1,1))
    x_label, y_label, z_label = np.reshape(joint_coord, (3,1,1,1,-1))
    heatmap = opts.mag*np.exp((-1.0/2.0/opts.sigma**2)*((x_range - x_label)**2 + (y_range - y_label)**2 + (z_range - z_label)**2), dtype=np.float32)
    heatmap = np.expand_dims(heatmap, 0)
    pad_width = [(int((ceil(s/8.0)*8-s)/2), int(ceil((ceil(s/8.0)*8-s)/2))) for s in shape]
    heatmap = np.pad(heatmap, [(0,0)] + pad_width + [(0,0)],  'edge')
    return heatmap
    
    
def get_heatmap_p(heatmap, opts):
    heatmap = np.squeeze(heatmap)
    s = heatmap[:,:,:,0].shape
    xv, yv, zv = np.meshgrid(np.arange(s[1]), np.arange(s[0]), np.arange(s[2]))

    for i in range(heatmap.shape[-1]):
        ind = np.unravel_index(np.argmax(heatmap[:,:,:,i]), s)
        heatmap[:,:,:,i] = opts.mag * np.exp((-1.0/2/opts.sigma**2)*((xv - ind[1])**2 + (yv - ind[0])**2 + (zv - ind[2])**2),dtype=np.float32)

    return np.expand_dims(heatmap, 0)
    
