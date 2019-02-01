import tensorflow as tf
import scipy.io as sio
import os
import nibabel as nib
import numpy as np
from random import shuffle, randint, choice, uniform
from math import floor, ceil
from dataset import get_dataset_from_indexable

import sys

rots = [[], [(1, (0,1))], [(1, (1,2))], [(1, (2,0))], [(2, (0,1))], [(1, (0,1)), (1, (1,2))],
        [(1, (0,1)), (1, (2,0))], [(1, (1,2)), (1, (0,1))], [(2, (1,2))], [(1, (2,0)), (1, (1,2))],
        [(2, (2,0))], [(3, (0,1))], [(2, (0,1)), (1, (1,2))], [(2, (0,1)), (1, (2,0))],
        [(2, (1,2)), (1, (2,0))], [(1, (0,1)), (2, (1,2))], [(1, (0,1)), (2, (2,0))],
        [(1, (1,2)), (2, (0,1))], [(3, (1,2))], [(3, (2,0))], [(3, (0,1)), (1, (1,2))],
        [(2, (0,1)), (1, (1,2)), (1, (0,1))], [(3, (1,2)), (1, (2,0))], [(1, (0,1)), (3, (1,2))]]
        
def random_rot(*arg):
    rot = choice(rots)
    res = []
    for a in arg:
        for k, axes in rot:
            a = np.rot90(a, k, axes)
        res.append(a)
    if len(res) == 1:
        return res[0]
    else:
        return res

def crop_shift(volume, joint_coord, joint_coord_p, c_s):
    crop_0 = c_s[0] if c_s[0] is list else [c_s[0], c_s[0]]
    crop_1 = c_s[1] if c_s[1] is list else [c_s[1], c_s[1]]
    crop_2 = c_s[2] if c_s[2] is list else [c_s[2], c_s[2]]
    shape = volume.shape
    volume = np.squeeze(volume[crop_0[0]:shape[0]-crop_0[1], crop_1[0]:shape[1]-crop_1[1], crop_2[0]:shape[2]-crop_2[1]])
    joint_coord = joint_coord - np.array([[c_s[3]], [c_s[4]], [c_s[5]]])
    if joint_coord_p is not None:
        joint_coord_p = joint_coord_p - np.array([[c_s[3]], [c_s[4]], [c_s[5]]])
    return volume, joint_coord, joint_coord_p
    

def read_nifti(nii_filename):
    data = nib.load(nii_filename)
    return np.squeeze(data.get_data().astype(np.float32))
        

class Dataset():

    def __init__(self, opts):
        self.data_test = opts.data_test
        rawdata_path = opts.rawdata_path
        c_s_dict = {'043015raw': [20, 20, 20, 0, 0, 20], '043015': [0, 0, 0, 0, 0, 20]}
        dataset_dict = []
        for folder in [os.path.join(rawdata_path, f) for f in sorted(os.listdir(rawdata_path))]:
            folder_basename = os.path.basename(folder)
            # get joint coord label
            label_filename = os.path.join(folder, folder_basename + '.mat')
            if os.path.isfile(label_filename):
                joint_coord = sio.loadmat(label_filename)['joint_coord'].reshape((-1, 15, 3)).transpose((0,2,1))  
            else: 
                joint_coord = np.array([])
            # get filenames
            niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            dataset_dict.append(
                {'foldername': folder_basename, 'filenames': niinames, 'labels': joint_coord.astype(np.int32), 'crop_and_shift': c_s_dict.get(folder_basename, None)})
        self.dataset_dict = dataset_dict
        
    
    def get_data_list(self, stage):
        if stage == 'pretrain':
            filenames = []
            for d in self.dataset_dict:
                filenames.extend(d['filenames'])
            return filenames
        #n_train_dict = {'040716': 150, '043015raw': 400, '031616': 90, '102617': 90, '031615': 90, '022618': 90}
        #{old, new, old, old, old, old}
        n_train_dict = {'040716': 150, '043015': 400, '031616': 90, '102617': 90, '031615': 90, '022618': 90}
        n_train_dict[self.data_test] = 0
        data_list = []
        for dataset in self.dataset_dict:
            n_labeled = dataset['labels'].shape[0]
            n_train = n_train_dict.get(dataset['foldername'],  2*(n_labeled//3))
            if stage == 'train':
                iterator = range(n_train)
                #iterator = list(range(n_train)) * (int(np.ceil(120.0 / n_train)) if n_train else 0)
            else:
                iterator = range(n_train, n_labeled)
            for i in iterator:
                data_list.append({'filename': dataset['filenames'][i], 
                                  'label':dataset['labels'][i], 
                                  'foldername':int(''.join(list(filter(str.isdigit, dataset['foldername'])))), 
                                  'crop_and_shift': dataset['crop_and_shift'],
                                  'label_prior':dataset['labels'][i-1 if i else 0]})
        return data_list

    def get_output_type_shape(self, stage, opts):
        nc_in = 1 + opts.temporal * (opts.nJoint + opts.nBone)
        if stage == 'pretrain':
            return (tf.float32, tf.float32, tf.int32), ([None, None, None, 1], [None, None, None, 1], [])
        elif stage == 'test':
            return (tf.float32, tf.int32, tf.int64, tf.int64), ([None, None, None, 1], [3, opts.nJoint + opts.nBone], [3], [])
        else:
            return (tf.float32, tf.float32), ([None, None, None, nc_in], [None, None, None, opts.nJoint + opts.nBone])

    def _get_dataset(self, stage, opts):
        is_shuffle = stage == 'train' or stage == 'pretrain'
        batch_size = 1 if stage == 'test' else opts.batch_size
        dtype, shape = self.get_output_type_shape(stage, opts)
        data_list = self.get_data_list(stage)
        map_fn = getattr(sys.modules[__name__], stage + '_map_fn')
        f = lambda i: map_fn(data_list[i], opts)           
        return get_dataset_from_indexable(f, dtype, shape, len(data_list), batch_size, is_shuffle)
            

    def get_dataset(self, opts):
        print("construct dataset")
        if opts.run == 'pretrain':
            print('get dataset for pretraining')
            return self._get_dataset('pretrain', opts)
        elif opts.run == 'test':
            print('get dataset for testing')
            return self._get_dataset('test', opts)
        else:
            print('get dataset for training and validation')
            return self._get_dataset('train', opts), self._get_dataset('val', opts)


def pretrain_map_fn(nii_filename, opts):
    volume = read_nifti(nii_filename)
    if opts.rot:
        volume = random_rot(volume)
    va, vb, label = crop_pretrain(volume)
    if opts.scale:
        va = uniform(1 - opts.scale, 1 + opts.scale) * va
        vb = uniform(1 - opts.scale, 1 + opts.scale) * vb
    return va,vb,label
    
    
def train_map_fn(data_dict, opts):
    volume, heatmap = val_map_fn(data_dict, opts)
    if opts.rot:
        volume, heatmap = random_rot(volume, heatmap)
    if opts.scale:
        if opts.scale_type == 'mul':
            volume[:,:,:,0] = uniform(1 - opts.scale, 1 + opts.scale) * volume[:,:,:,0]
        elif opts.scale_type == 'exp':
            volume[:,:,:,0] = volume[:,:,:,0] ** uniform(1 - opts.scale, 1 + opts.scale)
        else:
            raise Exception('scale type error')
    return volume, heatmap
    
    
def val_map_fn(data_dict, opts):
    volume = read_nifti(data_dict['filename'])
    joint_coord = data_dict['label']
    joint_coord_p = data_dict['label_prior'] if opts.temporal else None
    if data_dict['crop_and_shift'] is not None:
        volume, joint_coord, joint_coord_p = crop_shift(volume, joint_coord, joint_coord_p, data_dict['crop_and_shift'])
    if opts.norm:
        mean = np.mean(volume)
        std = np.std(volume)
    volume, heatmap, heatmap_p = crop_trainval(volume, joint_coord, joint_coord_p, opts.crop_size, opts.mag, opts.sigma, opts.bone)
    if opts.norm:
        volume = (volume - mean) / std
    if heatmap_p is not None:
        return np.concatenate((volume, heatmap_p), axis=-1), heatmap    
    else:
        return volume, heatmap
                      
    
def test_map_fn(data_dict, opts):
    volume = read_nifti(data_dict['filename'])
    joint_coord = data_dict['label']
    if data_dict['crop_and_shift'] is not None:
        volume, joint_coord, _ = crop_shift(volume, joint_coord, None, data_dict['crop_and_shift'])
    if opts.norm:
        volume = (volume - np.mean(volume)) / np.std(volume)
    return crop_test(volume), joint_coord.astype(np.int32), volume.shape, data_dict['foldername']


def crop_trainval(volume, joint_coord, joint_coord_p, crop_size, mag, sigma, bone):
    # size crop size
    crop_size_x, crop_size_y, crop_size_z = crop_size
    # generate random point
    s = volume.shape
    x_0, y_0, z_0 = randint(0, s[1] - crop_size_x), randint(0, s[0] - crop_size_y), randint(0, s[2] - crop_size_z)
    volume = volume[y_0:y_0+crop_size_y, x_0:x_0+crop_size_x, z_0:z_0+crop_size_z]
    # generate heatmap
    y_range = np.reshape(np.arange(y_0, y_0+crop_size_y, dtype=np.float32), (-1,1,1,1))
    x_range = np.reshape(np.arange(x_0, x_0+crop_size_x, dtype=np.float32), (1,-1,1,1))
    z_range = np.reshape(np.arange(z_0, z_0+crop_size_z, dtype=np.float32), (1,1,-1,1))
    
    bone = np.array(bone)
    
    def gen_hmap(joint):
        #joint = joint.astype(np.int64)
        x_label, y_label, z_label = np.reshape(joint, (3,1,1,1,-1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2
        if bone.size:
            vx = x_label[:, :, :, bone[:, 1]] - x_label[:, :, :, bone[:, 0]]
            vy = y_label[:, :, :, bone[:, 1]] - y_label[:, :, :, bone[:, 0]]
            vz = z_label[:, :, :, bone[:, 1]] - z_label[:, :, :, bone[:, 0]]
            vv = vx**2 + vy**2 + vz**2
            wv = dx[:, :, :, bone[:, 0]] * vx + dy[:, :, :, bone[:, 0]] * vy + dz[:, :, :, bone[:, 0]] * vz
            is_outside = np.logical_or(wv <= 0, wv >= vv)
            d_pb = dd[:,:,:,bone[:,0]] - wv**2 / (vv + 1e-8)
            '''
            if np.amin(d_pb) < 0:
                print('start')
                i, j, k, c = np.unravel_index(np.argmin(d_pb), d_pb.shape)
                y, x, z = y_range[i, 0, 0, 0], x_range[0, j, 0, 0], z_range[0, 0, j, 0]
                print(x, y, z)
                b0x, b0y, b0z = x_label[0, 0, 0, bone[c, 0]], y_label[0, 0, 0, bone[c, 0]], z_label[0, 0, 0, bone[c, 0]]
                print(b0x, b0y, b0z)
                b1x, b1y, b1z = x_label[0, 0, 0, bone[c, 1]], y_label[0, 0, 0, bone[c, 1]], z_label[0, 0, 0, bone[c, 1]]
                print(b1x, b1y, b1z)
                Vx, Vy, Vz = b1x - b0x, b1y-b0y, b1z-b0z
                VV = Vx**2 + Vy**2 + Vz**2
                Wx, Wy, Wz = x - b0x, y - b0y, z-b0z
                WV = Wx * Vx + Wy * Vy + Wz * Vz
                WW = Wx**2 + Wy**2 + Wz**2
                print(np.amin(d_pb), WW - WV**2 / VV, VV, vv[0, 0, 0, c])
            '''
            
            d_out = np.minimum(dd[:,:,:,bone[:,1]], dd[:,:,:,bone[:,0]])
            d_pb[is_outside] = d_out[is_outside]
            
            
            
            return np.concatenate((mag*np.exp((-1.0/2.0/sigma**2)*dd, dtype=np.float32), mag*np.exp((-1.0/2.0/sigma**2)*d_pb, dtype=np.float32)), axis=-1)
        else:
            return mag*np.exp((-1.0/2.0/sigma**2)*dd, dtype=np.float32)
    #x_label, y_label, z_label = np.reshape(joint_coord, (3,1,1,1,-1))
    #heatmap = mag*np.exp((-1.0/2.0/sigma**2)*((x_range - x_label)**2 + (y_range - y_label)**2 + (z_range - z_label)**2), dtype=np.float32)
    heatmap = gen_hmap(joint_coord)
    if joint_coord_p is not None:
        heatmap_p = gen_hmap(joint_coord_p)
        #x_label, y_label, z_label = np.reshape(joint_coord_p, (3,1,1,1,-1))
        #heatmap_p = mag*np.exp((-1.0/2.0/sigma**2)*((x_range - x_label)**2 + (y_range - y_label)**2 + (z_range - z_label)**2), dtype=np.float32)
    else:
        heatmap_p = None
    return np.expand_dims(volume, -1), heatmap, heatmap_p
    
                
def crop_test(volume):
    pad_width = [(int((ceil(s/8.0)*8-s)/2), int(ceil((ceil(s/8.0)*8-s)/2))) for s in volume.shape]
    v = np.pad(volume, pad_width,  'edge')
    v = np.expand_dims(v, -1)
    return v
        

def crop_pretrain(volume):
    crop_size = 32
    th = (crop_size + 3)**2;
    s = volume.shape
    s = [s[0] - crop_size, s[1] - crop_size, s[2] - crop_size]
    p1x, p1y, p1z = randint(0, s[0]), randint(0, s[1]), randint(0, s[2])
    for ii in range(20):
        p2x, p2y, p2z = randint(0, s[0]), randint(0, s[1]), randint(0, s[2])
        dx, dy, dz = p1x - p2x, p1y-p2y, p1z-p2z
        if dx**2 > th or dy**2 > th or dz**2 > th:
            break
    
    def get_direction(d):
        if d < -crop_size/2:
            return 1
        elif d > crop_size/2:
            return 2
        else:
            return 0
        
    x, y, z = get_direction(dx), get_direction(dy), get_direction(dz)
        
    va = volume[p1x:p1x+crop_size, p1y:p1y+crop_size, p1z:p1z+crop_size]    
    vb = volume[p2x:p2x+crop_size, p2y:p2y+crop_size, p2z:p2z+crop_size]
    label = x + 3*y + 9*z - 1
    if label < 0:
        label = randint(0, 25)
    
    return np.expand_dims(va, -1), np.expand_dims(vb, -1), label    
    
    
