import yaml
import os
from argparse import Namespace
import glob
from shutil import copy2

def save_yaml(path, params_dict, key_to_save=None, key_to_drop=None):
    params_dict = to_dict(params_dict)
    if key_to_save is not None:
        params_dict = dict((k, params_dict[k]) for k in key_to_save)   
    elif key_to_drop is not None:
        params_dict = dict((k, params_dict[k]) for k in params_dict.keys() if k not in key_to_drop)  
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(params_dict, f, default_flow_style=True, allow_unicode=True)
        

def load_yaml(path, params_dict=None, key_to_load=None, key_to_drop=None):
    with open(path) as f:
        load_dict = yaml.load(f)
        if key_to_load is not None:
            load_dict = dict((k, load_dict[k]) for k in key_to_load)   
        elif key_to_drop is not None:
            load_dict = dict((k, load_dict[k]) for k in load_dict.keys() if k not in key_to_drop) 
        if params_dict is not None:
            if type(params_dict) is Namespace:
                params_dict = to_dict(params_dict)
                params_dict.update(load_dict)
                return Namespace(**params_dict)
            else:
                params_dict.update(load_dict)
                return params_dict
        else:
            return load_dict


def to_namespace(d):
    if type(d) is dict:
        return Namespace(**d)
    elif type(d) is Namespace:
        return d 
    else:
        return Namespace(**vars(d))
        

def to_dict(ns):
    if type(ns) is dict:
        return ns
    else:
        return vars(ns)
    
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def print_args(args):
    args = to_dict(args)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
        
        
def copyfiles(src, des, exp):
    mkdir(des)
    for f in glob.glob(os.path.join(src, exp)):
        copy2(f, des)
        
        
def check_arg(parse_fn=None, valid_list=None):
    def new_parse_fn(x):
        if parse_fn is not None:
            x = parse_fn(x)
        if valid_list is not None:
            if x not in valid_list:
                raise ValueError()
        return x
    return new_parse_fn
    
    
def to_bool(s):
    try:
        return bool(float(s))
    except ValueError:
        return s.lower() in ['true', 't', 'yes']
        
    
def set_random_seed(module_list, seed = 0):
    if type(module_list) is not list:
        module_list = [module_list]
    for m in module_list:
        m = m.lower()
        if m in ['python', 'py', 'random']:
            import random
            random.seed(seed)
        elif m in ['np', 'numpy']:
            import numpy as np
            np.random.seed(seed)
        elif m in ['tf', 'tensorflow']:
            import tensorflow as tf
            tf.set_random_seed(seed)
        else:
            raise ValueError(' ')
            
def get_gpu(ngpu, threshold):
    tmp = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    memory_gpu=[int(x.split()[2]) for x in tmp.readlines()]
    gpu_available = []
    for i, mem in enumerate(memory_gpu):
        if mem > threshold:
            gpu_available.append(str(i))
    if len(gpu_available) < ngpu:
        raise Exception('not enough gpu available')
    return ','.join(gpu_available[:ngpu])

        