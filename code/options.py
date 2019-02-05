import argparse
import os
import datetime
from utils import *


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        
        # path
        self.parser.add_argument('--rawdata_path', default='../rawdata/', type=str,
                                 help='raw data path')
        self.parser.add_argument('--tfrecords_path', default='../tfrecords/', type=str,
                                 help='tfrecords path')
        self.parser.add_argument('--output_path', default='../results/',
                                 type=str, help='output path')
        self.parser.add_argument('--name', type=str, default='')
                                 
        # GPU
        self.parser.add_argument('--gpu_id', type=str, default='0')
        self.parser.add_argument('--ngpu', type=int, default=0)
        
        # run
        self.parser.add_argument('--run', type=check_arg(str, ['train', 'test', 'pretrain']), default='train')
        
        self.parser.add_argument('--data_test', type=str, default='031616')
        
        # train
        self.parser.add_argument('--use_pretrain', type=str, default='')
        self.parser.add_argument('--continue', type=str, default='')
        self.parser.add_argument('--epoch_continue', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size') #pretrin 32 # train 5
        self.parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for') #30
        self.parser.add_argument('--save_freq', type=int, default=50)
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate') #adam: 0.0001,  pretrain: 0.001
        self.parser.add_argument('--lr_decay_ep', type=float, default=100)
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
        self.parser.add_argument('--lr_decay_method', type=str, default='exp')
        self.parser.add_argument('--optimizer', type=str, default='adam')
        self.parser.add_argument('--random_seed', type=int, default='123')
        self.parser.add_argument('--k_init', type=str, default='glorot_uniform')
        self.parser.add_argument('--no_mid_supervise', action='store_true', default=False)
        self.parser.add_argument('--train_like_test', type=float, default=0.0)
        self.parser.add_argument('--boneLambda', type=float, default=0.0)
                
        # input
        self.parser.add_argument('--rot', action='store_true', default=False)
        self.parser.add_argument('--scale', type=float, default=0.0)
        self.parser.add_argument('--scale_type', type=str, default='mul')
        self.parser.add_argument('--norm', action='store_true', default=False)
        self.parser.add_argument('--use_global', action='store_true', default=False)
        self.parser.add_argument('--joint', type=str, default='all') #15
        self.parser.add_argument('--bone', type=str, default='[]')
        self.parser.add_argument('--crop_size', type=str, default='64,64,64')
        self.parser.add_argument('--temporal', action='store_true', default=False)
        
        # heat map
        self.parser.add_argument('--sigma', type=float, default=2.0)
        self.parser.add_argument('--mag', type=float, default=10.0)
        
        # network structure
        self.parser.add_argument('--network', type=str, default='shg')
        self.parser.add_argument('--nStacks', type=int, default=1)
        self.parser.add_argument('--depth', type=int, default=3)
        self.parser.add_argument('--nFeat', type=int, default=96) #96
        self.parser.add_argument('--normlayer', type=check_arg(str, ['bn', 'in', 'none']), default='bn')
        self.parser.add_argument('--res', action='store_true', default=False)
        self.parser.add_argument('--res2', action='store_true', default=False)
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # gpu
        if self.opt.ngpu:
            self.opt.gpu_id = get_gpu(self.opt.ngpu, 11000)
        else:
            self.opt.ngpu = len(self.opt.gpu_id.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)
        # heatmap zoom
        self.opt.fac = 2 if self.opt.network == 'simple' else 1
        # crop size
        self.opt.crop_size = [int(s) for s in self.opt.crop_size.split(',')]
        # time
        self.opt.time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # joint
        if self.opt.joint == 'all':
            self.opt.joint = list(range(15))
        else:
            self.opt.joint = [(int(j)-1) for j in self.opt.joint.split(',')]
        self.opt.nJoint = len(self.opt.joint)
        # bone
        if self.opt.boneLambda:
            self.opt.bone = eval(self.opt.bone)
        else:
            self.opt.bone = []
        self.opt.nBone = len(self.opt.bone)
        # name
        if self.opt.name == '':
            self.opt.name = self.opt.time
        if self.opt.run == 'pretrain':
            self.opt.name += '_pretrain'
        # random seed
        set_random_seed(['py','np','tf'], self.opt.random_seed)

        expr_dir = os.path.join(self.opt.output_path, self.opt.name)
        if self.opt.run == 'test':
            # load from disk
            self.opt = load_yaml(os.path.join(expr_dir, 'opt.yaml'), self.opt, key_to_drop=['run'])
        else:
            # save to disk
            mkdir(expr_dir)
            save_yaml(os.path.join(expr_dir, 'opt.yaml'), self.opt, 
                      key_to_drop=['gpu_id', 'ngpu', 'save_freq', 'buffer_size'])
            copyfiles('./', os.path.join(expr_dir, 'backup_code'), '*.py')
            
        print_args(self.opt)
             
        return self.opt
        

