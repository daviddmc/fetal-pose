from options import Options
from data import Dataset
from visualize import image_summary
from model import get_network
from losses import mse_loss
from test import test_result, save_test_result,  first_heatmap_p, get_heatmap_p, reset_dict
import tensorflow as tf
import os
import time
from pretrain import pretrain
from optimizer import Optimizer
import numpy as np
import random
import numpy as np


np.random.seed(123)
random.seed(123)

# options
opts = Options().parse()

if opts.run == 'pretrain':
    pretrain(opts)
    os._exit(0)
    

def train_fun(dataset, opts):
    # dataset and iterator
    dataset_train, dataset_val = dataset.get_dataset(opts)
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    if opts.train_like_test:
        volume, label = iterator.get_next()
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1 + opts.temporal * opts.nJoint])
        labels = tf.placeholder(tf.float32, shape=[None, None, None, None, opts.nJoint])
    else:
        inputs, labels = iterator.get_next()
    
    # network
    outputs, training = get_network(inputs, opts)

    # loss
    loss, mean_update_ops = mse_loss(outputs, labels, opts)
    
    # summary
    writer_train = tf.summary.FileWriter(os.path.join(opts.output_path, opts.name, 'logs', 'train'), tf.get_default_graph())
    writer_val = tf.summary.FileWriter(os.path.join(opts.output_path, opts.name, 'logs', 'val'))
    summary_op = tf.summary.merge_all()
    
    # varlist
    name_list = [ns[0] for ns in tf.train.list_variables(os.path.join(opts.output_path, opts.use_pretrain, 'pretrainmodel.ckpt'))] if opts.use_pretrain != '' else []
    pretrain_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network) if v.name[:-2] in name_list]
    newtrain_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network) if v.name[:-2] not in name_list]
    print('pretrain var: %d, newtrain var: %d' % (len(pretrain_list), len(newtrain_list)))  
              
    # optimizer
    optimizer = Optimizer(opts, pretrain_list, newtrain_list, dataset_train.length)
    train_op = optimizer.get_train_op(loss)
    my_update_op = tf.group(mean_update_ops)
    
    # save and load
    saver = tf.train.Saver(var_list=newtrain_list + pretrain_list)
    if opts.use_pretrain != '':
        saver_pretrain = tf.train.Saver(var_list = pretrain_list)

    # main loop
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        if opts.use_pretrain != '':
            saver_pretrain.restore(sess, os.path.join(opts.output_path, opts.use_pretrain + 'pretrain', 'pretrainmodel.ckpt'))
        if opts.epoch_continue > 0:
            saver.restore(sess, os.path.join(opts.output_path, opts.use_continue, 'model%d.ckpt'%opts.epoch_continue))
        print('training loop start')
        start_train = time.time()
        for epoch in range(opts.epoch_continue + 1, opts.epochs + 1):
            print('epoch: %d' % epoch)
            start_ep = time.time()
            # train
            print('training')
            sess.run(iterator.make_initializer(dataset_train))
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    if opts.train_like_test:
                        v, l = sess.run([volume, label])
                        if random.random() < opts.train_like_test:
                            l_p = sess.run(outputs[-1], feed_dict={training:False, inputs:v})
                            v = np.concatenate([v[:,:,:,:,:1], l_p], axis=-1)
                        summary_train, _ = sess.run([summary_op, train_op], feed_dict={training: True, inputs:v, labels:l})
                    else:
                        summary_train, _ = sess.run([summary_op, train_op], feed_dict={training: True})
                except tf.errors.OutOfRangeError:
                    writer_train.add_summary(summary_train, epoch)
                    break
            print('step: %d' % optimizer.get_global_step(sess))
            # validation
            print('validation')
            sess.run(iterator.make_initializer(dataset_val))
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    if opts.train_like_test:
                        v, l = sess.run([volume, label])
                        if random.random() < opts.train_like_test:
                            l_p = sess.run(outputs[-1], feed_dict={training:False, inputs:v})
                            v = np.concatenate([v[:,:,:,:,:1], l_p], axis=-1)
                        summary_val, _ = sess.run([summary_op, my_update_op], feed_dict={training: False, inputs:v, labels:l})
                    else:
                        summary_val, _ = sess.run([summary_op, my_update_op], feed_dict={training: False})
                except tf.errors.OutOfRangeError:
                    writer_val.add_summary(summary_val, epoch)
                    break
            # save model
            if epoch % opts.save_freq == 0 or epoch == opts.epochs:
                print('save model')
                saver.save(sess, os.path.join(opts.output_path, opts.name, 'model%d.ckpt'%epoch))
            print("epoch end, elapsed time: %ds, total time: %ds" %(time.time() - start_ep, time.time() - start_train))
        print('training loop end')
        writer_train.close()
        writer_val.close()
    opts.run = 'test'
    
    
def test_fun(dataset, opts):
    test_arg_list = []
    for test_rot in [None]:
        for test_flip in [None]:
            test_arg_list.append((test_rot, test_flip))
    for test_arg in test_arg_list:
    #for test_arg in [(None, None)]:
        opts.test_arg = test_arg
        # dataset and iterator
        dataset_val = dataset.get_dataset(opts)
        iterator = dataset_val.make_one_shot_iterator()
        volume, joint_coord, shape, data_num = iterator.get_next()
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1 + opts.temporal * opts.nJoint])
        dn_p = 0
        
        # network
        outputs, _ = get_network(inputs, opts)
            
        # save and load
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network))
                
        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('restore trained model')
            saver.restore(sess, os.path.join(opts.output_path, opts.name, 'model%d.ckpt'%opts.epochs))
            print('test start')
            res = []
            while True:
                try:
                    v, joint, s, dn = sess.run([volume, joint_coord, shape, data_num])
                    if opts.temporal:
                        if np.squeeze(dn) != dn_p:
                            output_val = first_heatmap_p(joint, s, opts)
                            dn_p = np.squeeze(dn)
                        else:
                            output_val = get_heatmap_p(output_val, opts)
                        output_val = sess.run(outputs[-1], feed_dict={inputs: np.concatenate([v, output_val], axis=-1)})
                    else:
                        output_val = sess.run(outputs[-1], feed_dict={inputs: v})
                    res.append(test_result(output_val, joint, s, dn, opts))
                except tf.errors.OutOfRangeError:
                    break
            save_test_result(res, opts)
            reset_dict()
        tf.reset_default_graph()
        print("test end, elapsed time: ", time.time() - start)
    

if __name__ == '__main__':
    dataset = Dataset(opts)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(123)
        if opts.run == 'train':
            train_fun(dataset, opts)
    tf.reset_default_graph()
    test_fun(dataset, opts)
