from data import Dataset
from model import get_network
from losses import cross_entropy_loss
import tensorflow as tf
import os
import time
    
def pretrain(opts):
    # dataset and iterator
    dataset = Dataset(opts.rawdata_path)
    dataset_train = dataset.get_dataset(opts)
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    v_a, v_b, label = iterator.get_next()
    
    # network
    outputs, training = get_network(tf.concat((v_a, v_b), axis=0), opts)
    
    # save and load
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network))

    # loss
    loss, accuracy = cross_entropy_loss(outputs, label, opts)
    
    # summary
    writer_train = tf.summary.FileWriter(os.path.join(opts.output_path, opts.time, 'logs'), tf.get_default_graph())
    summary_op = tf.summary.merge_all()
    
    # optimizer
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(opts.lr, global_step, dataset_train.length*67, 0.1, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step, colocate_gradients_with_ops=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(update_ops + [train_op])

    # main loop
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        print('training loop start')
        start_train = time.clock()
        for epoch in range(1, opts.epochs + 1):
            print('epoch: %d' % epoch)
            start_ep = time.clock()
            # train
            print('training')
            sess.run(iterator.make_initializer(dataset_train))
            while True:
                try:
                    summary_train, _ = sess.run([summary_op, train_op], feed_dict={training: True})
                    writer_train.add_summary(summary_train, tf.train.global_step(sess, global_step))
                except tf.errors.OutOfRangeError:
                    break
            print('step: %d' % tf.train.global_step(sess, global_step))
            # save model
            if epoch % opts.save_freq == 0 or epoch == opts.epochs:
                print('save model')
                saver.save(sess, os.path.join(opts.output_path, opts.time, 'pretrainmodel.ckpt'))
            print("epoch end, elapsed time: %ds, total time: %ds" %(time.clock() - start_train, time.clock() - start_ep))
        print('training loop end')
        writer_train.close()