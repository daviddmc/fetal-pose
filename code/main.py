from options import Options
from data import Dataset
from model import get_network, get_heatmap, downsample
from losses import mse_loss, gan_loss
from test import test_result, save_test_result, reset_dict
import os
import time
from optimizer import Optimizer
import numpy as np
import random
import numpy as np
import tensorflow as tf

np.random.seed(123)
random.seed(123)

# options
opts = Options().parse()


def train_fun(dataset, opts):
    # dataset and iterator
    dataset_train, dataset_val = dataset.get_dataset(opts)
    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes
    )
    inputs, labels, loc = iterator.get_next()

    if opts.downsample:
        inputs = downsample(inputs)

    labels, v = get_heatmap(labels, opts)

    # network
    outputs, d_fake, d_real, training = get_network(inputs, labels, opts)

    # loss
    loss, mean_update_ops, labels = mse_loss(outputs, labels, loc, v, opts)
    loss_g, loss_d, update_g, update_d = gan_loss(d_fake, d_real, outputs, labels, opts)

    # summary
    writer_train = tf.summary.FileWriter(
        os.path.join(opts.output_path, opts.name, "logs", "train"),
        tf.get_default_graph(),
    )
    writer_val = tf.summary.FileWriter(
        os.path.join(opts.output_path, opts.name, "logs", "val")
    )
    summary_op = tf.summary.merge_all()

    # varlist
    varlist_g_saved = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network
    )
    varlist_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opts.network)
    if opts.network_d:
        varlist_d = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=opts.network_d
        )
    else:
        varlist_d = []
    # print(varlist_g)
    print("num of var: %d" % (len(varlist_g) + len(varlist_d)))

    # optimizer
    optimizer = Optimizer(opts, varlist_g, varlist_d, dataset_train.length)
    train_op = optimizer.get_train_op(loss, loss_g, loss_d)
    if update_g is not None:
        my_update_op = tf.group(mean_update_ops + [update_g, update_d])
    else:
        my_update_op = tf.group(mean_update_ops)

    # save and load
    saver = tf.train.Saver(var_list=varlist_g_saved)

    # main loop
    with tf.Session(
        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    ) as sess:
        sess.run(tf.global_variables_initializer())
        if opts.epoch_continue > 0:
            print("continue: %s" % opts.use_continue)
            saver.restore(
                sess,
                os.path.join(
                    opts.output_path,
                    opts.use_continue,
                    "model%d.ckpt" % opts.epoch_continue,
                ),
            )
        print("training loop start")
        start_train = time.time()
        for epoch in range(
            opts.epoch_continue + 1, opts.epoch_continue + opts.epochs + 1
        ):
            print("epoch: %d" % epoch)
            start_ep = time.time()
            # train
            print("training")
            sess.run(iterator.make_initializer(dataset_train))
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    summary_train, _ = sess.run(
                        [summary_op, train_op], feed_dict={training: True}
                    )
                except tf.errors.OutOfRangeError:
                    writer_train.add_summary(summary_train, epoch)
                    break
            dataset.update_datalist("train")
            print("step: %d" % optimizer.get_global_step(sess))
            # validation
            print("validation")
            sess.run(iterator.make_initializer(dataset_val))
            sess.run(tf.local_variables_initializer())
            summary_val = None
            while True:
                try:
                    summary_val, _ = sess.run(
                        [summary_op, my_update_op], feed_dict={training: False}
                    )
                except tf.errors.OutOfRangeError:
                    if summary_val is not None:
                        writer_val.add_summary(summary_val, epoch)
                    break
            dataset.update_datalist("val")
            # save model
            if epoch % opts.save_freq == 0 or epoch == opts.epochs:
                print("save model")
                saver.save(
                    sess,
                    os.path.join(opts.output_path, opts.name, "model%d.ckpt" % epoch),
                )
            print(
                "epoch end, elapsed time: %ds, total time: %ds"
                % (time.time() - start_ep, time.time() - start_train)
            )
        print("training loop end")
        writer_train.close()
        writer_val.close()
    opts.run = "test"


def test_fun(dataset, opts):
    test_arg_list = []
    for test_rot in [None]:
        for test_flip in [None]:
            test_arg_list.append((test_rot, test_flip))
    for test_arg in test_arg_list:
        opts.test_arg = test_arg
        # dataset and iterator
        dataset_test = dataset.get_dataset(opts)
        iterator = dataset_test.make_one_shot_iterator()
        volume, joint_coord, shape, data_num = iterator.get_next()

        # if opts.downsample:
        #    volume = downsample(volume)

        inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])

        # network
        outputs, _, _, _ = get_network(inputs, None, opts)

        # save and load
        saver = tf.train.Saver(
            var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network
            )
        )

        start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("restore trained model")
            saver.restore(
                sess,
                os.path.join(
                    opts.output_path,
                    opts.name,
                    "model%d.ckpt" % (opts.epoch_continue + opts.epochs),
                ),
            )
            print("test start")
            res = []
            while True:
                try:
                    v, joint, s, dn = sess.run([volume, joint_coord, shape, data_num])
                    output_val = sess.run(outputs[-1], feed_dict={inputs: v})
                    res.append(test_result(output_val, joint, s, dn, opts))
                except tf.errors.OutOfRangeError:
                    break
            save_test_result(res, opts)
            reset_dict()
        tf.reset_default_graph()
        print("test end, elapsed time: ", time.time() - start)


if __name__ == "__main__":
    dataset = Dataset(opts)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(123)
        if opts.run == "train":
            train_fun(dataset, opts)
    if opts.run == "test":
        tf.reset_default_graph()
        test_fun(dataset, opts)
