from options import OptionsInference
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from model import get_network
from test_utils import average_coord
import scipy.io as sio


if __name__ == "__main__":
    opts = OptionsInference().parse()
    opts.run = "test"
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    # network
    outputs, _, _, _ = get_network(inputs, None, opts)
    # save and load
    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opts.network)
    )
    # inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("restore trained model")
        saver.restore(
            sess,
            os.path.join(
                opts.output_path,
                opts.name,
                "model%d.ckpt" % (opts.epoch_continue),
            ),
        )
        joint_coords = []
        down_factor = int(2**opts.depth)
        for f in sorted(os.listdir(opts.rawdata_path)):
            if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                continue
            nii = nib.load(os.path.join(opts.rawdata_path, f))
            img = nii.get_fdata()
            if opts.norm:
                img = img / np.percentile(img[img > 0], 99)
            v = np.zeros(
                (
                    1,
                    int(np.ceil(img.shape[0] / down_factor) * down_factor),
                    int(np.ceil(img.shape[1] / down_factor) * down_factor),
                    int(np.ceil(img.shape[2] / down_factor) * down_factor),
                    1,
                )
            )
            v[0, : img.shape[0], : img.shape[1], : img.shape[2], 0] = img

            output_val = sess.run(outputs[-1], feed_dict={inputs: v})

            c = np.argmax(np.reshape(output_val, (-1, output_val.shape[-1])), 0)
            s = (v.shape[1], v.shape[2], v.shape[3])
            joint_coord = np.zeros((3, len(c)))
            for j in range(len(c)):
                ind = np.array(np.unravel_index(c[j], s))
                x, y, z = average_coord(output_val[0, :, :, :, j], ind)
                joint_coord[0, j] = x
                joint_coord[1, j] = y
                joint_coord[2, j] = z

            joint_coords.append(joint_coord)

        if joint_coords:
            sio.savemat(
                opts.output_label,
                {"joint_coord": np.stack(joint_coords)},
            )
        else:
            print("empty")
