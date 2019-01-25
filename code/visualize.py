import tensorflow as tf
import math


joint_name = ['ankle_l', 'ankle_r', 'knee_l', 'knee_r', 'bladder', 'elbow_l', 'elbow_r', 'eye_l', 'eye_r', 'hip_l', 'hip_r', 'shoulder_l', 'shoulder_r', 'wrist_l', 'wrist_r'];


def image_summary(inputs, labels, coord_rel, outputs, opts):
    inputs = inputs[:1]
    labels = labels[:1]
    outputs = outputs[-1][:1]

    for i, j in enumerate(opts.joint):
        z = tf.minimum(tf.maximum(coord_rel[0,2,i], 0), 63)
        if i == 0:
            tf.summary.image('input', inputs[:, :, :, z, :1])
        tf.summary.image('label %s' % joint_name[j], tf.cast(255.0/opts.mag*labels[:, :, :, z, i:i+1], tf.uint8))
        tf.summary.image('output %s'% joint_name[j], tf.cast(255.0/opts.mag*tf.nn.relu(outputs[:, :, :, z, i:i+1]), tf.uint8))
        tf.summary.scalar('max %s' % joint_name[j], tf.reduce_max(outputs[:, :, :, :, i]))


def loss_summary(mean, joint, bone):
    for i, j in enumerate(joint):
        tf.summary.scalar('%s_loss' % joint_name[j], mean[i])
    for i, j in enumerate(bone):
        tf.summary.scalar('%s_%s_loss' % (joint_name[j[0]], joint_name[j[1]]), mean[i + len(joint)])
    tf.summary.scalar('all', mean[-1])