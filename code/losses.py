import tensorflow as tf
from visualize import loss_summary


def cross_entropy_loss(outputs, labels, opts):
    labels = tf.one_hot(labels, 26)

    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    tf.summary.scalar('cross_entropy', loss)
    
    prediction = tf.argmax(outputs, 1)
    correct_answer = tf.argmax(labels, 1)
    equality = tf.equal(prediction, correct_answer)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    return loss, accuracy
    

def mse_loss(outputs, labels, opts):

    if opts.no_mid_supervise:
        outputs = outputs[-1:]

    loss_one = 0.0
    for out in outputs:
        loss_one += tf.reduce_mean(tf.square(out - labels), axis=(0, 1, 2, 3))
    if opts.boneLambda:
        loss_all = tf.reduce_mean(loss_one) 
        #loss_all = tf.reduce_mean(loss_one[:opts.nJoint]) + opts.boneLambda * tf.reduce_mean(loss_one[-opts.nBone:])
    else:
        loss_all = tf.reduce_mean(loss_one)

    mean = [0]*(opts.nJoint + opts.nBone + 1)
    update_op = [0]*(opts.nJoint+ opts.nBone + 1)
    for i in range(opts.nJoint + opts.nBone):
        mean[i], update_op[i] = tf.metrics.mean(loss_one[i], updates_collections=tf.GraphKeys.UPDATE_OPS)
            
    mean[-1], update_op[-1] = tf.metrics.mean(loss_all, updates_collections=tf.GraphKeys.UPDATE_OPS)

    loss_summary(mean, opts.joint, opts.bone)
    return loss_all, update_op
