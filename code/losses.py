import tensorflow as tf
from visualize import loss_summary, gan_loss_summary


def cross_entropy_loss(outputs, labels, opts):
    labels = tf.one_hot(labels, 26)
    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    tf.summary.scalar("cross_entropy", loss)
    prediction = tf.argmax(outputs, 1)
    correct_answer = tf.argmax(labels, 1)
    equality = tf.equal(prediction, correct_answer)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return loss, accuracy


def spatial_soft_argmax(x, T):

    shape = tf.shape(x)
    d, h, w, c = shape[1], shape[2], shape[3], shape[4]
    pos1, pos2, pos3 = tf.meshgrid(
        tf.lin_space(-1.0, 1.0, num=d),
        tf.lin_space(-1.0, 1.0, num=h),
        tf.lin_space(-1.0, 1.0, num=w),
        indexing="ij",
    )
    pos1 = tf.reshape(pos1, [d * h * w])
    pos2 = tf.reshape(pos2, [d * h * w])
    pos3 = tf.reshape(pos3, [d * h * w])
    x = tf.reshape(tf.transpose(x, [0, 4, 1, 2, 3]), [-1, d * h * w])

    if T:
        weight = tf.nn.softmax(T * x)
        weight_sum = 1.0
    else:
        weight = tf.nn.relu(x)
        weight_sum = tf.reduce_sum(weight, [1], keepdims=True) + 1e-3

    epos1 = tf.reduce_sum(pos1 * weight, [1], keepdims=True) / weight_sum
    epos2 = tf.reduce_sum(pos2 * weight, [1], keepdims=True) / weight_sum
    epos3 = tf.reduce_sum(pos3 * weight, [1], keepdims=True) / weight_sum

    dr2 = (pos1 - epos1) ** 2 + (pos2 - epos2) ** 2 + (pos3 - epos3) ** 2
    var = tf.reduce_sum(dr2 * weight, [1], keepdims=True) / weight_sum
    var = tf.reshape(var, [-1, c])

    return var


def loc_loss(outputs, loc, opts):
    var = spatial_soft_argmax(outputs[-1], opts.locT)
    mask = tf.logical_and(tf.greater_equal(loc, -1.0), tf.less_equal(loc, 1.0))
    mask = tf.cast(tf.reduce_all(mask, [1], keepdims=False), tf.float32)
    return tf.reduce_mean(var * mask, axis=(0, 1))


def mse_loss(outputs, labels, loc, hm_var, opts):

    loss_one = 0.0
    for out in outputs:
        loss_one += tf.reduce_mean(tf.square(out - labels), axis=(0, 1, 2, 3))

    loss_all = tf.reduce_mean(loss_one)
    mean = [0] * (opts.nJoint + 1)
    update_op = [0] * (opts.nJoint + 1)
    for i in range(opts.nJoint):
        mean[i], update_op[i] = tf.metrics.mean(
            loss_one[i], updates_collections=tf.GraphKeys.UPDATE_OPS
        )
    mean[-1], update_op[-1] = tf.metrics.mean(
        loss_all, updates_collections=tf.GraphKeys.UPDATE_OPS
    )

    loss_summary(mean, opts.joint)

    if opts.locLambda:
        locloss = loc_loss(outputs, loc, opts)
        loss_all = loss_all + opts.locLambda * locloss
        m_locloss, u_op_locloss = tf.metrics.mean(
            locloss, updates_collections=tf.GraphKeys.UPDATE_OPS
        )
        tf.summary.scalar("loc_loss", m_locloss)
        update_op.append(u_op_locloss)

    if opts.sigma == 0:
        varloss = tf.reduce_mean(hm_var)
        loss_all = loss_all + opts.varLambda * varloss
        m_varloss, u_op_varloss = tf.metrics.mean(
            varloss, updates_collections=tf.GraphKeys.UPDATE_OPS
        )
        tf.summary.scalar("var_loss", m_varloss)
        update_op.append(u_op_varloss)

    return loss_all, update_op, labels


def gan_loss(d_fake, d_real, outputs, target, opts):

    if d_fake is None:
        return None, None, None, None

    d_fake = d_fake[-1]
    d_real = d_real[-1]

    if opts.network_d == "disc":
        loss_g = opts.gan_coef * tf.reduce_mean(tf.square(d_fake - 1.0))
        loss_d = opts.gan_coefdf * tf.reduce_mean(
            tf.square(d_fake)
        ) + opts.gan_coefdt * tf.reduce_mean(tf.square(d_real - 1.0))
    else:
        loss_g = opts.gan_coef * tf.reduce_mean(tf.square(d_fake - outputs[-1]))
        loss_d = opts.gan_coefdt * tf.reduce_mean(
            tf.square(d_real - target)
        ) - opts.gan_coefdf * tf.reduce_mean(tf.square(d_fake - outputs[-1]))

    mean_g, update_g = tf.metrics.mean(
        loss_g, updates_collections=tf.GraphKeys.UPDATE_OPS
    )
    mean_d, update_d = tf.metrics.mean(
        loss_d, updates_collections=tf.GraphKeys.UPDATE_OPS
    )

    gan_loss_summary(mean_g, mean_d)

    return loss_g, loss_d, update_g, update_d
