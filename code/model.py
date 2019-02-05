import tensorflow as tf
import contextlib

def bn(inputs, training, normlayer):
    if normlayer == 'bn':
        return tf.layers.batch_normalization(inputs=inputs, training=training, fused=True, scale=False)
    elif normlayer == 'in':
        return tf.contrib.layers.instance_norm(inputs=inputs, scale=False)
    else:
        return inputs
        

def conv3d(inputs, nc_out, k=3, s=1, activation=None, k_init='glorot_uniform', use_bias=True):
    k_init = tf.keras.initializers.get(k_init)
    out = tf.layers.conv3d(inputs, filters=nc_out, kernel_size=[k,k,k], strides=(s,s,s),
                           padding="same", activation=None, 
                           kernel_initializer=k_init, use_bias=use_bias)
    return out

'''hourglass'''

def resblock(inputs, nc_out, training, normlayer, k_init):
    nc_in = inputs.get_shape().as_list()[-1]
    nc_mid = nc_out // 2
    residual = inputs
    out = bn(inputs, training, normlayer)
    out = conv3d(tf.nn.relu(out), nc_mid, 1, k_init=k_init)
    out = bn(out, training, normlayer)
    out = conv3d(tf.nn.relu(out), nc_mid, 3, k_init=k_init)
    out = bn(out, training, normlayer)
    out = conv3d(tf.nn.relu(out), nc_out, 1, k_init=k_init)
    if nc_in != nc_out:
        out += conv3d(residual, nc_out, 1, k_init=k_init)
    else:
        out += residual
    return out
    
def hourglass(inputs, depth, nFeat, training, normlayer, k_init, gpu_id, pretrain):
    with tf.device('/gpu:%d'%gpu_id[0]):
        encoders = [inputs]
        for ii in range(depth):
            low = tf.layers.max_pooling3d(encoders[-1], pool_size=2, strides=2)
            low = resblock(low, nFeat, training, normlayer, k_init)
            encoders.append(low)
        
        decoders = [resblock(encoders[-1], nFeat, training, normlayer, k_init)]
    
    if pretrain:
        return decoders[-1]
    
    skips = []
    for i, encoder in enumerate(encoders[0:-1]):
        with tf.device('/gpu:%d'%gpu_id[1]) if i==0 else tf.device('/gpu:%d'%gpu_id[0]):
            skips.append(resblock(encoder, nFeat, training, normlayer, k_init))
    
    with tf.device('/gpu:%d'%gpu_id[1]):
        for skip in skips[-1::-1]:
            low = resblock(decoders[-1], nFeat, training, normlayer, k_init)
            up = tf.keras.layers.UpSampling3D()(low)
            decoders.append(up + skip)
        
    return decoders[-1]


def stacked_hourglass(x, opts, training):
    inputs = x
    nStacks = opts.nStacks
    depth = opts.depth
    nFeat = opts.nFeat
    nClasses = opts.nJoint + opts.nBone
    normlayer = opts.normlayer
    is_pretrain = opts.run == 'pretrain'
    k_init = opts.k_init

    if nStacks == opts.ngpu:
        gpu_ids = list(zip(range(nStacks), range(nStacks)))
    elif 2 * nStacks == opts.ngpu:
        gpu_ids = list(zip(range(0, 2*nStacks, 2), range(1, 2*nStacks, 2)))
    elif opts.ngpu == 1:
        gpu_ids = [(0, 0)] * nStacks
    else:
        raise Exception('n GPU Error')

    with tf.variable_scope('shg'):
        # head
        with tf.variable_scope('head'):
            with tf.device('/gpu:%d'%gpu_ids[0][0]):
                x = conv3d(x, nFeat//2, 5, k_init=k_init)
                x = bn(x, training, normlayer)
                x = tf.nn.relu(x)
                x = resblock(x, nFeat//2, training, normlayer, k_init)
                x = resblock(x, nFeat, training, normlayer, k_init)
        out = []
        for i in range(nStacks):
            # hg
            with tf.variable_scope('hg%d'%i):
                y = hourglass(x, depth, nFeat, training, normlayer, k_init, gpu_ids[i], is_pretrain)
                if is_pretrain:
                    break
                
            with tf.device('/gpu:%d'%gpu_ids[i][1]):
                # res
                y = resblock(y, nFeat, training, normlayer, k_init)
                # fc
                y = conv3d(y, nFeat, 1, k_init=k_init)
                y = bn(y, training, normlayer)
                if not (opts.res2 and opts.temporal):
                    y = tf.nn.relu(y)
                # score
                score = conv3d(y, nClasses, 1, k_init=k_init)
                if opts.res2 and opts.temporal:
                    score = score + inputs[:,:,:,:,1:]
                out.append(score)

            if i < (nStacks - 1):
                with tf.device('/gpu:%d'%gpu_ids[i+1][0]):
                    fc_ = conv3d(y, nFeat, 1, k_init=k_init)
                    if opts.res and opts.temporal:
                        score_ = conv3d(score + inputs[:,:,:,:,1:], nFeat, 1, k_init=k_init)
                    else:
                        score_ = conv3d(score, nFeat, 1, k_init=k_init)
                    x = x + fc_ + score_
    if is_pretrain:
        out = pretrain_output(y, training, normlayer)
    return out

def pretrain_output(x, training, normlayer):
    with tf.variable_scope('pretrain_out'):
        x = bn(x, training, normlayer)
        x = tf.nn.relu(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = bn(x, training, normlayer)
        x = tf.nn.relu(x)
        x_a = x[:x.shape[0]//2]
        x_b = x[x.shape[0]//2:]
        x_ab = tf.concat((x_a, x_b), axis=1)
        y = tf.layers.dense(x_ab, 1024, activation=tf.nn.relu)
        y = tf.layers.dense(y, 26)
    return y
    
''' simple baseline'''

def basic_block(x, out_planes, s, training, k_init):
    in_planes = x.get_shape().as_list()[-1]

    out = conv3d(x, out_planes, 3, s=s, k_init=k_init, use_bias=False)
    out = tf.layers.batch_normalization(inputs=out, training=training, fused=True)
    out = tf.nn.relu(out)
    
    out = conv3d(out, out_planes, 3, k_init=k_init, use_bias=False)
    out = tf.layers.batch_normalization(inputs=out, training=training, fused=True)
    
    if s != 1 or in_planes != out_planes:
        residual = conv3d(x, out_planes, 1, s=s, k_init=k_init, use_bias=False)
        residual = tf.layers.batch_normalization(inputs=residual, training=training, fused=True)
    else:
        residual = x    
    out = tf.nn.relu(out + residual)
    
    return out

def simple_res(x, opts, training):

    #nDown = opts.nStacks
    nDeconv = opts.depth
    nFeat = opts.nFeat
    nClasses = opts.nJoint + opts.nBone
    k_init = opts.k_init

    with tf.variable_scope('simple'):
        with tf.variable_scope('head'):
            x = conv3d(x, nFeat, 7, s=2, k_init=k_init, use_bias=False)
            x = tf.layers.batch_normalization(inputs=x, training=training, fused=True)
            x = tf.nn.relu(x)
            #x = tf.layers.max_pooling3d(x, pool_size=3, strides=2, padding='same')
        
        layers = [2, 2, 2, 2]
        strides = [1, 2, 2, 2]
        planes = [nFeat, nFeat*2, nFeat*4, nFeat*8]
        
        with tf.variable_scope('mid'):
            for l, s, p in zip(layers, strides, planes):
                x = basic_block(x, p, s, training, k_init)
                for i in range(1, l):
                    x = basic_block(x, p, 1, training, k_init)
        
        with tf.variable_scope('deconv'):
            for i in range(nDeconv):
                x = tf.layers.conv3d_transpose(x, filters=nFeat*4, kernel_size=(4,4,4), strides=(2,2,2), padding='same', use_bias=False)
                x = tf.layers.batch_normalization(inputs=x, training=training, fused=True)
                x = tf.nn.relu(x)
            
        with tf.variable_scope('final'):
            x = conv3d(x, nClasses, 1, k_init=k_init)
    return [x]
            

        
def get_network(volume, opts):

    if opts.run == 'test':
        training = False
    else:
        training = tf.placeholder(tf.bool, shape=())

    if opts.network == 'shg':
        outputs = stacked_hourglass(volume, opts, training)
    elif opts.network == 'simple':
        outputs = simple_res(volume, opts, training)
    else:
        raise Exception('network error')
    return outputs, training
