import tensorflow as tf

"""basic"""


def bn(inputs, training, scale=False, center=True):
    return tf.layers.batch_normalization(
        inputs=inputs, training=training, fused=True, scale=scale, center=center
    )


def conv3d(
    inputs, nc_out, k=3, s=1, activation=None, k_init="glorot_uniform", use_bias=False
):
    k_init = tf.keras.initializers.get(k_init)
    out = tf.layers.conv3d(
        inputs,
        filters=nc_out,
        kernel_size=[k, k, k],
        strides=(s, s, s),
        padding="same",
        activation=activation,
        kernel_initializer=k_init,
        use_bias=use_bias,
    )
    return out


def deconv3d(inputs, nc_out, k=3, s=1, k_init="glorot_uniform", use_bias=False):
    k_init = tf.keras.initializers.get(k_init)
    out = tf.layers.conv3d_transpose(
        inputs,
        filters=nc_out,
        kernel_size=[k, k, k],
        strides=(s, s, s),
        padding="same",
        kernel_initializer=k_init,
        use_bias=use_bias,
    )
    return out


def resblock_bottleneck(inputs, nc_out, training, k_init):
    nc_in = inputs.get_shape().as_list()[-1]
    nc_mid = nc_out // 2
    out = tf.nn.relu(bn(inputs, training))
    if nc_in != nc_out:
        residual = conv3d(out, nc_out, 1, k_init=k_init)
    else:
        residual = inputs
    out = conv3d(out, nc_mid, 1, k_init=k_init)
    out = tf.nn.relu(bn(out, training))
    out = conv3d(out, nc_mid, 3, k_init=k_init)
    out = tf.nn.relu(bn(out, training))
    out = conv3d(out, nc_out, 1, k_init=k_init)

    return out + residual


def resblock_basic(inputs, nc_out, s, training, k_init):
    nc_in = inputs.get_shape().as_list()[-1]
    out = tf.nn.relu(bn(inputs, training))
    if s != 1 or nc_in != nc_out:
        residual = conv3d(inputs, nc_out, 1, s=s, k_init=k_init)
    else:
        residual = inputs
    out = conv3d(inputs, nc_out, 3, s=s, k_init=k_init)
    out = tf.nn.relu(bn(out, training))
    out = conv3d(out, nc_out, 3, k_init=k_init)
    return out + residual


"""hourglass"""


class hourglass:
    def __init__(self, inputs, depth, nFeat, training, k_init, gpu_id):

        self.inputs = inputs
        with tf.device("/gpu:%d" % gpu_id[0]):
            encoders = [inputs]
            for _ in range(depth):
                low = tf.layers.max_pooling3d(encoders[-1], pool_size=2, strides=2)
                low = resblock_bottleneck(low, nFeat, training, k_init)
                encoders.append(low)
            decoders = [resblock_bottleneck(encoders[-1], nFeat, training, k_init)]

        skips = []
        for i, encoder in enumerate(encoders[0:-1]):
            with tf.device("/gpu:%d" % gpu_id[1]) if i == 0 else tf.device(
                "/gpu:%d" % gpu_id[0]
            ):
                skips.append(resblock_bottleneck(encoder, nFeat, training, k_init))

        with tf.device("/gpu:%d" % gpu_id[1]):
            for skip in skips[-1::-1]:
                low = resblock_bottleneck(decoders[-1], nFeat, training, k_init)
                up = tf.keras.layers.UpSampling3D()(low)
                decoders.append(up + skip)

        self.outputs = decoders[-1]


class stacked_hourglass:
    def __init__(self, x, opts, training, reuse=None):
        self.inputs = x
        self.hg = []
        nStacks = opts.nStacks
        depth = opts.depth
        nFeat = opts.nFeat
        nClasses = opts.nJoint
        k_init = opts.k_init

        if nStacks == opts.ngpu:
            gpu_ids = list(zip(range(nStacks), range(nStacks)))
        elif 2 * nStacks == opts.ngpu:
            gpu_ids = list(zip(range(0, 2 * nStacks, 2), range(1, 2 * nStacks, 2)))
        elif opts.ngpu == 1:
            gpu_ids = [(0, 0)] * nStacks
        else:
            raise Exception("n GPU Error")

        with tf.variable_scope("shg", reuse=reuse):
            # head
            with tf.variable_scope("head"):
                with tf.device("/gpu:%d" % gpu_ids[0][0]):
                    x = conv3d(x, nFeat // 2, 5, k_init=k_init)
                    x = bn(x, training)
                    x = tf.nn.relu(x)
                    x = resblock_bottleneck(x, nFeat // 2, training, k_init)
                    x = resblock_bottleneck(x, nFeat, training, k_init)
            out = []
            for i in range(nStacks):
                # hg
                with tf.variable_scope("hg%d" % i):
                    self.hg.append(
                        hourglass(x, depth, nFeat, training, k_init, gpu_ids[i])
                    )
                    y = self.hg[-1].outputs

                with tf.device("/gpu:%d" % gpu_ids[i][1]):
                    # res
                    y = resblock_bottleneck(y, nFeat, training, k_init)
                    # fc
                    y = conv3d(y, nFeat, 1, k_init=k_init)
                    y = bn(y, training)
                    y = tf.nn.relu(y)
                    # score
                    score = conv3d(y, nClasses, 1, k_init=k_init)
                    out.append(score)

                if i < (nStacks - 1):
                    with tf.device("/gpu:%d" % gpu_ids[i + 1][0]):
                        fc_ = conv3d(y, nFeat, 1, k_init=k_init)
                        score_ = conv3d(score, nFeat, 1, k_init=k_init)
                        x = x + fc_ + score_

        self.outputs = out


""" UNet """


class unet:
    def __init__(self, x, opts, training):

        nfeat = [opts.nFeat, 2 * opts.nFeat, 4 * opts.nFeat, 8 * opts.nFeat]
        nClasses = opts.nJoint

        with tf.variable_scope("unet"):

            down1_feat = self.conv3x3(x, nfeat[0], training)

            down2_feat = tf.layers.max_pooling3d(down1_feat, pool_size=2, strides=2)
            down2_feat = self.conv3x3(down2_feat, nfeat[1], training)

            down3_feat = tf.layers.max_pooling3d(down2_feat, pool_size=2, strides=2)
            down3_feat = self.conv3x3(down3_feat, nfeat[2], training)

            bottom_feat = tf.layers.max_pooling3d(down3_feat, pool_size=2, strides=2)
            bottom_feat = self.conv3x3(bottom_feat, nfeat[3], training)

            up1_feat = self.upconcat(bottom_feat, down3_feat, nfeat[2])
            up1_feat = self.conv3x3(up1_feat, nfeat[2], training)

            up2_feat = self.upconcat(up1_feat, down2_feat, nfeat[1])
            up2_feat = self.conv3x3(up2_feat, nfeat[1], training)

            up3_feat = self.upconcat(up2_feat, down1_feat, nfeat[0])
            up3_feat = self.conv3x3(up3_feat, nfeat[0], training)

            self.outputs = [conv3d(up3_feat, nClasses, 1)]

    def conv3x3(self, x, nfeat, training):
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        return x

    def upconcat(self, x, y, nfeat):
        x = deconv3d(x, nfeat, k=2, s=2)
        x = tf.concat((y, x), -1)
        return x


"""HRNet"""


class hrnet:
    def __init__(self, x, opts, training, reuse=None):

        self.n_feat = [opts.nFeat, 2 * opts.nFeat, 4 * opts.nFeat, 8 * opts.nFeat]
        nClasses = opts.nJoint
        self.k_init = opts.k_init
        self.training = training

        with tf.variable_scope("hrnet", reuse=reuse):

            x = conv3d(x, self.n_feat[0] // 2, 5, k_init=self.k_init)
            x = bn(x, training)
            x_list = [tf.nn.relu(x)]
            x = conv3d(x, self.n_feat[0], 3, k_init=self.k_init)

            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 2)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 2)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 3)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 3)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 4)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 4)
            x_list = self.trans(x_list)
            x_list = self.fuse(x_list, 1)
            x_list = self.trans(x_list)
            y = bn(x_list[0], training)
            y = tf.nn.relu(y)
            score = conv3d(y, nClasses, 1, k_init=self.k_init)
            self.outputs = [score]

    def fuse_layer(self, x, i, j):
        if i == j:
            return x
        elif i < j:
            x = self.up(x, 2 ** (j - i), self.n_feat[i])
            return x
        else:
            for _ in range(i - j):
                x = self.down(x, self.n_feat[j])
            return x

    def trans(self, x_list):
        for i in range(len(x_list)):
            x_list[i] = resblock_basic(
                x_list[i], self.n_feat[i], 1, self.training, self.k_init
            )
        return x_list

    def fuse(self, x_list, n_scale_out):
        y_list = []
        for i in range(n_scale_out):
            y = self.fuse_layer(x_list[0], i, 0)
            for j in range(1, len(x_list)):
                y = y + self.fuse_layer(x_list[j], i, j)
            y_list.append(y)
        return y_list

    def down(self, x, nc):
        x = bn(x, self.training)
        x = tf.nn.relu(x)
        x = conv3d(x, nc, k=3, s=2, k_init=self.k_init)
        return x

    def up(self, x, s, nc):
        x = bn(x, self.training)
        x = tf.nn.relu(x)
        x = conv3d(x, nc, k=1, k_init=self.k_init)
        x = tf.keras.layers.UpSampling3D(size=(s, s, s))(x)
        return x


""" discriminator """


class disc:
    def __init__(self, x, opts, training, reuse=False):

        # nFeat = opts.nFeat // 4
        nFeat = 16
        nfeat = [nFeat, 2 * nFeat, 4 * nFeat, 8 * nFeat]

        with tf.variable_scope("disc", reuse=reuse):

            down1_feat = self.conv3x3(x, nfeat[0], training)

            down2_feat = tf.layers.max_pooling3d(down1_feat, pool_size=2, strides=2)
            down2_feat = self.conv3x3(down2_feat, nfeat[1], training)

            down3_feat = tf.layers.max_pooling3d(down2_feat, pool_size=2, strides=2)
            down3_feat = self.conv3x3(down3_feat, nfeat[1], training)

            bottom_feat = tf.layers.max_pooling3d(down3_feat, pool_size=2, strides=2)
            bottom_feat = self.conv3x3(bottom_feat, nfeat[2], training)

            ######

            bottom_feat = tf.layers.max_pooling3d(bottom_feat, pool_size=2, strides=2)
            bottom_feat = self.conv3x3(bottom_feat, nfeat[2], training)

            bottom_feat = tf.layers.max_pooling3d(bottom_feat, pool_size=2, strides=2)
            bottom_feat = self.conv3x3(bottom_feat, nfeat[3], training)

            self.outputs = [conv3d(bottom_feat, 1, 1)]

    def conv3x3(self, x, nfeat, training):
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        return x


class autoencoder:
    def __init__(self, x, opts, training, reuse=False):

        nFeat = opts.nFeat // 4
        nfeat = [nFeat, 2 * nFeat, 4 * nFeat, 8 * nFeat]

        nClasses = opts.nJoint

        with tf.variable_scope("autoencoder", reuse=reuse):

            down1_feat = self.conv3x3(x, nfeat[0], training)

            down2_feat = tf.layers.max_pooling3d(down1_feat, pool_size=2, strides=2)
            down2_feat = self.conv3x3(down2_feat, nfeat[1], training)

            down3_feat = tf.layers.max_pooling3d(down2_feat, pool_size=2, strides=2)
            down3_feat = self.conv3x3(down3_feat, nfeat[2], training)

            bottom_feat = tf.layers.max_pooling3d(down3_feat, pool_size=2, strides=2)
            bottom_feat = self.conv3x3(bottom_feat, nfeat[3], training)

            # self.outputs = [conv3d(bottom_feat, 1, 1)]

            up1_feat = self.upconcat(bottom_feat, down3_feat, nfeat[2])
            up1_feat = self.conv3x3(up1_feat, nfeat[2], training)

            up2_feat = self.upconcat(up1_feat, down2_feat, nfeat[1])
            up2_feat = self.conv3x3(up2_feat, nfeat[1], training)

            up3_feat = self.upconcat(up2_feat, down1_feat, nfeat[0])
            up3_feat = self.conv3x3(up3_feat, nfeat[0], training)

            self.outputs = [conv3d(up3_feat, nClasses, 1)]

    def conv3x3(self, x, nfeat, training):
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        x = conv3d(x, nfeat, 3)
        x = bn(x, training)
        x = tf.nn.relu(x)
        return x

    def upconcat(self, x, y, nfeat):
        x = deconv3d(x, nfeat, k=2, s=2)
        # x = tf.concat((y, x), -1)
        return x


def get_model(volume, target, opts):

    if opts.run == "test":
        training = False
    else:
        training = tf.placeholder(tf.bool, shape=())

    if opts.network == "shg":
        model = stacked_hourglass(volume, opts, training)
    elif opts.network == "unet":
        model = unet(volume, opts, training)
    elif opts.network == "hrnet":
        model = hrnet(volume, opts, training)
    else:
        raise Exception("network error")

    if opts.run != "test" and opts.network_d != "":
        if opts.network_d == "disc":
            model_d_fake = disc(
                tf.concat([volume, model.outputs[-1]], 4), opts, training
            )
            model_d_real = disc(
                tf.concat([volume, target], 4), opts, training, reuse=True
            )
        elif opts.network_d == "autoencoder":
            model_d_fake = autoencoder(
                tf.concat([volume, model.outputs[-1]], 4), opts, training
            )
            model_d_real = autoencoder(
                tf.concat([volume, target], 4), opts, training, reuse=True
            )
        else:
            raise Exception("network error")
    else:
        model_d_fake = model_d_real = None
    return model, model_d_fake, model_d_real, training


def get_network(volume, target, opts):
    model, model_d_fake, model_d_real, training = get_model(volume, target, opts)
    if model_d_fake is not None:
        return model.outputs, model_d_fake.outputs, model_d_real.outputs, training
    else:
        return model.outputs, None, None, training


def get_heatmap(labels, opts):

    sigma_0 = 2.0

    if opts.sigma == 0:
        if not opts.notinv:
            with tf.variable_scope(opts.network):
                inv_var = tf.get_variable(
                    "inv_var",
                    shape=[1, 1, 1, 1, opts.nJoint],
                    initializer=tf.constant_initializer((1 / sigma_0) ** 2),
                    trainable=opts.varLambda > 0,
                )
                if opts.minsig:
                    inv_var = tf.minimum(inv_var, 1.0 / opts.minsig**2)
            hm_var = 1 / inv_var
        else:
            with tf.variable_scope(opts.network):
                hm_var = tf.get_variable(
                    "hm_var",
                    shape=[1, 1, 1, 1, opts.nJoint],
                    initializer=tf.constant_initializer(sigma_0**2),
                    trainable=opts.varLambda > 0,
                )
                if opts.minsig:
                    hm_var = tf.maximum(hm_var, opts.minsig**2)
            inv_var = 1 / hm_var
        if opts.hmmax:
            labels = opts.mag * tf.exp(-0.5 * inv_var * labels)
        else:
            if opts.correct:
                labels = (
                    opts.mag
                    * (sigma_0**3)
                    * (inv_var**1.5)
                    * tf.exp(-0.5 * inv_var * labels)
                )
            else:
                labels = (
                    opts.mag
                    * (sigma_0**2)
                    * inv_var
                    * tf.exp(-0.5 * inv_var * labels)
                )
    else:
        hm_var = None

    return labels, hm_var


def downsample(volume):
    volume = tf.layers.max_pooling3d(volume, pool_size=2, strides=2)
    volume = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(volume)
    return volume
