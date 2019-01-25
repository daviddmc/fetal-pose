import tensorflow as tf


class Optimizer():
    
    def __init__(self, opts, pre_list, new_list, step_per_ep):
        self.global_step = tf.Variable(0, trainable=False)
        self.two_stage = opts.epoch_continue == 0 and opts.use_pretrain != ''
        self.pre_list = pre_list
        self.new_list = new_list
        self.step_per_ep = step_per_ep
        self.epochs = opts.epochs
        self.learning_rate = self.get_lr(opts.lr, opts.lr_decay_ep, opts.lr_decay_gamma, opts.lr_decay_method)
        self.method = opts.optimizer
        if self.method == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        elif self.method == 'sgd':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9, use_nesterov = True)
        elif self.method.startswith('adamw'):
            wd = self.method[5:]
            wd = float(wd) if wd else 0.0
            self.optimizer = tf.contrib.opt.AdamWOptimizer(wd, learning_rate = self.learning_rate)
        else:
            raise Exception('optimizer name error')
        
    def get_global_step(self, sess):
        return tf.train.global_step(sess, self.global_step)
        
    def get_train_op(self, loss):
        grad_and_var = self.optimizer.compute_gradients(loss, var_list=self.pre_list+self.new_list, colocate_gradients_with_ops=True)
        if self.two_stage:
            for i in range(len(grad_and_var)):
                g, v = grad_and_var[i]
                if v in self.pre_list:
                    g = self.pretrain_var_grad(g)
                grad_and_var[i] = (g, v)
        train_op = self.optimizer.apply_gradients(grad_and_var, global_step=self.global_step)
        train_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [train_op])
        return train_op
        
    def get_lr(self, lr, lr_decay_ep, lr_decay_gamma, method):
        if method == 'exp':
            return 1e-8 + tf.train.exponential_decay(lr, self.global_step, int(lr_decay_ep * self.step_per_ep), lr_decay_gamma, staircase=True)
        elif method == 'cos_restart':
            return tf.train.cosine_decay_restarts(lr, self.global_step, int(lr_decay_ep * self.step_per_ep), alpha=lr_decay_gamma)
        elif method == '':
            return lr  
        else:
            raise Exception('optimizer name error')  
        
    def pretrain_var_grad(self, g):
        if self.method == 'adam':
            return tf.to_float(self.global_step > self.step_per_ep * self.epochs // 2) * g
        elif self.method == 'sgd':
            return 0.01 * g
        
    