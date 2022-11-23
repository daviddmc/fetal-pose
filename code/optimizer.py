import tensorflow as tf


class Optimizer():
    
    def __init__(self, opts, varlist_g, varlist_d, step_per_ep):
        self.global_step = tf.Variable(0, trainable=False)
        self.varlist_g = varlist_g
        self.varlist_d = varlist_d
        self.step_per_ep = step_per_ep
        self.epochs = opts.epochs
        self.learning_rate = self.get_lr(opts.lr, opts.lr_decay_ep, opts.lr_decay_gamma, opts.lr_decay_method)
        self.method = opts.optimizer
        if self.method == 'adam':
            self.optimizer_g = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            if self.varlist_d:
                self.optimizer_d = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        elif self.method == 'sgd':
            if self.varlist_d:
                self.optimizer_d = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9, use_nesterov = True)
            self.optimizer_g = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9, use_nesterov = True)
        elif self.method.startswith('adamw'):
            wd = self.method[5:]
            wd = float(wd) if wd else 0.0
            if self.varlist_d:
                self.optimizer_d = tf.contrib.opt.AdamWOptimizer(wd, learning_rate = self.learning_rate)   
            self.optimizer_g = tf.contrib.opt.AdamWOptimizer(wd, learning_rate = self.learning_rate)
        else:
            raise Exception('optimizer name error')
        
    def get_global_step(self, sess):
        return tf.train.global_step(sess, self.global_step)
        
    def get_train_op(self, loss, loss_g, loss_d):
        if loss_g is not None:
            grad_and_var_g = self.optimizer_g.compute_gradients(loss + loss_g, var_list=self.varlist_g, colocate_gradients_with_ops=True)
            train_op_g = self.optimizer_g.apply_gradients(grad_and_var_g, global_step=self.global_step)
            grad_and_var_d = self.optimizer_d.compute_gradients(loss_d, var_list=self.varlist_d, colocate_gradients_with_ops=True)
            train_op_d = self.optimizer_d.apply_gradients(grad_and_var_d)
            train_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [train_op_g, train_op_d])
        else:
            grad_and_var_g = self.optimizer_g.compute_gradients(loss, var_list=self.varlist_g, colocate_gradients_with_ops=True)
            train_op_g = self.optimizer_g.apply_gradients(grad_and_var_g, global_step=self.global_step)
            train_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [train_op_g])
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
        
        
    