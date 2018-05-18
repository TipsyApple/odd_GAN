import tensorflow as tf
import numpy as np

from data import *

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class GanModel():
    def name(self):
        return 'GanModel'
    
    def __init__(self, data, generator=None, discriminator=None):
        # self._size = 
        self.data = data
        self._generator = generator if generator!=None else DefaultGenerator()
        self._discriminator = discriminator if discriminator!=None else DefaultDiscriminator()

        self.X = tf.placeholder(tf.float32, shape=[None, 1024, 1024, 3], name='real_image_in') # 需要改
        self.Z = tf.placeholder(tf.float32, shape=[None, 1024, 1024, 3], name='blur_image_in') # 需要改
        
        # net
        self.G_sample = self._generator(self.Z)
        self.D_real,_ = self._discriminator(self.X)
        self.D_fake,_ = self._discriminator(self.G_sample, reuse = True)

        # loss
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        self.G_loss = - tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self._discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self._generator.vars)

        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self._discriminator.vars]

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_folder=None, training_epoches = 1000000, batch_size = 4):
        self.sess.run(tf.global_variables_initializer())
        print('---------- Networks initialized -------------')

        for epoch in range(training_epoches):
            # update D
            print('%d/%d'%(epoch,training_epoches), end='\r')
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            n_d = 1
            for _ in range(n_d):
                X_b = self.data(batch_size)
                self.sess.run(self.clip_D)
                self.sess.run(
                        self.D_solver,
                        feed_dict={self.X: X_b, self.Z: X_b}
                        )
            # update G
            self.sess.run(
                self.G_solver,
                feed_dict={self.Z: X_b}
            )

            # print loss. save images.
            if epoch % 10 == 0 or epoch < 10:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.Z: X_b})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.Z: X_b})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))
        
        print('----------------- Thanks! -------------------')

    def _add_gt_image(self):
        pass
        # add back mean
        # image = self._image + cfg.PIXEL_MEANS

        # BGR to RGB (opencv uses BGR)
        # resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        # self._gt_image = tf.reverse(resized, axis=[-1])

class DefaultGenerator(object):
    def __init__(self, name=None):
        self.name = "default_generator"
    
    def __call__(self, Z):
        with tf.variable_scope(self.name) as scope:
            g = tf.contrib.layers.conv2d(Z, num_outputs=16, kernel_size=4, stride=2)
            g = tf.contrib.layers.conv2d_transpose(g, 3, 3, stride=2,
                                                    activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class DefaultDiscriminator(object):
    def __init__(self, name=None):
        self.name = "default_discriminator"
    
    def __call__(self, X, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tf.contrib.layers.conv2d(X, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tf.contrib.layers.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm)
            shared = tf.contrib.layers.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm)
            shared = tf.contrib.layers.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm)

            shared = tf.contrib.layers.flatten(shared)
    
            d = tf.contrib.layers.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tf.contrib.layers.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm)
            q = tf.contrib.layers.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d, q
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

if __name__ == '__main__':
    data = Data()
    print(data()[0])
    run_net = GanModel(Data())
    run_net.train()