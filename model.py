import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import MnistDataset
from ops import conv2d, dense, deconv2d

class VAE():
    
    def __init__(self):
        # Data and training :
        self.batch_size = 64
        self.dataset = MnistDataset(self.batch_size)
        self.n_epochs = 20
        
        # Model hyperparameters :
        self.z_size = 20
        
        # TODO : make this more modular, adapt to other datasets
        self.images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        
        # estimators of mean and sdt in latent space
        z_mu, z_sigma = self.encoder(self.images)
        eps = tf.random.normal([self.batch_size, self.z_size], 0, 1, dtype=tf.float32)
        z = z_mu + (z_sigma * eps)
        
        # generated images
        self.gen_images = self.decoder(z)
        
        tiny = 1e-7 # to avoid nan from logs
        
        # latent loss : KL divergence between our estimators and a normal distribution
        self.lat_loss = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.square(z_sigma)
            - tf.log(tf.square(z_sigma)) - 1, 1)
        
        # generative loss : distance between 
        flat = tf.reshape(self.images, shape=[self.batch_size, 28*28])
        flat = tf.clip_by_value(flat, tiny, 1 - tiny)
        gen_flat = tf.reshape(self.gen_images, shape=[self.batch_size, 28*28])
        gen_flat = tf.clip_by_value(gen_flat, tiny, 1 - tiny)
        # cross entropy between original and reconstructed image
        self.gen_loss = -tf.reduce_sum(flat * tf.log(gen_flat) 
            + (1-flat) * tf.log(1 - gen_flat), 1)
        
        # total loss is the sum of both losses :
        self.loss = tf.reduce_mean(self.gen_loss + self.lat_loss)
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        # other stuff
        self.save_path = os.getcwd() + "/saves/model"
    
    def encoder(self, images):
        """
        The encoding network. It converts an image into a mean and a standard
        deviation in latent space.
        
        Arguments:
            - images (Tensor): input image
        Returns:
            - mu (Tensor): mean
            - sigma (Tensor): standard deviation
        """
        with tf.variable_scope("encoder"):
            conv1 = tf.nn.relu(conv2d(self.images, 5, 16, "conv1"))
            conv2 = tf.nn.relu(conv2d(conv1, 5, 32, "conv2"))
            flat = tf.reshape(conv2, [self.batch_size, 7*7*32])
            
            dense_mu = dense(flat, self.z_size, "dense_mu")
            dense_sigma = dense(flat, self.z_size, "dense_sigma")
        
        return dense_mu, dense_sigma
    
    def decoder(self, z):
        """
        The decoding (or generative) network. It converts a vector from the
        latent space into an image.
        
        Arguments:
            - z (Tensor): a vector in latent space
        Returns:
            - image (Tensor): generated image
        """
        with tf.variable_scope("decoder"):
            z_expand1 = dense(z, 7*7*32, "expand1")
            z_matrix = tf.nn.relu(tf.reshape(z_expand1, [self.batch_size, 7, 7, 32]))
            deconv1 = tf.nn.relu(deconv2d(z_matrix, 5, 
                [self.batch_size, 14, 14, 16], "deconv1"))
            deconv2 = deconv2d(deconv1, 5, [self.batch_size, 28, 28, 1], "deconv2")
            gen_image = tf.nn.sigmoid(deconv2)
        
        return gen_image
    
    @staticmethod
    def plot_grid(images):
        """
        Utility function for plotting 8x8 grids of images.
        """
        for i in range(8):
            for j in range(8):
                if j == 0:
                    row = images[8*i+j]
                else:
                    row = np.concatenate((row, images[8*i+j]), axis=1)
            if i == 0:
                stack = row
            else:
                stack = np.concatenate((stack, row), axis=0)
        plt.imshow(stack, cmap='gray')
        plt.show()
    
    def recode_vis(self, session=None):
        """
        Plots a visualization of 64 samples of the data.
        """
        idx = np.random.randint(self.dataset.size, size=64)
        #print(self.dataset[idx])
        original_imgs = self.dataset.X[idx]
        self.plot_grid(original_imgs[:, :, :, 0])
        # generated images
        if not session:
            sess = tf.Session()
            try:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)
            except:
                sess.run(tf.global_variables_initializer())
        else:
            sess = session
        generated_imgs = sess.run(self.gen_images, 
                                  feed_dict={self.images: original_imgs})
        self.plot_grid(generated_imgs[:, :, :, 0])
        if not session:
            sess.close()
    
    def gen_vis(self):
        """
        Plots a visualization of generated samples in a 8x8 grid.
        (Use after training for interesting results, of course)
        """
        zs = np.random.normal([64, self.z_size]) # 64 normal vectors used
        
    
    def train(self, vis=False):
        """
        Train the model on the dataset.
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                print('epoch %s' % epoch)
                for batch, _ in self.dataset:
                    _, lat_l, gen_l = sess.run((self.opt, self.lat_loss, self.gen_loss), 
                                               feed_dict={self.images: batch})
                    print('lat_loss : %s, gen_loss : %s' % (lat_l, gen_l))
                if vis:
                    self.recode_vis(sess) # Plot some reconstructed images
            save_path = saver.save(sess, self.save_path)
            print("Model saved in path : %s" % save_path)









































