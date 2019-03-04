import numpy as np
import tensorflow as tf
from load_data import load_data
from ops import conv2d, dense, conv2d_tr, deconv2d

class VAE():
    
    def __init__(self):
        self.data = load_data()
        
        self.z_size = 20
        self.dense_units = 512
        self.batch_size = 64
        
        self.images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        
        # estimators of mean and sdt in latent space
        z_mu, z_sigma = self.encoder(images)
        eps = tf.random.normal([self.batch_size, self.z_size], 0., 1., dtype=tf.float32)
        z = z_mu + (eps * z_sigma)
        
        # generated images
        self.gen_images = self.decoder(z)
        
        tiny = 1e-8 # to avoid zeros in logs
        
        self.
