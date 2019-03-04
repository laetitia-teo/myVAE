import numpy as np
import tensorflow as tf

def load_data():
    mnist = tf.keras.datasets.mnist
    (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
    X = np.concatenate((Xtrain, Xtest), axis=0)
    X = np.expand_dims(X, axis=-1) # add channel
    y = tf.keras.utils.to_categorical(np.concatenate((ytrain, ytest), axis=0))
    # TODO : normalize inputs
    return X, y
