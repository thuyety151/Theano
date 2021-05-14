import _pickle as cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T

import matplotlib
import matplotlib.pyplot as plt

from network import Network,FullyConnectedLayer,SoftmaxLayer,ConvPoolLayer
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

def load_data_shared(filename="./content/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.cmap'] = 'gray'
    for i in range(9):
      plt.subplot(1,10,i+1)
      plt.imshow(test_data[0][i].reshape(28,28))
      plt.axis('off')
      plt.title(str(test_data[1][i]))

    plt.show()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

if __name__ =="__main__":

    training_data, validation_data, test_data = load_data_shared()

    mini_batch_size = 10

    # net = Network([
    #         FullyConnectedLayer(n_in=784, n_out=100),
    #         SoftmaxLayer(n_in=100, n_out=10)
    #     ], mini_batch_size)

    # net.SGD(training_data, 1, mini_batch_size, 0.1, validation_data, test_data)

    # add a convolutional layer: 
    
    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                        filter_shape=(20, 1, 5, 5),
                        poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], 
                mini_batch_size)

    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
