from network import VariationalAutoEncoder
import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import theano

def load_data(dataset='/Users/sarin1991gmailcom/Documents/opt/Analytics/analytics/DyadicFlow/Dataset/MNIST/Data/mnist.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    return train_set, valid_set, test_set

def batchTrain(variational_autoencoder,X,BatchSize=128):
    """This is the function which handles the training of the Convolutional Neural Network"""
    X = X.astype('float32')
    print X.shape
    BS = BatchSize
    NB = len(X)/BS
    k = 0
    error = []
    for j in range(NB+1):
        x = X[j*BS:(j+1)*BS]
        e = variational_autoencoder.Train(x)
        error.append(e)
        print e
        k = k + 1
    print np.mean(error)
    np.random.shuffle(X)
        
def main():
    train_set, valid_set, test_set = load_data()
    X_train = train_set[0].reshape((-1,1,28,28))
    X_test = test_set[0].reshape((-1,1,28,28))
    X_valid = valid_set[0].reshape((-1,1,28,28))
    
    LearningRate = theano.shared(np.asarray(0.01,dtype=theano.config.floatX))
    variational_autoencoder = VariationalAutoEncoder(LearningRate = LearningRate)
    learn = theano.function([],updates = [(LearningRate,0.1*LearningRate)])
    
    E = []
    for i in range(50):
        batchTrain(variational_autoencoder,X_train)
        if i%5==0:
            E.append(np.mean(variational_autoencoder.ReconProb(X_test)))
        if i%20==0:
            learn()
    
if __name__=='__main__':
    main()