import lasagne
from lasagne.layers import Layer,MergeLayer
from lasagne.objectives import binary_crossentropy
import theano
import theano.tensor as T
import numpy as np
import BatchNormalization as BN
from lasagne.utils import as_tuple

class Upscale2DLayer(Layer):
    """
    2D upscaling layer
    Performs 2D upscaling over the two trailing axes of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.
    mode : {'repeat', 'dilate'}
        Upscaling mode: repeat element values or upscale leaving zeroes between
        upscaled elements. Default is 'repeat'.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    Using ``mode='dilate'`` followed by a convolution can be
    realized more efficiently with a transposed convolution, see
    :class:`lasagne.layers.TransposedConv2DLayer`.
    """

    def __init__(self, incoming, scale_factor, mode='repeat', **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

        if mode not in {'repeat', 'dilate'}:
            msg = "Mode must be either 'repeat' or 'dilate', not {0}"
            raise ValueError(msg.format(mode))
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        upscaled = input
        if self.mode == 'repeat':
            if b > 1:
                upscaled = T.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = T.extra_ops.repeat(upscaled, a, 2)
        elif self.mode == 'dilate':
            if b > 1 or a > 1:
                output_shape = self.get_output_shape_for(input.shape)
                upscaled = T.zeros(shape=output_shape, dtype=input.dtype)
                upscaled = T.set_subtensor(upscaled[:, :, ::a, ::b], input)
        return upscaled
        
class GaussianLayer(MergeLayer):

    def __init__(self, mean_network, log_sig_network, **kwargs):
        super(GaussianLayer, self).__init__([mean_network, log_sig_network], **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0])

    def get_output_for(self, inputs, **kwargs):
        mean,log_sigma2 = inputs
        e = self.rng.normal(size = mean.shape)
        z = mean + T.exp(0.5*log_sigma2)*e
        return z
        
    def get_log_p(self, x_network, deterministic=False, **kwargs):
        mean_network, log_sig_network = self.input_layers
        mean = lasagne.layers.get_output(mean_network,deterministic=deterministic)
        log_sigma2 = lasagne.layers.get_output(log_sig_network,deterministic=deterministic)
        x = lasagne.layers.get_output(x_network)
        axes = tuple(range(1,len(self.input_shapes[0])))
        
        log_coeff = -1*(0.5*T.log(2*np.pi)+0.5*log_sigma2)
        log_expon = -0.5*T.sqr(x-mean)/T.exp(10**-10+log_sigma2)
        log_P = (log_coeff+log_expon).sum(axis=axes)
        return log_P
        
class BinaryLayer(Layer):
    
    def __init__(self, sigmoid_network, **kwargs):
        super(BinaryLayer, self).__init__(sigmoid_network, **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shape):
        return input_shape
        
    def get_output_for(self, input, **kwargs):
        e = self.rng.uniform(size = input.shape)
        z = T.where((input-e)<0,0,1)
        return z
        
    def get_log_p(self, x_network, deterministic=False, **kwargs):
        out = lasagne.layers.get_output(self.input_layer,deterministic=deterministic)
        x = lasagne.layers.get_output(x_network)
        axes = tuple(range(1,len(self.input_shape)))
        
        log_P = -1*binary_crossentropy(out,x)
        return log_P.sum(axis=axes)
        
class Encoder_Class(object):
    def __init__(self, x):
        self.x = x
        self.BuildNetworks()
        self.KLDiv_qp = self.getKLDivergence()
        self.sample = self.getsample()
        
    def BuildNetworks(self):
        network = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=self.x,name='EncoderInput')
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name='Encoderh1')
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name='Encoderh2')
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)
        self.mean_network = lasagne.layers.DenseLayer(network,num_units=100,nonlinearity=lasagne.nonlinearities.linear,name='EncoderMean')
        self.mean_network = BN.batch_norm(self.mean_network)
        self.log_sig_network = lasagne.layers.DenseLayer(network,num_units=100,nonlinearity=lasagne.nonlinearities.linear,name='EncoderLogsig')
        self.log_sig_network = BN.batch_norm(self.log_sig_network)
        self.mean = lasagne.layers.get_output(self.mean_network)
        self.log_sigma2 = lasagne.layers.get_output(self.log_sig_network)
        
        #Calculating Z
        self.z_network = GaussianLayer(mean_network=self.mean_network,log_sig_network=self.log_sig_network,name='EncoderGauss')
        self.z = lasagne.layers.get_output(self.z_network)
        
    def getKLDivergence(self):
        KLD = 1+self.log_sigma2-T.sqr(self.mean)-T.exp(self.log_sigma2)
        return 0.5*KLD.sum(axis=1)
        
    def getsample(self):
        return theano.function([self.x],self.z,allow_input_downcast=True)
        
class Decoder_Class(object):
    def __init__(self, x, z_network,binary=False):
        self.z_network = z_network
        self.x = x
        self.binary = binary
        self.BuildNetworks()
        self.getP()
        self.sample = self.getsample()
        
    def BuildNetworks(self):
        network = lasagne.layers.DenseLayer(self.z_network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name='Decoderh1')
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name='Decoderh2')
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)
        
        if self.binary:
            sigmoid_network = lasagne.layers.DenseLayer(network,num_units=784,nonlinearity=lasagne.nonlinearities.sigmoid,name='DecoderSigmoid')
            sigmoid_network = BN.batch_norm(sigmoid_network)
            sigmoid_network = lasagne.layers.ReshapeLayer(sigmoid_network,([0],1,28,28))
            self.x_recon_network = BinaryLayer(sigmoid_network=sigmoid_network)
        else:
            mean_network = lasagne.layers.DenseLayer(network,num_units=784,nonlinearity=lasagne.nonlinearities.linear,name='DecoderMean')
            mean_network = BN.batch_norm(mean_network)
            mean_network = lasagne.layers.ReshapeLayer(mean_network,([0],1,28,28))
            
            log_sig_network = lasagne.layers.DenseLayer(network,num_units=784,nonlinearity=lasagne.nonlinearities.linear,name='DecoderSig')
            log_sig_network = BN.batch_norm(log_sig_network)
            log_sig_network = lasagne.layers.ReshapeLayer(log_sig_network,([0],1,28,28))
        
            #Calculating x_recon
            self.x_recon_network = GaussianLayer(mean_network=mean_network,log_sig_network=log_sig_network)
        
        self.x_recon = lasagne.layers.get_output(self.x_recon_network)
                
    def getP(self):
        self.log_P = self.x_recon_network.get_log_p(lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=self.x))
        
        self.log_P_det = self.x_recon_network.get_log_p(lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=self.x),deterministic=True)
    
    def generated_data(self, n_samples):
        z = np.random.randn(n_samples,100)
        return None
            
    def getsample(self):
        return theano.function([self.x],self.x_recon,allow_input_downcast=True)
            
class VariationalAutoEncoder():
    
    #Class Methods
    
    def __init__(self,Optimizer = 'adam',LearningRate = 0.0001, Momentum = 0.9,BatchNormalization = False,binary=False):
        
        #Class Members
        
        self.LearningRate = LearningRate
        self.Momentum = Momentum
        self.BatchNormalization = BatchNormalization
        
        #Initialization Optimization Function
        if Optimizer.lower() == 'sgd':
            self.Optimization = self.SGD
        elif Optimizer.lower() == 'rmsprop':
            self.Optimization = self.RMSProp
        elif Optimizer.lower() == 'adagrad':
            self.Optimization = self.AdaGrad
        elif Optimizer.lower() == 'adadelta':
            self.Optimization = self.AdaDelta
        elif Optimizer.lower() == 'adam':
            self.Optimization = self.Adam
        else:
            raise ValueError('Wrong Inputs for Optimizer')
            
        #Building Network
        self.x = T.tensor4('x')
        self.Encoder = Encoder_Class(self.x)
        self.z = self.Encoder.z
        self.Decoder = Decoder_Class(self.x,self.Encoder.z_network,binary=binary)
        self.x_recon = self.Decoder.x_recon
        
        self.params = self.getparams()
        self.Train = self.getTrainFunction() 
        self.log_P = theano.function([self.x],self.Decoder.log_P_det,allow_input_downcast=True)       
    
    #Get State Method
    def getstate(self):
        return lasagne.layers.get_all_param_values([self.Decoder.x_recon_network])
        
    #Set State Method
    def setstate(self,state):
        lasagne.layers.set_all_param_values([self.Decoder.x_recon_network],state)
    
    def getparams(self):
        return lasagne.layers.get_all_params([self.Decoder.x_recon_network],trainable=True)
            
    #Train Function
    def getTrainFunction(self):
        loss = -(self.Encoder.KLDiv_qp + self.Decoder.log_P)
        loss = loss.mean()
        updates = self.Optimization(loss,self.params)
        Train = theano.function([self.x],loss,updates=updates,allow_input_downcast=True)
        return Train
    
    #Reconstruction Probability
    def ReconProb(self,X):
        return self.log_P(X)
        
    #Optimization Algorithms
    def SGD(self,loss,params):
        return lasagne.updates.momentum(loss,params,self.LearningRate,self.Momentum)
        
    def RMSProp(self,loss,params):
        return lasagne.updates.rmsprop(loss,params,self.LearningRate)
        
    def AdaGrad(self,loss,params):
        return lasagne.updates.adagrad(loss,params,self.LearningRate)
        
    def AdaDelta(self,loss,params):
        return lasagne.updates.adadelta(loss,params)
        
    def Adam(self,loss,params):
        return lasagne.updates.adam(loss,params,self.LearningRate)