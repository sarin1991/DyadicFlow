import lasagne
from lasagne.layers import Layer,MergeLayer
from lasagne.objectives import binary_crossentropy
import theano
import theano.tensor as T
import numpy as np
import BatchNormalization as BN
from lasagne.utils import as_tuple
from collections import OrderedDict
from theano.compile.nanguardmode import NanGuardMode

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

class DyadicFlowLayer(MergeLayer):
    """The inputs are expected to have first dimension as batch size, second as rank and the rest are
       are corresponding to the shape in y. These are flattened and operated on."""

    def __init__(self, y_network, u_network, v_network, epsilon = 10**-3, **kwargs):
        self.epsilon = epsilon
        super(DyadicFlowLayer, self).__init__([y_network, u_network, v_network], **kwargs)
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0])
            
    def get_output_for(self, inputs, **kwargs):
        y, u, v = inputs
        yshape, ushape, vshape = self.input_shapes
        
        y_flattened = y.flatten(2).dimshuffle(0,1,'x')
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        
        input_projection = T.batched_dot(v_flattened,y_flattened)
        Output_pertubration = T.batched_dot(u_flattened.dimshuffle(0,2,1),input_projection)
        x = y + self.epsilon*Output_pertubration.reshape(y.shape)
        return x
    
    def getDet(self):
        y_network, u_network, v_network = self.input_layers
        u = lasagne.layers.get_output(u_network)
        v = lasagne.layers.get_output(v_network)

        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        basis_modifier = T.batched_dot(v_flattened,u_flattened.dimshuffle(0,2,1))
        basis_identity = T.identity_like(basis_modifier[0]).dimshuffle('x',0,1)
        basis_transform = basis_identity+self.epsilon*basis_modifier
        #Determinant Calculation
        #determinant_func = lambda basis_transform_mat: T.nlinalg.det(basis_transform_mat)
        #determinant,updates = theano.scan(determinant_func,sequences=basis_transform)
        
        #Robust Determinant Calc
        determinant = 1 + self.epsilon*((v_flattened*u_flattened).sum(axis=(1,2)))
        updates = {}
        
        return determinant,updates
        
    def getInv(self, x):
        y_network, u_network, v_network = self.input_layers
        u = lasagne.layers.get_output(u_network)
        v = lasagne.layers.get_output(v_network)
        
        x_flattened = x.flatten(2).dimshuffle(0,1,'x')
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        basis_modifier = T.batched_dot(v_flattened,u_flattened.dimshuffle(0,2,1))
        basis_identity = T.identity_like(basis_modifier[0]).dimshuffle('x',0,1)
        basis_transform = basis_identity+self.epsilon*basis_modifier
        #Inverse Calculation
        #inverse_func = lambda basis_transform_mat: T.nlinalg.matrix_inverse(basis_transform_mat)
        #inverse,updates = theano.scan(inverse_func,sequences=basis_transform)
        
        #Robust Inverse Calc
        inverse = basis_identity - self.epsilon*basis_modifier
        updates = {}
        
        input_projection = T.batched_dot(v_flattened,x_flattened)
        inverse_adjustment = T.batched_dot(inverse,input_projection)
        output_modification = T.batched_dot(u_flattened.dimshuffle(0,2,1),inverse_adjustment)
        y_inv = x - self.epsilon*output_modification.reshape(x.shape)
        return y_inv,updates
    
    def getTrace(self):
        y_network, u_network, v_network = self.input_layers
        mean_network, log_sig_network = y_network.input_layers
        log_sig2 = lasagne.layers.get_output(log_sig_network)
        u = lasagne.layers.get_output(u_network)
        v = lasagne.layers.get_output(v_network)
        log_sig2_flattened = log_sig2.flatten(2)
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        
        secondOrderProduct = (self.epsilon**2)*(v_flattened*T.batched_dot(T.batched_dot(u_flattened,u_flattened.dimshuffle(0,2,1)),v_flattened)).sum(axis=1)
        firstOrderProduct = self.epsilon*(2*v_flattened*u_flattened).sum(axis=1)
        
        trace = (T.exp(log_sig2_flattened)*(1+firstOrderProduct+secondOrderProduct)).sum(axis=1)
        return trace
        
    def getKLD(self):
        y_network, u_network, v_network = self.input_layers
        mean_network, log_sig_network = y_network.input_layers
        mean = lasagne.layers.get_output(mean_network)
        log_sig2 = lasagne.layers.get_output(log_sig_network)
        u = lasagne.layers.get_output(u_network)
        v = lasagne.layers.get_output(v_network)
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        mean_flattened = mean.flatten(2).dimshuffle(0,1,'x')
        log_sig2_flattened = log_sig2.flatten(2)
        
        trace = self.getTrace()
        
        input_projection = T.batched_dot(v_flattened,mean_flattened)
        Output_pertubration = T.batched_dot(u_flattened.dimshuffle(0,2,1),input_projection)
        mean_new = mean_flattened + self.epsilon*Output_pertubration
        mean_inner_product = T.batched_dot(mean_new.dimshuffle(0,2,1),mean_new).flatten(1)
        
        determinant,updates = self.getDet()
        det_abs = T.abs_(determinant)
        ln_det = 2*T.log(T.maximum(10**-10,det_abs)) + log_sig2_flattened.sum(axis=1)
        
        KLD = 0.5*(trace + mean_inner_product - log_sig2_flattened.shape[1] - ln_det)        
        return KLD,updates
               
    def get_log_p(self, x_network, deterministic=False, **kwargs):
        y_network, u_network, v_network = self.input_layers
        yshape, ushape, vshape = self.input_shapes
        x = lasagne.layers.get_output(x_network)
        
        y_inv,updates = self.getInv(x)
        y_inv_network = lasagne.layers.InputLayer(shape=yshape,input_var=y_inv)
        
        log_P_y,_ = y_network.get_log_p(y_inv_network)
        
        determinant, updates_det = self.getDet()
        updates.update(updates_det)
        det_abs = T.abs_(determinant)
        
        log_P = log_P_y - T.log(T.maximum(10**-10,det_abs))
        return log_P,updates
        
class GaussianSampleLayer(MergeLayer):
    
    def __init__(self, mean_network, log_sig_network, **kwargs):
        super(GaussianSampleLayer, self).__init__([mean_network, log_sig_network], **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0])

    def get_output_for(self, inputs, **kwargs):
        mean,log_sigma2 = inputs
        e = self.rng.normal(size = mean.shape)
        z = mean + T.exp(0.5*log_sigma2)*e
        return z
        
class GaussianLogProbabilityLayer(MergeLayer):
    
    def __init__(self, mean_network, log_sig_network, x_target_network, **kwargs):
        super(GaussianLogProbabilityLayer, self).__init__([mean_network, log_sig_network, x_target_network], **kwargs)
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0][0])

    def get_output_for(self, inputs, **kwargs):
        mean,log_sigma2,x_target = inputs
        axes = tuple(range(1,len(self.input_shapes[0])))
        
        log_coeff = -1*(0.5*T.log(2*np.pi)+0.5*log_sigma2)
        log_expon = -0.5*T.sqr(x-mean)/T.exp(10**-10+log_sigma2)
        log_P = (log_coeff+log_expon).sum(axis=axes)
        return log_P
        
class GaussianKLDLayer(MergeLayer):
    
    def __init__(self, mean_network, log_sig_network, **kwargs):
        super(GaussianKLDLayer, self).__init__([mean_network, log_sig_network], **kwargs)
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0][0])

    def get_output_for(self, inputs, **kwargs):
        mean, log_sigma2 = inputs
        KLD = 1+log_sigma2-T.sqr(mean)-T.exp(T.clip(log_sigma2,-10,10))
        return -0.5*KLD.flatten(2).sum(axis=1)
        
class DyadicFlowSampleLayer(MergeLayer):
    
    def __init__(self, y_network, u_network, v_network, epsilon = 10**-3, **kwargs):
        super(DyadicFlowSampleLayer, self).__init__([y_network, u_network, v_network], **kwargs)
        self.epsilon = epsilon
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0])

    def get_output_for(self, inputs, **kwargs):
        y,u,v = inputs
        y_flattened = y.flatten(2).dimshuffle(0,1,'x')
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        
        input_projection = T.batched_dot(v_flattened,y_flattened)
        Output_pertubration = T.batched_dot(u_flattened.dimshuffle(0,2,1),input_projection)
        x = y + self.epsilon*Output_pertubration.reshape(y.shape)
        return x
        
class DyadicFlowKLDLayer(MergeLayer):
    
    def __init__(self, mean_network, log_sig_network, u_network, v_network, epsilon = 10**-3, **kwargs):
        super(DyadicFlowKLDLayer, self).__init__([mean_network, log_sig_network, u_network, v_network], **kwargs)
        self.epsilon = epsilon
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0][0])
    
    def getDet(self, u_flattened, v_flattened):
        basis_modifier = T.batched_dot(v_flattened,u_flattened.dimshuffle(0,2,1))
        basis_identity = T.identity_like(basis_modifier[0]).dimshuffle('x',0,1)
        basis_transform = basis_identity+self.epsilon*basis_modifier
        #Determinant Calculation
        determinant_func = lambda basis_transform_mat: T.nlinalg.det(basis_transform_mat)
        determinant,_ = theano.scan(determinant_func,sequences=basis_transform)
        return determinant
        
    def getTrace(self, log_sigma2_flattened, u_flattened, v_flattened):        
        secondOrderProduct = (self.epsilon**2)*(v_flattened*T.batched_dot(T.batched_dot(
                              u_flattened,u_flattened.dimshuffle(0,2,1)),v_flattened)).sum(axis=1)
        firstOrderProduct = self.epsilon*(2*v_flattened*u_flattened).sum(axis=1)
        
        trace = (T.exp(T.maximum(log_sigma2_flattened,-10**7))*(1+firstOrderProduct+secondOrderProduct)).sum(axis=1)
        return trace
        
    def get_output_for(self, inputs, **kwargs):
        mean,log_sigma2,u,v = inputs
        mean_flattened = mean.flatten(2).dimshuffle(0,1,'x')
        log_sigma2_flattened = log_sigma2.flatten(2)
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        
        trace = self.getTrace(log_sigma2_flattened, u_flattened, v_flattened)
        
        input_projection = T.batched_dot(v_flattened,mean_flattened)
        Output_pertubration = T.batched_dot(u_flattened.dimshuffle(0,2,1),input_projection)
        mean_new = mean_flattened + self.epsilon*Output_pertubration
        mean_inner_product = T.batched_dot(mean_new.dimshuffle(0,2,1),mean_new).flatten(1)
        
        determinant = self.getDet(u_flattened, v_flattened)
        det_abs = T.abs_(determinant)
        ln_det = 2*T.log(T.maximum(10**-10,det_abs)) + log_sigma2_flattened.sum(axis=1)
        
        KLD = 0.5*(trace + mean_inner_product - log_sigma2_flattened.shape[1] - ln_det)        
        return KLD
        
class DyadicFlowLogProbabilityLayer(MergeLayer):
    
    def __init__(self, mean_network, log_sig_network, u_network, v_network, x_target_network, epsilon = 10**-3, **kwargs):
        super(DyadicFlowLogProbabilityLayer, self).__init__([mean_network, log_sig_network, u_network, v_network, x_target_network], **kwargs)
        self.epsilon = epsilon
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0][0])
    
    def getGaussianKLD(self,mean,log_sigma2,y_inv):
        axes = tuple(range(1,len(self.input_shapes[0])))
        log_coeff = -1*(0.5*T.log(2*np.pi)+0.5*log_sigma2)
        log_expon = -0.5*T.sqr(y_inv-mean)/T.exp(T.maximum(log_sigma2,-10**7))
        log_P = (log_coeff+log_expon).sum(axis=axes)
        return log_P
        
    def getInv(self, u_flattened, v_flattened, x_target_flattened, mean):
        basis_modifier = T.batched_dot(v_flattened,u_flattened.dimshuffle(0,2,1))
        basis_identity = T.identity_like(basis_modifier[0]).dimshuffle('x',0,1)
        basis_transform = basis_identity+self.epsilon*basis_modifier
        #Inverse Calculation
        inverse_func = lambda basis_transform_mat: T.nlinalg.matrix_inverse(basis_transform_mat)
        inverse,_ = theano.scan(inverse_func,sequences=basis_transform)
        
        input_projection = T.batched_dot(v_flattened,x_target_flattened)
        inverse_adjustment = T.batched_dot(inverse,input_projection)
        output_modification = T.batched_dot(u_flattened.dimshuffle(0,2,1),inverse_adjustment)
        y_inv = (x_target_flattened - self.epsilon*output_modification).reshape(mean.shape)
        return y_inv
        
    def getDet(self, u_flattened, v_flattened):
        basis_modifier = T.batched_dot(v_flattened,u_flattened.dimshuffle(0,2,1))
        basis_identity = T.identity_like(basis_modifier[0]).dimshuffle('x',0,1)
        basis_transform = basis_identity+self.epsilon*basis_modifier
        #Determinant Calculation
        determinant_func = lambda basis_transform_mat: T.nlinalg.det(basis_transform_mat)
        determinant,_ = theano.scan(determinant_func,sequences=basis_transform)
        return determinant
            
    def get_output_for(self, inputs, **kwargs):
        mean,log_sigma2,u,v,x_target = inputs
        x_target_flattened = x_target.flatten(2).dimshuffle(0,1,'x')
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        
        y_inv = self.getInv(u_flattened,v_flattened,x_target_flattened,mean)
        log_P_y = self.getGaussianKLD(mean,log_sigma2,y_inv)
        
        determinant = self.getDet(u_flattened, v_flattened)
        det_abs = T.abs_(determinant)
        
        log_P = log_P_y - T.log(T.maximum(10**-10,det_abs))
        return log_P
        
class BinarySampleLayer(Layer):
    
    def __init__(self, sigmoid_network, **kwargs):
        super(BinarySampleLayer, self).__init__(sigmoid_network, **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shape):
        return input_shape
        
    def get_output_for(self, input, **kwargs):
        e = self.rng.uniform(size = input.shape)
        z = T.where((input-e)<0,0,1)
        return z
        
class BinaryLogProbabilityLayer(MergeLayer):
    
    def __init__(self, sigmoid_network, x_target_network, **kwargs):
        super(BinaryLogProbabilityLayer, self).__init__([sigmoid_network, x_target_network], **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0]
        
    def get_output_for(self, inputs, **kwargs):
        prediction, target = inputs
        prediction_stable = prediction.clip(10**-5,1-10**-5)
        log_P = -1*binary_crossentropy(prediction_stable, target)
        return log_P.flatten(2).sum(axis=1)

class RBMSampleLayer(Layer):
    
    def __init__(self, sigmoid_network, num_samples, **kwargs):
        super(RBMSampleLayer, self).__init__(sigmoid_network, **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        self.num_samples = num_samples
        
    def get_output_shape_for(self, input_shape):
        return input_shape
        
    def get_output_for(self, input, **kwargs):
        x_shape = self.input_shape
        x_shape = tuple([self.num_samples,input.shape[0]]+list(x_shape[1:]))
        e = self.rng.uniform(size = x_shape)
        x = T.where((input.dimshuffle(tuple(['x']+range(len(self.input_shape))))-e)<0,0,1)
        return x
        
class RBMSigmoidLayer(MergeLayer):
    
    def __init__(self, y_network, u_network, v_network, b_network, **kwargs):
        super(RBMSigmoidLayer, self).__init__([y_network, u_network, v_network, b_network], **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams()
        
    def get_output_shape_for(self, input_shapes):
        return tuple(input_shapes[0])
        
    def get_output_for(self, inputs, **kwargs):
        y,u,v,b = inputs
        y_shape, u_shape, v_shape, b_shape = self.input_shapes
        x_shape = list(y_shape[1:])
        x_shape[0] = y.shape[1]
        u_flattened = u.flatten(3)
        v_flattened = v.flatten(3)
        y_flattened = y.flatten(3).dimshuffle(1,2,0)
        b_flattened = b.flatten(2).dimshuffle(0,1,'x')
        Energy = b_flattened + T.batched_dot(u_flattened.dimshuffle(0,2,1),T.batched_dot(v_flattened,y_flattened))
        x_prob = 1/(1+T.exp(-Energy.clip(-20,20)))
        return x_prob.mean(axis=2).reshape(tuple(x_shape))
                        
class Encoder_Class(object):
    def __init__(self, x, k=1):
        self.x = x
        self.k = k
        self.BuildNetworks()
        self.sample = self.getsample()
        
    def BuildNetworks(self):
        network = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=self.x)
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = BN.batch_norm(network)
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = BN.batch_norm(network)
        mean_network = lasagne.layers.DenseLayer(network,num_units=50,nonlinearity=lasagne.nonlinearities.linear)
        mean_network = BN.batch_norm(mean_network)
        log_sig_network = lasagne.layers.DenseLayer(network,num_units=50,nonlinearity=lasagne.nonlinearities.linear)
        log_sig_network = BN.batch_norm(log_sig_network)
        
        u_network = lasagne.layers.DenseLayer(network,num_units=self.k*50,nonlinearity=lasagne.nonlinearities.linear)
        u_network = BN.batch_norm(u_network)
        u_network = lasagne.layers.ReshapeLayer(u_network,([0],self.k,50))
        v_network = lasagne.layers.DenseLayer(network,num_units=self.k*50,nonlinearity=lasagne.nonlinearities.linear)
        v_network = BN.batch_norm(v_network)
        v_network = lasagne.layers.ReshapeLayer(v_network,([0],self.k,50))
        
        #Calculating Z
        y_network = GaussianSampleLayer(mean_network=mean_network,log_sig_network=log_sig_network)
        self.z_network = DyadicFlowSampleLayer(y_network = y_network, u_network = u_network, v_network = v_network)
        self.KLD_network = DyadicFlowKLDLayer(mean_network = mean_network, log_sig_network = log_sig_network, u_network = u_network, v_network = v_network)
        
    def getsample(self):
        z = lasagne.layers.get_output(self.z_network)
        return theano.function([self.x],z,allow_input_downcast=True)
        
class Decoder_Class(object):
    def __init__(self, x, z_network, k=1):
        self.z_network = z_network
        self.x = x
        self.k = k
        self.BuildNetworks()
        self.sample = self.getsample()
        
    def BuildNetworks(self):
        #Inputnetwork
        self.x_target = T.tensor4()
        self.x_target_network = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var = self.x_target)
        
        network = lasagne.layers.DenseLayer(self.z_network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = BN.batch_norm(network)
        network = lasagne.layers.DropoutLayer(network)

        mean_network = lasagne.layers.DenseLayer(network,num_units=784,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        mean_network = BN.batch_norm(mean_network)
        mean_network = lasagne.layers.ReshapeLayer(mean_network,([0],1,28,28))
        
        log_sig_network = lasagne.layers.DenseLayer(network,num_units=784,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        log_sig_network = BN.batch_norm(log_sig_network)
        log_sig_network = lasagne.layers.ReshapeLayer(log_sig_network,([0],1,28,28))
        
        #Calculating y
        y_network = GaussianSampleLayer(mean_network=mean_network,log_sig_network=log_sig_network)
        
        u_network = lasagne.layers.DenseLayer(network,num_units=self.k*784,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        u_network = BN.batch_norm(u_network)
        u_network = lasagne.layers.ReshapeLayer(u_network,([0],self.k,28,28))
        
        v_network = lasagne.layers.DenseLayer(network,num_units=self.k*784,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify)
        v_network = BN.batch_norm(v_network)
        v_network = lasagne.layers.ReshapeLayer(v_network,([0],self.k,28,28))
        
        #Calculating x_recon
        self.x_recon_network = DyadicFlowSampleLayer(y_network = y_network, u_network = u_network, v_network = v_network)
        self.x_log_P_network = DyadicFlowLogProbabilityLayer(mean_network = mean_network,log_sig_network = log_sig_network, u_network = u_network, 
                                                             v_network = v_network, x_target_network = self.x_target_network)

    def generated_data(self, n_samples):
        z = np.random.randn(n_samples,2)
        return self.sample(z)
            
    def getsample(self):
        x_recon = lasagne.layers.get_output(self.x_recon_network)
        return theano.function([self.x],x_recon,allow_input_downcast=True)
            
class VariationalAutoEncoder():
    
    #Class Methods
    
    def __init__(self,Optimizer = 'adam',LearningRate = 0.0001, Momentum = 0.9):
        
        #Class Members
        
        self.LearningRate = LearningRate
        self.Momentum = Momentum
        
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
        self.Encoder = Encoder_Class(self.x, k = 20)
        self.Decoder = Decoder_Class(self.x,self.Encoder.z_network, k = 5)
        
        self.params = self.getparams()
        self.theano_train = self.getTrainFunction() 
        self.log_P = theano.function([self.x,self.Decoder.x_target],lasagne.layers.get_output(self.Decoder.x_log_P_network),allow_input_downcast=True)       
    
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
        KLD, log_P = lasagne.layers.get_output([self.Encoder.KLD_network,self.Decoder.x_log_P_network])
        loss = KLD - log_P
        loss = loss.mean()
        updates = self.Optimization(loss,self.params)
        Train = theano.function([self.x,self.Decoder.x_target],loss,updates=updates,allow_input_downcast=True,mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        return Train
        
    #Train Function
    def Train(self,X):
        return self.theano_train(X,X)
    
    #Reconstruction Probability
    def ReconProb(self,X,BS = 128):
        P = []
        NB = len(X)/BS
        for j in range(NB+1):
            x = X[j*BS:(j+1)*BS]
            P.extend(list(self.log_P(x,x)))
        return P
        
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