from tensorflow.examples.tutorials.mnist import input_data
from tensorpack import imgaug, dataset, ModelDesc, InputDesc, DataFlow
from tensorpack.callbacks.saver import *
from tensorpack.callbacks import *
from tensorpack.train.interface import *
from tensorpack.train import *
from tensorpack.utils import *
from tensorpack.models.utils import *
from tensorpack.tfutils import argscope
from tensorpack.models import *
from tensorpack.tfutils.common import *
from tensorpack.models.shapes import ConcatWith
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.predict import *
from tensorpack.tfutils.sessinit import SaverRestore
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d 
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args


from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset
from tensorpack.train import TrainConfig
import malis
import malis_loss
from malis_loss import *

def shape3d (a):
    if type(a) == int:
        return [a,a,a]
    if isinstance (a, (list, tuple)):
        assert len (a) == 3
        return list (a)
    raise RuntimeError("Illegal shape: {}".format(a))

def shape5d (a, data_format='NDHWC'):
    return [1] + shape3d (a) + [1]

@layer_register(log_shape=True)
def Conv3D (x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None,
           nl=tf.identity, split=1, use_bias=True,
           data_format='NDHWC'):
    in_shape = x.get_shape().as_list()
    channel_axis = 4
    in_channel = in_shape[channel_axis]
    
    assert in_channel is not None, "[Conv3D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0
    
    kernel_shape = shape3d (kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape5d(stride, data_format=data_format)

    
    if W_init is None:
        W_init = tf.variance_scaling_initializer(scale=2.0)
    if b_init is None:
        b_init = tf.constant_initializer()
    
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)
    
    assert split == 1
    conv = tf.nn.conv3d(x, W, stride, padding, data_format=data_format)

    ret = nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret

@layer_register(log_shape=True)
def DeConv3D(x, out_channel, kernel_shape,
             stride, padding='SAME',
             W_init=None, b_init=None,
             nl=tf.identity, use_bias=True,
             data_format='NDHWC'):
    
    in_shape = x.get_shape().as_list()
    channel_axis = 4
    in_channel = in_shape[channel_axis]
    
    assert in_channel is not None, "[DeConv3D] Input cannot have unknown channel!"
    assert isinstance(out_channel, int), out_channel
    
    if W_init is None:
        W_init = tf.variance_scaling_initializer(scale=2.0)
    if b_init is None:
        b_init = tf.constant_initializer()
    
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv3DTranspose(
            out_channel, kernel_shape,
            strides=stride, padding=padding,
            data_format='channels_last' if data_format == 'NDHWC' else 'channels_first',
            activation=lambda x: nl(x, name='output'),
            use_bias=use_bias,
            kernel_initializer=W_init,
            bias_initializer=b_init,
            trainable=True)
        ret = layer.apply(x, scope=tf.get_variable_scope())
        
    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
######################################################################################################

@layer_register()
def InstanceNorm3D(x, epsilon=1e-5, use_affine=True, gamma_init=None, data_format='NDHWC'):
    """
    Instance Normalization, as in the paper:
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`_.
    Args:
        x (tf.Tensor): a 4D tensor.
        epsilon (float): avoid divide-by-zero
        use_affine (bool): whether to apply learnable affine transformation
    """
    shape = x.get_shape().as_list()
    assert len(shape) == 5, "Input of InstanceNorm has to be 4D!"

    if data_format == 'NDHWC':
        axis = [1, 2, 3]
        ch = shape[4]
        new_shape = [1, 1, 1, 1, ch]
    else:
        axis = [2, 3, 4]
        ch = shape[1]
        new_shape = [1, ch, 1, 1, 1]
    assert ch is not None, "Input of InstanceNorm require known channel!"

    mean, var = tf.nn.moments(x, axis, keep_dims=True)

    if not use_affine:
        return tf.divide(x - mean, tf.sqrt(var + epsilon), name='output')

    beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)
    if gamma_init is None:
        gamma_init = tf.constant_initializer(1.0)
    gamma = tf.get_variable('gamma', [ch], initializer=gamma_init)
    gamma = tf.reshape(gamma, new_shape)
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')

######################################################################################################
def INReLU(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return LeakyReLU(x, name=name)

def INELU(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return tf.nn.elu (x, name=name)

def INELU2D(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.elu (x, name=name)

@layer_register(log_shape=True)
def residual(x, chan, kernel_shape=3):
    with argscope([Conv3D], nl=INELU, stride=1, kernel_shape=kernel_shape):
        input = x
        return (LinearWrap(x)
                .Conv3D('conv0', chan, padding='SAME', kernel_shape=(1,3,3))
                .Conv3D('conv1', chan/2, padding='SAME')
                .Conv3D('conv2', chan, padding='SAME', nl=tf.identity)
                # .InstanceNorm('inorm')
                ()) + input

@layer_register(log_shape=True)
def residual_enc(x, chan, kernel_shape=3):
    with argscope([Conv3D, DeConv3D], nl=INELU, stride=1, kernel_shape=kernel_shape):
        x = (LinearWrap(x)
            # .Dropout('drop', 0.75)
            .Conv3D('conv_i', chan, stride=(1, 2, 2))
            .residual('res_', chan, kernel_shape=kernel_shape)
            .Conv3D('conv_o', chan, stride=1, kernel_shape=(1,3,3)) 
            ())
        return x

@layer_register(log_shape=True)
def residual_dec(x, chan, kernel_shape=3):
    with argscope([Conv3D, DeConv3D], nl=INELU, stride=1, kernel_shape=kernel_shape):
                
        x = (LinearWrap(x)
            .DeConv3D('deconv_i', chan, stride=1, kernel_shape=(1,3,3)) 
            .residual('res2_', chan, kernel_shape=kernel_shape)
            .DeConv3D('deconv_o', chan, stride=(1, 2, 2)) 
            # .Dropout('drop', 0.75)
            ())
        return x

# @layer_register(log_shape=True)
# def dilation_convs_res (x, chan, nlayers, data_format="NDHWC"):
#     if data_format == "NDHWC":
#         x = tf.squeeze (x);
#     ret = x
#     with argscope([Conv2D], W_init=tf.truncated_normal_initializer(stddev=0.02), 
#         use_bias=False, nl=INELU2D, kernel_shape=3, padding='SAME'):
#         for l in range (nlayers):
#             x = Conv2D ('Dilation_' + str (l), x, chan, dilation_rate=2**l)
#             ret = ret + x
#     if data_format == "NDHWC":
#         ret = tf.expand_dims (ret, 0)
#     return ret

def cvt2sigm(x, name='ToRangeSigm'):
    with tf.variable_scope(name):
        return (x / 1.0 + 1.0) / 2.0

def wbce_malis (logits, affs, gt_affs, gt_seg, neighborhood, affs_shape, name='wbce_malis', limit_z=False):
    with tf.name_scope (name):

        pos_cnt = tf.cast (tf.count_nonzero (tf.cast (gt_affs, tf.int32)), tf.float32)
        neg_cnt = tf.cast (tf.constant (np.prod (affs_shape)), tf.float32) - pos_cnt
        pos_weight = neg_cnt / pos_cnt
        summary.add_tensor_summary (pos_weight, types=['scalar'])
        weighted_bce_losses = tf.nn.weighted_cross_entropy_with_logits (targets=gt_affs, logits=logits, pos_weight=pos_weight)

        malis_weights, pos_weights, neg_weights = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name='malis_weights', limit_z=limit_z)
        pos_weights = tf.identity (pos_weights, name='pos_weight')
        neg_weights = tf.identity (neg_weights, name='neg_weight')
        
        malis_weighted_bce_loss = tf.reduce_mean (tf.multiply (malis_weights, weighted_bce_losses), name='malis_weighted_bce_loss')

        return malis_weighted_bce_loss

def sqr_malis (logits, affs, gt_affs, gt_seg, neighborhood, affs_shape, name='sqr_malis'):
    with tf.name_scope (name):

        malis_weights, pos_weights, neg_weights = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name='malis_weights')
        pos_weights = tf.identity (pos_weights, name='pos_weight')
        neg_weights = tf.identity (neg_weights, name='neg_weight')
        
        loss = tf.reduce_mean (tf.multiply (malis_weights, tf.square (affs - gt_affs)))

        return loss

