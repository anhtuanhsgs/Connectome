import os, sys, argparse, glob

import numpy as np
import tensorflow as tf
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
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset
from tensorpack.train import TrainConfig
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe

from PIL import Image
import matplotlib.pyplot as plt
from img_aug_func import *

from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)

import malis
from funcs import *
import funcs
from funcs import *
from malis_loss import *
import time
# import tf_learn_func
#######################################################################################
input_shape = (16, 256, 256)
nhood = malis.mknhood3d (1)
affs_shape = (len(nhood),) + input_shape
NB_FILTERS = 32

tf_nhood = tf.constant (nhood)
#######################################################################################

np.random.seed (999)

class MyDataFlow (DataFlow):
    def __init__ (self, set_type, X, y):
        self.set_type = set_type
        self.volume = X
        self.gt_seg = y

        self.iter_per_epoch = 0
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        for i in range (len (self.volume)):
            self.iter_per_epoch += self.volume[i].shape[0] * self.volume[i].shape[1] * self.volume[i].shape[2] \
                    / (input_shape[0] * input_shape[1] * input_shape[2])

    def get_random_block (self, size):
        volume_idx = self.data_rand.randint (len (self.volume))
        volume = self.volume[volume_idx]
        gt_seg = self.gt_seg[volume_idx]

        x0_rand = self.data_rand.randint (volume.shape[1] - size[1] + 1)
        y0_rand = self.data_rand.randint (volume.shape[2] - size[2] + 1)
        z0_rand = self.data_rand.randint (volume.shape[0] - size[0] + 1)

        x0 = max (0, x0_rand); y0 = max (0, y0_rand); z0 = max (0, z0_rand)

        volume_patch = volume [z0:z0_rand + size[0],
                              x0:x0_rand + size[1],
                              y0:y0_rand + size[2]]
        volume_patch = volume_patch.astype (np.float32) / 255.0

        gt_seg_patch = gt_seg [z0:z0_rand + size[0],
                          x0:x0_rand + size[1],
                          y0:y0_rand + size[2]].copy ()

        return volume_patch, gt_seg_patch

    def data_dropout (self, block, labels, ndropout):
        droped = []
        for i in range (ndropout):
            x = self.data_rand.randint (len (block))
            while (x in droped):
                x = self.data_rand.randint (len (block))
            droped += [x]
        ret_block = []
        ret_labels = []
        for i in range (len (block)):
            if i in droped:
                continue
            ret_block.append (block[i])
            ret_labels.append (labels[i])
        ret_block, ret_labels = np.array (ret_block), np.array (ret_labels)
        return ret_block, ret_labels

    def sliding (self, imgs, x0, y0, size, max_range):
        n = self.data_rand.randint (len (imgs[0]))
        mode = self.data_rand.randint (2)
        
        def thres_hold (x):
            if (x < -max_range):
                x = -max_range
            if (x > max_range):
                x = max_range
            return x
        
        slide_dist = ( thres_hold (int (self.data_rand.normal (0, 10))), thres_hold (int (self.data_rand.normal (0, 10))))
        
        ret = []
        
        for img in imgs:
            x1 = x0 + slide_dist[0]
            y1 = y0 + slide_dist[1]
            if mode == 0:    
                img[n:, y0:y0+size[0], x0:x0+size[1]] = img[n:, y1:y1+size[0], x1:x1+size[1]]
            else:
                img[n, y0:y0+size[0], x0:x0+size[1]] = img[n, y1:y1+size[0], x1:x1+size[1]]
            ret.append (img[:,y0:y0+size[0],x0:x0+size[1]])
        return ret

    def get_data (self):
        ndropout = 3
        nblur = 3
        nblackout = 0
        max_range_slide = 18 * 2
        drop_1st = 1
        if self.set_type == 'train' or self.set_type == 'valid':
            drop_1st = 1
        for i in range (self.iter_per_epoch):
            if self.set_type == 'train' or self.set_type == 'valid':
                volume_patch, gt_seg_patch = self.get_random_block (size=(16+ndropout+drop_1st, 256 + max_range_slide, 256 + max_range_slide))
                # volume_patch, gt_seg_patch = self.get_random_block (size=(16+ndropout+drop_1st, 256, 256))
            else:
                volume_patch, gt_seg_patch = self.get_random_block (size=(16, 256, 256))

            if self.set_type == 'train' or self.set_type == 'valid':

                volume_patch, gt_seg_patch = self.sliding ([volume_patch, gt_seg_patch], 17, 17, (256, 256), max_range=17)

                seed = time_seed ()
                volume_patch, gt_seg_patch = apply_aug (volume_patch, gt_seg_patch, func=random_flip, seed=seed)
                volume_patch, gt_seg_patch = apply_aug (volume_patch, gt_seg_patch, func=random_reverse, seed=seed)
                volume_patch, gt_seg_patch = apply_aug (volume_patch, gt_seg_patch, func=random_square_rotate, seed=seed)
                volume_patch, gt_seg_patch = apply_aug (volume_patch, gt_seg_patch, func=random_elastic, seed=seed)
                volume_patch, gt_seg_patch = self.data_dropout (volume_patch, gt_seg_patch, ndropout=ndropout)
                if self.set_type == 'train' or self.set_type == 'valid':
                    # volume_patch = random_blackout (volume_patch, nblackout, self.data_rand)
                    volume_patch = random_gaussian_blur (volume_patch, nblur)

            gt_affs_patch = malis.seg_to_affgraph (gt_seg_patch, nhood)
            if self.set_type == 'train' or self.set_type == 'valid':
                yield [volume_patch[drop_1st:], gt_seg_patch[drop_1st:], gt_affs_patch[:,drop_1st:]]
            else:
                yield [volume_patch, gt_seg_patch, gt_affs_patch]


    def size (self):
        if self.set_type == 'valid':
            return self.iter_per_epoch / 2    
        return 200

class Model (ModelDesc):
    def __init__ (self):
        self.name = os.path.basename(__file__)
        self.name = self.name [:len (self.name) - 3]

    def _get_inputs (self):
        return [InputDesc(tf.float32, input_shape, 'volume'),
                InputDesc(tf.int32, input_shape, 'gt_seg'),
                InputDesc(tf.float32, affs_shape, 'gt_affs')]

    @auto_reuse_variable_scope
    def generator (self, volume):
        return tf_learn_func.arch_generator (volume, NB_FILTERS)
    
    def _build_graph (self, inputs):
        volume, gt_seg, gt_affs = inputs
        volume = tf.expand_dims (volume, 3)
        volume = tf.expand_dims (volume, 0)
#         image = image * 2 - 1
        
        with argscope(LeakyReLU, alpha=0.2),\
            argscope ([Conv3D, DeConv3D], use_bias=False, 
                kernel_shape=3, stride=2, padding='SAME',
                W_init=tf.contrib.layers.variance_scaling_initializer(factor=0.333, uniform=True)):
            _in = Conv3D ('in', volume, NB_FILTERS, kernel_shape=(3, 5, 5), stride=1, padding='SAME', nl=INELU, use_bias=True)
            e0 = residual_enc('e0', volume, NB_FILTERS*1) 
            e1 = residual_enc('e1',  e0, NB_FILTERS*2)
            e2 = residual_enc('e2',  e1, NB_FILTERS*4, kernel_shape=(1, 3, 3))           
            e3 = residual_enc('e3',  e2, NB_FILTERS*8, kernel_shape=(1, 3, 3))

            e3 = Dropout('dr', e3, rate=0.5)
            
            d3 = residual_dec('d3', e3 + e3, NB_FILTERS*4, kernel_shape=(1, 3, 3))
            d2 = residual_dec('d2', d3 + e2, NB_FILTERS*2, kernel_shape=(1, 3, 3))
            d1 = residual_dec('d1', d2 + e1, NB_FILTERS*1)
            d0 = residual_dec('d0', d1 + e0, NB_FILTERS*1)

            logits = funcs.Conv3D ('x_out', d0, len (nhood), kernel_shape=(3, 5, 5), stride=1, padding='SAME', nl=tf.identity, use_bias=True)

        logits = tf.squeeze (logits)
        logits = tf.transpose (logits, perm=[3,0,1,2], name='logits')

        affs = cvt2sigm (tf.tanh (logits))
        affs = tf.identity (affs, name='affs')

        ###################################################################################################################### 
        
        wbce_malis_loss = wbce_malis (logits, affs, gt_affs, gt_seg, nhood, affs_shape, name='wbce_malis', limit_z=False)
        wbce_malis_loss = tf.identity (wbce_malis_loss, name='wbce_malis_loss')

        dice_loss = tf.identity ((1. - dice_coe (affs, gt_affs, axis=[0,1,2,3], loss_type='jaccard')) * 0.1, name='dice_coe')
        tot_loss = tf.identity (wbce_malis_loss + dice_loss, name='tot_loss')
        ###################################################################################################################### 

        self.cost = tot_loss
        summary.add_tensor_summary (tot_loss, types=['scalar'])
        summary.add_tensor_summary (dice_loss, types=['scalar'])
        summary.add_tensor_summary (wbce_malis_loss, types=['scalar'])

    def _get_optimizer (self):
        lr = get_scalar_var('learning_rate', 5e-6, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

