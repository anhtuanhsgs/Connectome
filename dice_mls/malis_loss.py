import tensorflow as tf
import numpy as np
from malis import nodelist_like, malis_loss_weights
import malis

class MalisWeights(object):

    def __init__(self, output_shape, neighborhood, limit_z):
        self.output_shape = np.asarray(output_shape)
        self.neighborhood = np.asarray(neighborhood)
        self.limit_z = limit_z
        self.edge_list = nodelist_like(self.output_shape, self.neighborhood)

    def get_edge_weights(self, affs, gt_affs, gt_seg):

        assert affs.shape[0] == len(self.neighborhood)

        weights_neg, neg_npairs = self.malis_pass(affs, gt_affs, gt_seg, 1., 1., 1., pos=0)
        weights_pos, pos_npairs = self.malis_pass(affs, gt_affs, gt_seg, 1., 1., 1., pos=1)

        weights_pos += 1
        weights_neg += 1

        weights_neg_scaled = np.log2 (weights_neg)
        weights_pos_scaled = np.log2 (weights_pos)

        if (self.limit_z):
            weights_neg_scaled [0] *= 3

        ret =  weights_neg_scaled + weights_pos_scaled

        return ret , weights_pos_scaled, weights_neg_scaled

    def malis_pass(self, affs, gt_affs, gt_seg, z_scale, y_scale, x_scale, pos):

        # create a copy of the affinities and change them, such that in the
        #   positive pass (pos == 1): affs[gt_affs == 0] = 0
        #   negative pass (pos == 0): affs[gt_affs == 1] = 1

        pass_affs = np.copy(affs)
        pass_affs[gt_affs == (1 - pos)] = (1 - pos)

        pass_affs[0,:,:,:] *= z_scale
        pass_affs[1,:,:,:] *= y_scale
        pass_affs[2,:,:,:] *= x_scale

        weights = malis_loss_weights(
            gt_seg.astype(np.uint64).flatten(),
            self.edge_list[0].flatten(),
            self.edge_list[1].flatten(),
            pass_affs.astype(np.float32).flatten(),
            pos)

        weights = weights.reshape((-1,) + tuple(self.output_shape))
        assert weights.shape[0] == len(self.neighborhood)

        num_pairs = np.sum(weights, dtype=np.uint64)
        # print num_pairs
        # '1-pos' samples don't contribute in the 'pos' pass
        weights[gt_affs == (1 - pos)] = 0
        # print num_pairs, np.sum(weights, dtype=np.uint64), pos, np.sum (weights < 0)
        # print np.max (weights)
        # normalize
        weights = weights.astype(np.float32)
        num_pairs = np.sum(weights)
        # if num_pairs > 0:
        #     weights = weights/num_pairs

        return weights, num_pairs

def malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name=None, limit_z=False):
    '''Returns a tensorflow op to compute just the weights of the MALIS loss.
    This is to be multiplied with an edge-wise base loss and summed up to create
    the final loss. For the Euclidean loss, use ``malis_loss_op``.
    Args:
        affs (Tensor): The predicted affinities.
        gt_affs (Tensor): The ground-truth affinities.
        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background.
        neighborhood (Tensor): A list of spacial offsets, defining the
            neighborhood for each voxel.
        name (string, optional): A name to use for the operators created.
    Returns:
        A tensor with the shape of ``affs``, with MALIS weights stored for each
        edge.
    '''

    output_shape = gt_seg.get_shape().as_list()

    malis_weights = MalisWeights(output_shape, neighborhood, limit_z)
    malis_functor = lambda affs, gt_affs, gt_seg, mw=malis_weights: \
        mw.get_edge_weights(affs, gt_affs, gt_seg)

    weights = tf.py_func(
        malis_functor,
        [affs, gt_affs, gt_seg],
        [tf.float32, tf.float32, tf.float32],
        name=name)
    # print weights
    return weights

def malis_loss_op(affs, gt_affs, gt_seg, neighborhood, name=None):
    '''Returns a tensorflow op to compute the MALIS loss, using the squared
    distance to the target values for each edge as base loss.
    Args:
        affs (Tensor): The predicted affinities.
        gt_affs (Tensor): The ground-truth affinities.
        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background.
        neighborhood (Tensor): A list of spacial offsets, defining the
            neighborhood for each voxel.
        name (string, optional): A name to use for the operators created.
    Returns:
        A tensor with one element, the MALIS loss.
    '''

    weights, pos_weights, neg_weights = malis_weights_op(affs, gt_affs, gt_seg, neighborhood, name='malis_weights')
    edge_loss = tf.square(tf.subtract(gt_affs, affs))
    
    return tf.reduce_sum (tf.multiply(weights, edge_loss), name='malis_loss')

def one_hot_affs (affs):
    affs_0 = 1 - affs
    return np.hstack (affs_0, affs)


def mknhood_long ():
    ret = malis.mknhood3d (1).tolist ()
    for dz in [2, 3, 4]:
        ret.append ([-dz, 0, 0])
    for dy in [3, 9, 27]:
        ret.append ([0, -dy, 0])
    for dx in [3, 9, 27]:
        ret.append ([0, 0, -dx])
    return np.array (ret, dtype=np.int32) 