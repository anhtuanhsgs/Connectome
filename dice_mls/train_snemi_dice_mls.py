import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, sys, argparse, glob
#################################################################################
from tensorpack import *
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
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset
from tensorpack.train import TrainConfig
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
#################################################################################
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io
#################################################################################
import malis
from funcs import *
from model_snemi_dice_mls import *
#################################################################################


def read_im (paths):
    ret = []
    for path in paths:
        ret.append (io.imread (path))
    return ret

def get_data (X_train, X_valid, y_train, y_valid):
    train = PrefetchDataZMQ (MyDataFlow ('train', X_train, y_train), 4)
    # train = MyDataFlow ('train', X_train, y_train)
    valid = MyDataFlow ('valid', X_valid, y_valid)
    return train, valid

def get_config (X_train, X_valid, y_train, y_valid, model_path = None):
    data_train, data_valid = get_data (X_train, X_valid, y_train, y_valid)

    steps_per_epoch = data_train.size ()

    cur_loss = 'tot_loss'
    triggerk = 15
    visualk = 5
    config = TrainConfig(
        model=Model(),
        dataflow=data_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=triggerk),
            PeriodicTrigger(MinSaver('validation_' + cur_loss), every_k_epochs=triggerk),
            
            ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (150, 1e-4), (300, 5e-5), (600, 1e-5), (800, 1e-6)], interp='linear'),
            PeriodicTrigger(VisualizeRunner(), every_k_epochs = visualk),
            PeriodicTrigger(InferenceRunner(data_valid, [ScalarStats(cur_loss)]), every_k_epochs=5)
        ],
        session_init = SaverRestore (model_path) if model_path != None else None,
        # session_config=session_config,
        steps_per_epoch=data_train.size (),
        max_epoch=2000
    )
    return config

def sgm2img (imgs_sets):
    ret = []
    for imgs in imgs_sets:
        imgs = np.squeeze (imgs)
        imgs *= 255
        ret.append (imgs.astype (np.int32))
    return ret

class VisualizeRunner(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['volume', 'gt_seg', 'gt_affs'], ['affs', 'gt_affs', 'volume', 'wbce_malis/pos_weight', 'wbce_malis/neg_weight'])

    def _before_train(self):
        global args
        self.valid_ds = MyDataFlow ('valid', X_valid[:], y_valid[:])
        self.test_ds = MyDataFlow ('test', X_test[:], X_test[:])

    def _trigger(self):
        for valid_vol in self.valid_ds.get_data():
            affs, gt_affs, volume, wp, wn = self.pred(valid_vol)
            affs, gt_affs, volume, wp, wn = sgm2img ([affs, gt_affs, volume, wp, wn])

            #print affs.shape
            for i in range (volume.shape[0]):
                concated_img = [volume[i], gt_affs[0,i], affs[0,i], gt_affs[2,i], affs[2,i]]
                concated_img = np.concatenate (concated_img, 1)
                concated_w = [volume[i], wp[0,i], wp[2,i], wn[0,i], wn[2,i]]
                concated_w = np.concatenate (concated_w, 1)
                concated_img = np.concatenate ([concated_img, concated_w], 0)
                self.trainer.monitors.put_image ('valid_volume_' + str (i), concated_img)
            break

        for test_vol in self.test_ds.get_data():
            affs, gt_affs, volume, wp, wn = self.pred(test_vol)
            affs, gt_affs, volume = sgm2img ([affs, gt_affs, volume])

            for i in range (volume.shape[0]):
                concated_img = [volume[i], affs[0,i], affs[2,i]]
                concated_img = np.concatenate (concated_img, 1)
                self.trainer.monitors.put_image ('test_volume_' + str (i), concated_img)
            break

        # Error block visualization
        error_pos = (22, 100, 0)
        error_block = self.test_ds.volume[0][error_pos[0]:error_pos[0]+16, error_pos[1]:error_pos[1]+256, error_pos[2]:error_pos[2]+256]
        error_block_affs = np.repeat (np.expand_dims (error_block, 0), 3, 0)
        
        affs, gt_affs, volume, wp, wn = self.pred([error_block, error_block, error_block_affs])
        affs, gt_affs, volume = sgm2img ([affs, gt_affs, volume])
        for i in range (volume.shape[0]):
            concated_img = [volume[i], affs[0,i], affs[2,i]]
            concated_img = np.concatenate (concated_img, 1)
            self.trainer.monitors.put_image ('error_volume_' + str (i), concated_img)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--load',    help='load model')
    args = parser.parse_args()
    model_path = None

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.load:
        model_path = args.load

    base_path = '/home/Pearl/tuan/_Data/SNEMI3D/DATA/'

    train_path = [base_path + 'ac4_images_cc.tif',
                    base_path + 'ac2_images_cc.tif',
                    base_path + 'train-input.tif']
    train_label_path = [base_path + 'ac4_labels.tif',
                        base_path + 'ac2_labels.tif',
                        base_path + 'train-labels.tif']

    test_path = [base_path + 'ac1_images_cc.tif']
    test_label_path = [base_path + 'ac4_labels.tif']

    X_train = read_im (train_path)
    y_train = read_im (train_label_path)
    X_test = read_im (test_path)
    y_test = read_im (test_label_path)

    X_valid = [X_train[0][75:100,:,:], X_train[1][120:156,:,:], X_train[2][75:100,:,:]]
    y_valid = [y_train[0][75:100,:,:], y_train[1][120:156,:,:], y_train[2][75:100,:,:]]
    X_train = [X_train[0][:80,:,:], X_train[1][:120,:,:], X_train[2][:80,:,:]]
    y_train = [y_train[0][:80,:,:], y_train[1][:120,:,:], y_train[2][:80,:,:]]

    # X_valid = [X_train[0][75:100,:,:]]
    # y_valid = [y_train[0][75:100,:,:]]
    # X_train = [X_train[0][:80,:,:]]
    # y_train = [y_train[0][:80,:,:]]


    save_dir = '/home/Pearl/tuan/_Data/SNEMI3D/train_log/256_models/'
    save_fol = os.path.basename(__file__)

    logger.set_logger_dir (save_dir + save_fol)
    # logger.set_logger_dir ('/home/Pearl/tuan/_Data/SNEMI3D/train_log/FusionNet_Aff_Softmax_on_pretrainedSquare')
    config = get_config (X_train, X_valid, y_train, y_valid, model_path)
    launch_train_with_config (config, SimpleTrainer ())


