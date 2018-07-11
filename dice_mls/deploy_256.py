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
from scipy.ndimage.morphology import distance_transform_edt 
#################################################################################
import malis
from funcs import *
from model_snemi_dice_mls import *
from skimage import io
import os.path, math
from img_aug_func import *
#################################################################################


def build_w1 (shape):
	file_name = 'bleding_weight_256.tif'
	# if not (os.path.isfile (file_name)):
	# 	w = io.imread (file_name)
	# 	print 'Done loading bleding weight.'
	# 	return w
	factor = 1

	# cal_w = lambda ra, pa: pow (ra * (pa - ra), -factor)
	cal_w = lambda ra, pa: ra * (pa - ra)

	w = np.zeros (shape, dtype=np.float32)
	for z in range (shape[0]):
		wz = cal_w ((z + 1) * 16, (shape[0] + 1 ) * 16) 
		for y in range (shape[1]):
			wy = cal_w (y + 1, shape[1] + 1)
			for x in range (shape[2]):
				wx = cal_w (x + 1, shape[2] + 1)
				w[z][y][x] = wz + wx + wy
	print 'Done writting bleding weight.'
	io.imsave (file_name, w)
	return w

def build_w2 (shape):
	file_name = 'bleding_weight_256.tif'
	# if not (os.path.isfile (file_name)):
	# 	w = io.imread (file_name)
	# 	print 'Done loading bleding weight.'
	# 	return w
	factor = 1

	# cal_w = lambda ra, pa: pow (ra * (pa - ra), -factor)
	cal_w = lambda ra, pa: 1.0 - np.abs ( 0.5 - (1.0 * (pa - ra) / pa))

	w = np.zeros (shape, dtype=np.float32)
	for z in range (shape[0]):
		wz = cal_w ((z + 1) * 8, (shape[0] + 1 ) * 8) 
		for y in range (shape[1]):
			wy = cal_w (y + 1, shape[1] + 1)
			for x in range (shape[2]):
				wx = cal_w (x + 1, shape[2] + 1)
				# w[z][y][x] = 1.0 / np.exp (- math.pow (wz * wx * wy, 2.5) )
				w[z][y][x] = math.pow (wz * wx * wy, 6)
	print 'Done writting bleding weight.'
	io.imsave (file_name, w)
	ret = np.repeat (np.expand_dims (w, 0), 3, 0)
	ret[0,:,:,0] = 0;
	ret[:,0,:,1] = 0;
	ret[:,:,0,2] = 0;
	return ret


def bump_inference (vol_shape, shape, step, final_affs, weights_map):
	weights = build_w2 (shape)
	print 'Predicting.'
	for z0 in range (0, vol_shape[0], step[0]):
		for x0 in range (0, vol_shape[1], step[1]):
			for y0 in range (0, vol_shape[2], step[2]):

				if z0+shape[0] >= vol_shape[0]:
					z0 = vol_shape[0] - shape[0]
				if x0+shape[1] >= vol_shape[1]:
					x0 = vol_shape[1] - shape[1]
				if y0+shape[2] >= vol_shape[2]:
					y0 = vol_shape[2] - shape[2]

				print 'pos: ', z0, x0, y0

				vol = volume[z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]]
				pred = predict_func (vol)[0]

				mask = weights > weights_map[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]]
				mask = mask.astype (np.float32)
				pred *= mask

				weights_map[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]] = np.maximum (weights_map[:,z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]], weights)

				final_affs[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]] *= (1 - mask)
				final_affs[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]] += pred
			
				if z0+shape[0] > vol_shape[0] or x0+shape[1] > vol_shape[1] or y0+shape[2] > vol_shape[2]:
						break
	final_affs = final_affs.astype (np.float32)
	print 'Saving result.'
	np.save ('results/malis_wbce_' + dataset + '.npy', final_affs)
	print 'Done'

def average_inference (vol_shape, shape, step, final_affs, weights_map, aug=True, name='tmp'):
	weights = build_w2 (shape)
	print 'Predicting.'
	
	for z0 in range (0, vol_shape[0], step[0]):
		for x0 in range (0, vol_shape[1], step[1]):
			for y0 in range (0, vol_shape[2], step[2]):

				if z0+shape[0] >= vol_shape[0]:
					z0 = vol_shape[0] - shape[0]
				if x0+shape[1] >= vol_shape[1]:
					x0 = vol_shape[1] - shape[1]
				if y0+shape[2] >= vol_shape[2]:
					y0 = vol_shape[2] - shape[2]

				print 'pos: ', z0, x0, y0

				vol = volume[z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]]
				pred = np.ones ((3,) + vol.shape, dtype=np.float32)
				# aug_cnt = 0
				start_z = 0
				# z prefine
				if start_z == 1:
					pred[0] = 0
				if not aug:
					pred = predict_func (vol)[0]
				else:
					for rotn in range (4):
						for flipn in range (1, 5):
							# z prefine
							
							re_rotn = (4 - rotn) % 4
							re_flipn = flipn
							aug_img = square_rotate (flip (vol, flipn), rotn)
							aug_pred_affs  = predict_func (aug_img)[0]
							# aug_cnt += 1
							tmp = []
							for d in range (3):
							    tmp.append (flip (square_rotate (aug_pred_affs[d], re_rotn), re_flipn))
							pred_rev = np.array (tmp)
							# z prefine
							if start_z == 1:
								#pred[0] += pred_rev[0] / 16a
								if rotn == 0 and flipn == 4:
									pred [0] = pred_rev[0]
							pred[start_z:3] = np.minimum (pred[start_z:3], pred_rev[start_z:3])

				final_affs[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]] += np.multiply (pred, weights)
				weights_map[:, z0:z0+shape[0], x0:x0+shape[1], y0:y0+shape[2]] += weights
				
				if z0+shape[0] > vol_shape[0] or x0+shape[1] > vol_shape[1] or y0+shape[2] > vol_shape[2]:
						break

	weights_map [weights_map==0] = 1
	final_affs = np.divide (final_affs.astype (np.float32), weights_map)
	
	io.imsave ('weight_map.tif', weights_map.astype (np.float32))
	print 'Saving result.'
	np.save ('results/' + name + '_' + dataset + '.npy', final_affs)
	print 'Done'

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--load',    help='load model')
	parser.add_argument('--save',    help='load model')
	args = parser.parse_args()

	model_path = '/home/Pearl/tuan/_Data/SNEMI3D/train_log/256_models/train_snemi_dice_mls.py/model-150000'
	name = 'final_affs_pred'
	if args.load:
		model_path = args.load
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	nhood = malis.mknhood3d(1)
	model=Model ()

	predict_func = OfflinePredictor (PredictConfig (
        model = model,
        session_init = SaverRestore (model_path),
        input_names = ['volume'],
        output_names = ['affs']
    ))
	
	dataset = 'test'
	train_path = '/home/tuan/_Data/SNEMI3D/DATA/ac4_images_cc.tif'
	test_path = '/home/tuan/_Data/SNEMI3D/DATA/ac1_images_cc.tif'


	

	if dataset != 'test':
		volume = io.imread (train_path)
	else:
		volume = io.imread (test_path)


	vol_shape = volume.shape
	shape = (16, 256, 256)
	step = (7, 127, 127)


	final_affs = np.zeros ((3,) + vol_shape, dtype=np.float32)
	weights_map = np.zeros ((3,) + vol_shape, dtype=np.float32)

	average_inference (vol_shape, shape, step, final_affs, weights_map, name=model.name)
	# final_affs = np.divide (final_affs, weights_map)