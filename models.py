import argparse
import copy

import warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import sys, getopt, os

import numpy as np
import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil
from dnnlib.tflib.autosummary import autosummary

from training import misc
import pickle
import argparse

def create_model(config_id = 'config-f', gamma = None, height = 512, width = 512, cond = None, label_size = 0):
    train     = EasyDict(run_func_name='training.diagnostic.create_initial_pkl') # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    sched.minibatch_size_base = 192
    sched.minibatch_gpu_base = 3
    D_loss.gamma = 10
    desc = 'stylegan2'

    dataset_args = EasyDict() # (tfrecord_dir=dataset)

    if cond:
        desc += '-cond'; dataset_args.max_label_size = 'full' # conditioned on full label

    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        G.fmap_base = D.fmap_base = 8 << 10

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Gorig'   in config_id: G.architecture = 'orig'
        if 'Gskip'   in config_id: G.architecture = 'skip' # (default)
        if 'Gresnet' in config_id: G.architecture = 'resnet'
        if 'Dorig'   in config_id: D.architecture = 'orig'
        if 'Dskip'   in config_id: D.architecture = 'skip'
        if 'Dresnet' in config_id: D.architecture = 'resnet' # (default)

    # Configs A-D: Enable progressive growing and switch to networks that support it.
    if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
        sched.lod_initial_resolution = 8
        sched.G_lrate_base = sched.D_lrate_base = 0.001
        sched.G_lrate_dict = sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        sched.minibatch_size_base = 32 # (default)
        sched.minibatch_size_dict = {8: 256, 16: 128, 32: 64, 64: 32}
        sched.minibatch_gpu_base = 4 # (default)
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 8, 64: 4}
        G.synthesis_func = 'G_synthesis_stylegan_revised'
        D.func_name = 'training.networks_stylegan2.D_stylegan'

    # Configs A-C: Disable path length regularization.
    if config_id in ['config-a', 'config-b', 'config-c']:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns')

    # Configs A-B: Disable lazy regularization.
    if config_id in ['config-a', 'config-b']:
        train.lazy_regularization = False

    # Config A: Switch to original StyleGAN networks.
    if config_id == 'config-a':
        G = EasyDict(func_name='training.networks_stylegan.G_style')
        D = EasyDict(func_name='training.networks_stylegan.D_basic')

    if gamma is not None:
        D_loss.gamma = gamma

    G.update(resolution_h=height)
    G.update(resolution_w=width)
    D.update(resolution_h=height)
    D.update(resolution_w=width)

    sc.submit_target = dnnlib.SubmitTarget.DIAGNOSTIC
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    # [EDITED]
    kwargs.update(G_args=G, D_args=D, tf_config=tf_config, config_id=config_id,
        resolution_h=height, resolution_w=width, label_size = label_size)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_diagnostic(**kwargs)
    return f'network-initial-config-f-{height}x{width}-{label_size}.pkl'

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

def copy_weights(source_pkl, target_pkl, output_pkl):

    tflib.init_tf()

    with tf.Session():
        with tf.device('/gpu:0'):

            sourceG, sourceD, sourceGs = pickle.load(open(source_pkl, 'rb'))
            targetG, targetD, targetGs = pickle.load(open(target_pkl, 'rb'))
            
            # print('Source:')
            # sourceG.print_layers()
            # sourceD.print_layers() 
            # sourceGs.print_layers()
            
            # print('Target:')
            # targetG.print_layers()
            # targetD.print_layers() 
            # targetGs.print_layers()
            
            targetG.copy_compatible_trainables_from(sourceG)
            targetD.copy_compatible_trainables_from(sourceD)
            targetGs.copy_compatible_trainables_from(sourceGs)

            with open(os.path.join('./', output_pkl), 'wb') as file:
                pickle.dump((targetG, targetD, targetGs), file, protocol=pickle.HIGHEST_PROTOCOL)