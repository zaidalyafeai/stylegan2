import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base

def create_initial_pkl(
    G_args                  = {},         # Options for generator network.
    D_args                  = {},         # Options for discriminator network.
    tf_config               = {},         # Options for tflib.init_tf().
    config_id               = "config-f", # config-f is the only one tested ...
    num_channels            = 3,          # number of channels, e.g. 3 for RGB
    resolution_h            = 1024,       # height dimension of real/fake images
    resolution_w            = 1024,       # height dimension of real/fake images 
    label_size              = 0,          # number of labels for a conditional model
    ):   

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)

    resolution = resolution_h # training_set.shape[1]

    # Construct or load networks.
    with tf.device('/gpu:0'):
        print('Constructing networks...')
        G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, **G_args)
        D = tflib.Network('D', num_channels=num_channels, resolution=resolution, label_size=label_size, **D_args)
        Gs = G.clone('Gs')

    # Print layers and generate initial image snapshot.
    G.print_layers(); D.print_layers()
    pkl = 'network-initial-%s-%sx%s-%s.pkl' % (config_id, resolution_w, resolution_h, label_size)
    misc.save_pkl((G, D, Gs), pkl)
    print("Saving",pkl)