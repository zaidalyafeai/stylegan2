import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import runway
import os

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    global Gs
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        _G, _D, Gs = pickle.load(file, encoding='latin1')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    rnd = np.random.RandomState()
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    return Gs


generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'label': runway.number(min=0, max=1000000, default=22, step=1), # generate random labels
    'scale': runway.number(min=-1, max=1, default=0.25, step=0.05),  # magnitude of labels - 0 = no labels
    'truncation': runway.number(min=-1.5, max=1.5, default=1, step=0.1)
}

@runway.command('generate', inputs=generate_inputs, outputs={'image': runway.image})
def convert(model, inputs):
    z = inputs['z']
    label = int(inputs['label'])
    scale = inputs['scale']
    truncation = inputs['truncation']
    latents = z.reshape((1, 512))
    labels = scale * np.random.RandomState(label).randn(167)
    labels = labels.reshape((1,167)).astype(np.float32)
    images = model.run(latents, labels, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return {'image': output}


if __name__ == '__main__':
    runway.run()
