
from training.run_training import run_training
from models import copy_weights, create_model
import os

class Model:
    def __init__(self, pkl_path = None, dim = (512, 512)):
        self.pkl_path = pkl_path
        self.dim = dim
        if self.pkl_path == None:
            ffhq = 'stylegan2-ffhq-config-f.pkl'
            pkl_path = f'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/{ffhq}'
            os.system(f'wget {pkl_path}')
            source_pkl = create_model()
            copy_weights(source_pkl, ffhq, self.pkl_path)

    def start_training(self, data_path, out_dir):
        run_training(data_path, out_dir, resume = self.pkl_path)
