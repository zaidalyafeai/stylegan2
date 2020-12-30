
from training.run_training import run_training
from models import copy_weights, create_model
from utils import download_url
import os

class Model:
    def __init__(self, pkl_path = None, from_scratch= False, dim = (512, 512)):
        self.pkl_path = pkl_path
        self.dim = dim
        if self.pkl_path == None:
            ffhq = 'stylegan2-ffhq-config-f.pkl'
            pkl_path = f'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/{ffhq}'
            
            if not os.path.exists(ffhq):
                download_url(pkl_path, ffhq)
                
            self.pkl_path = 'surgery.pkl'
            if not os.path.exists(self.pkl_path):
                source_pkl = create_model(height=dim[0], width=dim[1])
                copy_weights(ffhq, source_pkl, self.pkl_path)
            if from_scratch:
                self.pkl_path = ffhq

    def start_training(self, data_path, out_dir):
        run_training(data_path, out_dir, resume = self.pkl_path)
