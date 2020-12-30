from utils import resize
from dataset_tool import create_from_images  
class datasets:

    def __init__(self, path, dim = (512, 512)):
        self.path = path 
        self.dim = dim 
    
    def prepare(self, records_path):
        print('resizing images ...')
        out_path = resize(self.path, dim = self.dim)
        print('creating records ...')
        create_from_images(records_path, out_path, shuffle = True)

