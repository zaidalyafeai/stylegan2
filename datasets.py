from utils import resize, resize_dirs
from dataset_tool import create_from_images, create_from_image_folders

class Dataset:
    def __init__(self, path, dim = (512, 512)):
        self.path = path 
        self.dim = dim 
    
    def prepare(self, tfrecord_dir, shuffle = True,
                with_sub_dirs = False, ignore_labels = 1):
        if with_sub_dirs:
            out_path = resize_dirs(self.path, 'dataset/resized', dim = self.dim)
            create_from_image_folders(tfrecord_dir, out_path, shuffle, ignore_labels)
        else:
            print('resizing images ...')
            out_path = resize(self.path, dim = self.dim)
            print('creating records ...')
            create_from_images(tfrecord_dir, out_path, shuffle = True)

