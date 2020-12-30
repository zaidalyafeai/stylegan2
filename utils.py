import glob
import os 
from PIL import Image

def find_latest_pkl(path):
  curr_best = 0
  latest_pkl = ''
  for pkl in glob.glob(f'{path}/*.pkl'):
    ckpt_number = int(pkl.split('-')[-1][:-4])
    if curr_best < ckpt_number:
      curr_best = ckpt_number
      latest_pkl = pkl
  return latest_pkl

def resize(path, dim = (512, 512)):
  dirs = os.listdir(path)
  out_path = f'{dim[0]}x{dim[1]}'
  for item in dirs:
    if os.path.isfile(path+item):
        im = Image.open(path+item)
        imResize = im.resize(dim, Image.ANTIALIAS).convert('RGB')
        imResize.save(f'{path}/{out_path}/{item}', 'JPEG', quality=90)
  return out_path