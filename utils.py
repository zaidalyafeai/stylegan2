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
  out_path = f'{path}/{dim[0]}x{dim[1]}'
  os.makedirs(out_path, exist_ok=True)
  for item in dirs:
    img_path = f'{path}/{item}'
    if os.path.isfile(img_path):
        im = Image.open(img_path)
        imResize = im.resize(dim, Image.ANTIALIAS).convert('RGB')
        imResize.save(f'{out_path}/{item}', 'JPEG', quality=90)
  return out_path