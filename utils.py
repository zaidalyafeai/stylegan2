import glob
import os 
from PIL import Image
import urllib.request
from tqdm import tqdm_notebook

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



class DownloadProgressBar(tqdm_notebook):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# https://stackoverflow.com/a/53877507
def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)