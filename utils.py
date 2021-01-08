import glob
import os 
from PIL import Image
import urllib.request
from tqdm import tqdm

import numpy as np
import PIL.Image
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import os

from IPython.display import HTML
from base64 import b64encode
import imageio

def show_animation(movie_name):
  mp4 = open(movie_name,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)

def imshow(a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        str_file = BytesIO()
        PIL.Image.fromarray(a).save(str_file, format)
        im_data = str_file.getvalue()
        try:
            disp = IPython.display.display(IPython.display.Image(im_data))
        except IOError:
            if jpeg_fallback and format != 'jpeg':
                print ('Warning: image was too large to display in format "{}"; '
                        'trying jpeg instead.').format(format)
                return imshow(a, format='jpeg')
            else:
                raise
        return disp

        
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def create_image_grid(images, scale=0.25, rows=1):
    w,h = images[0].size
    w = int(w*scale)
    h = int(h*scale)
    height = rows*h
    cols = ceil(len(images) / rows)
    width = cols*w
    canvas = PIL.Image.new('RGBA', (width,height), 'white')
    for i,img in enumerate(images):
        img = img.resize((w,h), PIL.Image.ANTIALIAS)
        canvas.paste(img, (w*(i % cols), h*(i // cols))) 
    return canvas

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
  for item in log_progress(dirs):
    img_path = f'{path}/{item}'
    if os.path.isfile(img_path):
        im = Image.open(img_path)
        imResize = im.resize(dim, Image.ANTIALIAS).convert('RGB')
        imResize.save(f'{out_path}/{item}', 'JPEG', quality=90)
  return out_path

def resize_dirs(path, out_dir, dim = (512, 512)):
  sub_dirs = os.listdir(path)
  for sub_dir in sub_dirs:
    out_path = f'{out_dir}/{sub_dir}'
    os.makedirs(out_path, exist_ok=True)
    for item in log_progress(os.listdir(f'{path}/{sub_dir}/')[:10]):
        img_path = f'{path}/{sub_dir}/{item}'
        if os.path.isfile(img_path):
            im = Image.open(img_path)
            imResize = im.resize(dim, Image.ANTIALIAS).convert('RGB')
            imResize.save(f'{out_path}/{item}', 'JPEG', quality=90)
  return out_dir

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# https://stackoverflow.com/a/53877507
def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# Taken from https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=1, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )