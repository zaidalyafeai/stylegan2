import glob
import os 
from PIL import Image
import urllib.request
from tqdm import tqdm

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

