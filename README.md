## SGAN
Easy impelementation of stylegans2. You can literally train a stylegan2 in less than 10 lines of code. Use this notebook for a quick start 
<a href="https://colab.research.google.com/github/zaidalyafeai/sgan/blob/master/demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" width = '100px' ></a>.

## Tranining 
In just a few lines you can use style tranfer or train a stylegan from scratch. 

```python 

from datasets import Dataset
from sgan import SGAN

dataset = Dataset('/path/to/dataset')
dataset.prepare('path/to/records')

model = SGAN()
model.train(dataset = 'path/to/records', out_dir = 'path/to/out')

```

## Visualization 
A set of functions for vis, interpolation and animation. Mostly tested in colab notebooks. 

### Load Model 
```python 
from sgan import SGAN
model = SGAN(pkl_path = '/path/to/pkl')
```

### Generate random 
```python 
sgan.generate_randomly()
```

### Generate grid 
```python 
model.generate_grid()
```

### Generate animation 
```python
model.generate_animation(size = 2, steps = 20)
```

## Sample Models 

### Mosaics 
![alt text](mosaic.png)

### Medusas
![alt text](medusa.png)

### Cats 
![alt text](cats.png)

## References 
- Gan-surgery: https://github.com/aydao/stylegan2-surgery
- WikiArt model: https://github.com/pbaylies/stylegan2 
- Starter-Notebook: https://github.com/Hephyrius/Stylegan2-Ada-Google-Colab-Starter-Notebook/