import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None
  loader = ImageLoader(dir_name,split='train')
  imageset = []
  for i in range(len(loader)):
    img, idx = loader[i]
    img = np.array(img).astype(np.float64)
    img *= 1.0/255.0
    imageset.append(img.flatten())
  loader = ImageLoader(dir_name,split='test')
  for i in range(len(loader)):
    img, idx = loader[i]
    img = np.array(img).astype(np.float64)
    img *= 1.0/255.0
    imageset.append(img.flatten())
  imageset = np.concatenate(imageset,axis=0).reshape(-1, 1) 

  scaler = StandardScaler()
  scaler.fit(imageset)
  print(scaler.mean_, scaler.scale_)
  return scaler.mean_, scaler.scale_

