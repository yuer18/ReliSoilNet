import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F
import warnings

from utils.utils import reshape_tensor, reshape_array, get_df_max_min, normalize, log_transform


class SNDataset(Dataset):
  def __init__(self, l8_dir, csv_dir, l8_bands: list = None, transform=None, return_point_id=False):
    self.l8_dir = l8_dir
    self.csv_dir = csv_dir
    self.l8_names = [f for f in os.listdir(l8_dir) if f.lower().endswith('.tif')]
    self.l8_names.sort()
    self.l8_bands = l8_bands if l8_bands else None
    self.transform = transform

    self.df = pd.read_csv(self.csv_dir)
    if 'Point_ID' not in self.df.columns:
      raise ValueError("CSV does not contain required column: 'Point_ID'")
    if 'OC' not in self.df.columns:
      raise ValueError("CSV does not contain required column: 'OC'")

    # LUCAS: Point_ID should be int-like
    self.df['Point_ID'] = pd.to_numeric(self.df['Point_ID'], errors='coerce')
    self.df = self.df.dropna(subset=['Point_ID'])
    self.df['Point_ID'] = self.df['Point_ID'].astype(int)
    self.df = self.df.set_index('Point_ID', drop=False)

    self.return_point_id = return_point_id

  def __len__(self):
    return len(self.l8_names)

  def __getitem__(self, index):
    l8_img_name = self.l8_names[index]
    l8_img_path = os.path.join(self.l8_dir, l8_img_name)

    point_id_str = l8_img_name.split('_')[0]
    try:
      point_id = int(point_id_str)
    except Exception as e:
      raise ValueError(f"Cannot parse Point_ID from filename '{l8_img_name}'. Got '{point_id_str}'.") from e

    if point_id not in self.df.index:
      raise KeyError(f"Point_ID={point_id} not found in CSV: {self.csv_dir}")

    row = self.df.loc[point_id]
    if isinstance(row, pd.DataFrame):  # duplicate Point_IDs
      row = row.iloc[0]

    oc = row['OC']
    if pd.isna(oc):
      raise ValueError(f"OC is NaN for Point_ID={point_id} in {self.csv_dir}")
    oc = float(oc)

    l8_img = io.imread(l8_img_path)
    if self.l8_bands:
      l8_img = l8_img[self.l8_bands, :, :]

    if self.transform:
      l8_img, oc = self.transform((l8_img, oc))

    if self.return_point_id:
      return l8_img, oc, str(point_id)
    else:
      return l8_img, oc


class SNDatasetClimate(Dataset):
  def __init__(self, l8_dir, csv_dir, climate_csv_folder,
               l8_bands: list = None, transform=None,
               dates=[
                 '20100101', '20100201', '20100301', '20100401', '20100501', '20100601',
                 '20100701', '20100801', '20100901', '20101001', '20101101', '20101201',
                 '20110101', '20110201', '20110301', '20110401', '20110501', '20110601',
                 '20110701', '20110801', '20110901', '20111001', '20111101', '20111201',
                 '20120101', '20120201', '20120301', '20120401', '20120501', '20120601',
                 '20120701', '20120801', '20120901', '20121001', '20121101', '20121201',
                 '20130101', '20130201', '20130301', '20130401', '20130501', '20130601',
                 '20130701', '20130801', '20130901', '20131001', '20131101', '20131201',
                 '20140101', '20140201', '20140301', '20140401', '20140501', '20140601',
                 '20140701', '20140801', '20140901', '20141001', '20141101', '20141201', '20150101'
               ],
               climate_dtype=torch.float32, normalize_climate=True, return_point_id=False):

    self.l8_dir = l8_dir
    self.csv_dir = csv_dir
    self.l8_names = [f for f in os.listdir(l8_dir) if f.lower().endswith('.tif')]
    self.l8_names.sort()
    self.l8_bands = l8_bands if l8_bands else None
    self.transform = transform

    self.df = pd.read_csv(self.csv_dir)
    if 'Point_ID' not in self.df.columns:
      raise ValueError("CSV does not contain required column: 'Point_ID'")
    if 'OC' not in self.df.columns:
      raise ValueError("CSV does not contain required column: 'OC'")

    # LUCAS int Point_ID
    self.df['Point_ID'] = pd.to_numeric(self.df['Point_ID'], errors='coerce')
    self.df = self.df.dropna(subset=['Point_ID'])
    self.df['Point_ID'] = self.df['Point_ID'].astype(int)
    self.df = self.df.set_index('Point_ID', drop=False)

    # Read climate CSV files (sorted for determinism)
    csv_files = sorted([
      f for f in os.listdir(climate_csv_folder)
      if os.path.isfile(os.path.join(climate_csv_folder, f)) and f.lower().endswith('.csv')
    ])
    if len(csv_files) == 0:
      raise ValueError(f"No climate CSV files found in: {climate_csv_folder}")

    self.climate_csv_files = csv_files
    self.clim_dfs = [pd.read_csv(os.path.join(climate_csv_folder, f)) for f in csv_files]

    # Normalize climate dfs (robust to NaNs in date columns)
    if normalize_climate:
      norm_clim = NormalizeClimDF(dates=dates, fill_strategy="interpolate_then_fill0", strict=False)
      self.clim_dfs = [norm_clim(clim_df, file_hint=csv_files[i]) for i, clim_df in enumerate(self.clim_dfs)]

    # index climate dfs by Point_ID for fast lookup
    for i in range(len(self.clim_dfs)):
      dfc = self.clim_dfs[i]
      if 'Point_ID' not in dfc.columns:
        raise ValueError(f"Climate CSV '{self.climate_csv_files[i]}' missing column 'Point_ID'")

      dfc['Point_ID'] = pd.to_numeric(dfc['Point_ID'], errors='coerce')
      dfc = dfc.dropna(subset=['Point_ID'])
      dfc['Point_ID'] = dfc['Point_ID'].astype(int)
      dfc = dfc.set_index('Point_ID', drop=False)
      self.clim_dfs[i] = dfc

    self.dates = [str(d) for d in dates]
    self.clim_dtype = climate_dtype
    self.return_point_id = return_point_id

  def __len__(self):
    return len(self.l8_names)

  def _get_row_dates_as_array(self, df: pd.DataFrame, point_id: int, file_hint: str):
    if point_id not in df.index:
      warnings.warn(f"[Climate] Point_ID={point_id} not found in climate file '{file_hint}'. Using zeros.")
      return np.zeros((len(self.dates),), dtype=np.float32)

    row = df.loc[point_id]
    if isinstance(row, pd.DataFrame):  # duplicates
      row = row.iloc[0]

    # ensure all date columns exist
    missing = [d for d in self.dates if d not in df.columns]
    if missing:
      raise ValueError(f"Climate file '{file_hint}' missing date columns: {missing[:5]}... (total {len(missing)})")

    vals = row[self.dates].to_numpy()
    vals = vals.astype(np.float32, copy=False)
    # final safety: any NaN -> 0
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return vals

  def __getitem__(self, index):
    l8_img_name = self.l8_names[index]
    l8_img_path = os.path.join(self.l8_dir, l8_img_name)

    point_id_str = l8_img_name.split('_')[0]
    try:
      point_id = int(point_id_str)
    except Exception as e:
      raise ValueError(f"Cannot parse Point_ID from filename '{l8_img_name}'. Got '{point_id_str}'.") from e

    if point_id not in self.df.index:
      raise KeyError(f"Point_ID={point_id} not found in main CSV: {self.csv_dir}")

    row = self.df.loc[point_id]
    if isinstance(row, pd.DataFrame):
      row = row.iloc[0]

    oc = row['OC']
    if pd.isna(oc):
      raise ValueError(f"OC is NaN for Point_ID={point_id} in {self.csv_dir}")
    oc = float(oc)

    # climate: shape (seq_len, n_features)
    clim_series_list = [
      self._get_row_dates_as_array(self.clim_dfs[i], point_id, self.climate_csv_files[i])
      for i in range(len(self.clim_dfs))
    ]
    clim_arr = np.stack(clim_series_list, axis=1)  # (T, F)

    l8_img = io.imread(l8_img_path)
    if self.l8_bands:
      l8_img = l8_img[self.l8_bands, :, :]

    if self.transform:
      l8_img, oc = self.transform((l8_img, oc))
      clim_arr = torch.tensor(clim_arr).to(dtype=self.clim_dtype)

    if self.return_point_id:
      return (l8_img, clim_arr), oc, str(point_id)
    else:
      return (l8_img, clim_arr), oc


#############################################################################################################
############################################# Transformations ###############################################
#############################################################################################################

class myNormalize:
  """Normalize the image and the target value"""
  def __init__(self, img_bands_min_max=[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min=0, oc_max=200):
    self.img_bands_min_max = img_bands_min_max
    self.oc_min = oc_min
    self.oc_max = oc_max

  def __call__(self, sample):
    img, oc = sample
    img = reshape_array(img)
    img[np.isnan(img)] = 0

    for band_min_max in self.img_bands_min_max:
      if band_min_max[1] != (0,1):
        if isinstance(band_min_max[0], tuple):
          img[band_min_max[0][0]:band_min_max[0][1]] = normalize(
            img[band_min_max[0][0]:band_min_max[0][1]],
            band_min_max[1][0], band_min_max[1][1]
          )
        elif isinstance(band_min_max[0], int):
          img[band_min_max[0]] = normalize(img[band_min_max[0]], band_min_max[1][0], band_min_max[1][1])
        else:
          raise ValueError('The first element of the tuple must be a tuple or an int')

    oc = normalize(oc, self.oc_min, self.oc_max)

    img[img > 1] = 1
    img[img < 0] = 0

    oc = oc if oc < 1 else 1
    oc = oc if oc > 0 else 0

    return img, oc


class myToTensor:
  def __init__(self, dtype=torch.float32, ouput_size=(64,64)):
    self.dtype = dtype
    self.resize = transforms.Resize(ouput_size)

  def __call__(self, sample):
    image, oc = sample
    return (
      self.resize(reshape_tensor(torch.from_numpy(image))).to(dtype=self.dtype),
      torch.tensor(oc).to(dtype=self.dtype)
    )


class Augmentations:
  """Data Augmentation Class"""
  def __init__(self, aug_prob=0.5, out_shape=(64,64)):
    self.aug_prob = aug_prob
    pad_h = out_shape[0] // 4
    pad_w = out_shape[1] // 4
    padding = (pad_w, pad_h, pad_w, pad_h)  # left, top, right, bottom

    self.aug = transforms.Compose([
      transforms.Pad(padding=padding, padding_mode='reflect'),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomRotation((0,90)),
      transforms.CenterCrop(size=out_shape),
    ])

  def __call__(self, sample):
    image, oc = sample
    return self.aug(image), oc


class RFTransform:
  def __init__(self, oc_max=87, oc_min=0):
    self.oc_max = oc_max
    self.oc_min = oc_min

  def __call__(self, sample):
    img, oc = sample
    img = reshape_array(img)
    oc = oc if oc < self.oc_max else self.oc_max
    oc = oc if oc > self.oc_min else self.oc_min
    img[np.isnan(img)] = 0
    return img, oc


class TensorCenterPixels:
  def __init__(self, pixel_radius=1, interpolate_center_pixel=False):
    self.pixel_radius = pixel_radius
    self.interpolate_center_pixel = interpolate_center_pixel

  def bilinear_interpolation(self, tensor):
    upsample = torch.nn.Upsample(
      size=(self.pixel_radius*2+1, self.pixel_radius*2+1),
      mode='bilinear',
      align_corners=True
    )
    resampled_tensor = upsample(tensor.unsqueeze(0)).squeeze(0)
    return resampled_tensor[:, self.pixel_radius:self.pixel_radius+1, self.pixel_radius:self.pixel_radius+1]

  def __call__(self, sample):
    image, oc = sample
    image = transforms.functional.center_crop(image, self.pixel_radius*2)
    if self.interpolate_center_pixel:
      image = self.bilinear_interpolation(image)
    return image, oc


class NormalizeClimDF:
  """
  Robust climate normalization:
  - only operates on date columns
  - fills NaNs in date columns by interpolation (time axis) then ffill/bfill then 0
  - safe min-max normalization
  """
  def __init__(self, dates, fill_strategy="interpolate_then_fill0", strict=False, eps=1e-12):
    self.dates = [str(d) for d in dates]
    self.fill_strategy = fill_strategy
    self.strict = strict
    self.eps = eps

  def __call__(self, df: pd.DataFrame, file_hint: str = ""):
    df = df.copy()
    df.columns = df.columns.map(str)

    missing = [d for d in self.dates if d not in df.columns]
    if missing:
      raise ValueError(f"Climate file '{file_hint}' missing required date columns: {missing[:5]}... (total {len(missing)})")

    # Only use date columns
    date_df = df[self.dates].copy()

    # to numeric + handle inf
    for c in self.dates:
      date_df[c] = pd.to_numeric(date_df[c], errors='coerce')
    date_df = date_df.replace([np.inf, -np.inf], np.nan)

    # fill NaNs in date columns
    if date_df.isna().values.any():
      if self.strict:
        raise ValueError(f"NaN found in climate date columns for file '{file_hint}' (strict=True).")
      # interpolate along time axis
      date_df = date_df.interpolate(axis=1, limit_direction='both')
      # fallback fills
      date_df = date_df.ffill(axis=1).bfill(axis=1)
      date_df = date_df.fillna(0.0)

    # normalize (global min-max over this file's date columns)
    X = date_df.to_numpy(dtype=np.float64, copy=False)
    vmin = np.min(X)
    vmax = np.max(X)
    if (vmax - vmin) < self.eps:
      Xn = np.zeros_like(X, dtype=np.float32)
    else:
      Xn = ((X - vmin) / (vmax - vmin)).astype(np.float32)

    df.loc[:, self.dates] = Xn
    return df


#############################################################################################################
#############################################      Tests      ###############################################
#############################################################################################################

if __name__ == "__main__":
  ds = SNDataset('ReliSoilNet\\dataset\\l8_images\\train\\', 'ReliSoilNet\\dataset\\LUCAS_2015.csv')
  print(len(ds))
  x = ds[0]
  print('OC: ', x[1], type(x[1]))
  print('image shape: ', x[0].shape, x[0].dtype)

  print("Testing the dataset with transforms...")
  mynorm = myNormalize()
  my_to_tensor = myToTensor()
  transform = transforms.Compose([mynorm, my_to_tensor])
  ds = SNDataset('ReliSoilNet\\dataset\\l8_images\\train\\', 'ReliSoilNet\\dataset\\LUCAS_2015.csv', transform=transform)
  rand = np.random.randint(0, len(ds))
  x = ds[rand]
  print('OC: ', x[1], type(x[1]))
  print('image shape: ', x[0].shape, x[0].dtype)
  print('image min: ', torch.min(x[0]), 'image max: ', torch.max(x[0]))

  print("Testing the dataset with transforms and Augmentations...")
  augment = Augmentations()
  aug_img = augment(x)
  print("Augmented image shape: ", aug_img[0].shape)


















