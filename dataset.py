import os,sys
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

class DatasetFromCsv(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None, transform_resize=None, soft_label=False, linear_loss=False):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode

        # ! meta features
        self.use_meta = False if meta_features is None else True
        if self.use_meta: 
            print ('use meta features (age etc...)')
        self.meta_features = meta_features

        self.transform = transform
        self.transform_resize = transform_resize
        self.soft_label = soft_label

        self.linear_loss = linear_loss

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index] # name,path,label,person_id,row_num,fold
  
        image = cv2.imread(row.path) # ! read a file path from @csv, so don't need different folders, but have to reindex labels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_resize = 0 # can't return none in class Data
        if self.transform_resize is not None: 
            res = self.transform_resize(image=image) # ! just resize, nothing more, need this to plot later
            image_resize = res['image'].astype(np.uint8) # https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

        # ! transform image for model
        res = self.transform(image=image) # ! uses albumentations
        image = res['image'].astype(np.float32)
        image = image.transpose(2, 0, 1) # makes channel x h x w instead of h x w x c

        if self.use_meta:
            temp_ = self.csv.iloc[index][self.meta_features].to_numpy().flatten().tolist()
            temp_ = [float(t) for t in temp_]
            data = (torch.tensor(image).float(), torch.tensor(temp_).float()) # name or index for @self.meta_features ? 
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data, row.path, image_resize # ! return path so we can debug and plot attribution model ?
        else:
            label_ = 0 
            if self.soft_label: 
                label_ = np.fromstring(self.csv.iloc[index].target_soft, dtype=float, sep=';') # ! @target_soft must exist
            # 
            # ! @data is tuple if use meta-data
            if self.linear_loss: 
                target = torch.tensor(self.csv.iloc[index].target).float()
            else: 
                target = torch.tensor(self.csv.iloc[index].target).long()
            return data, target, torch.FloatTensor(label_), row.path, image_resize 


def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    transforms_resize = albumentations.Compose([ # ! meant to used with attribution
        albumentations.Resize(image_size, image_size)
    ])

    return transforms_train, transforms_val, transforms_resize


def strong_aug(image_size): # https://github.com/albumentations-team/albumentations_examples
    return  albumentations.Compose([
            albumentations.RandomRotate90(),
            albumentations.Flip(),
            albumentations.Transpose(),
            albumentations.OneOf([
                albumentations.IAAAdditiveGaussianNoise(),
                albumentations.GaussNoise(),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.2),
                albumentations.MedianBlur(blur_limit=3, p=0.1),
                albumentations.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            albumentations.OneOf([
                albumentations.OpticalDistortion(p=0.3),
                albumentations.GridDistortion(p=0.1),
                albumentations.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
                albumentations.RandomBrightnessContrast(),
            ], p=0.3),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize()
        ])


def get_df(csv_name, soft_label=False, args=None):

    df = pd.read_csv(csv_name,na_values="-") 

    # ! replace na with 0 ? 
    if args.use_meta or args.linear_loss: 
        print ('\nhandle missing data by fill 0 or just remove\n')
        print ('before size :',df.shape[0])
        if args.fillna0: 
            print ('fill na')
            df = df.fillna(0)
        else: 
            print ('drop na')
            df = df.dropna()
        df = df.reset_index(drop=True)
        print ('after size :',df.shape[0])
    
    # class mapping
    if not args.linear_loss:
        diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df.label.unique()))}
        print ('diagnosis2idx {}'.format(diagnosis2idx))

        df['target'] = df['label'].map(diagnosis2idx)
        
        if soft_label: 
            df['target_soft'] = df['softlabel'] # @softlabel should be string of prob 0.1;0.2;....

        # count each disease in train 
        print ( 'df train labels {}'.format ( df.groupby('label').count() ) )
        our_label_index = list (np.arange(len(diagnosis2idx))) # ! to subset labels wanted later, here, we want all labels

    elif args.linear_loss: 
        if args.new_label is not None: 
            df['label'] = df[args.new_label].values
        #
        df['target'] = df['label'].values
        diagnosis2idx = None 
        our_label_index = None

    # need is_ext, may be useful later
    df['is_ext'] = 0
    
    return df, diagnosis2idx, our_label_index

