import __init__
from src.config import *
from src.utils_yolo import *
from src.yolov5 import *
import os
import shutil as sh
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import collections
from IPython.display import display
import argparse
import logging
import math
import random
import time


def create_file(df, split_df, train_file, train_folder, fold):
    os.makedirs(f'{train_file}/labels/train/', exist_ok=True)
    os.makedirs(f'{train_file}/images/train/', exist_ok=True)
    os.makedirs(f'{train_file}/labels/val/', exist_ok=True)
    os.makedirs(f'{train_file}/images/val/', exist_ok=True)

    list_image_train = split_df[split_df[f'fold_{fold}'] == 0]['image_id']
    train_df = df[df['image_id'].isin(list_image_train)].reset_index(drop=True)
    val_df = df[~df['image_id'].isin(list_image_train)].reset_index(drop=True)

    for train_img in tqdm(train_df.image_id.unique()):
        with open(f'{train_file}/labels/train/{train_img}.txt', 'w+') as f:
            row_df = train_df[train_df['image_id'] == train_img]
            row_df['x_center'] = row_df['x_center'] / row_df['w']
            row_df['y_center'] = row_df['y_center'] / row_df['h']
            row_df['width'] = row_df['width'] / row_df['w']
            row_df['height'] = row_df['height'] / row_df['h']
            row = row_df[['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{train_img}.png',
                f'{train_file}/images/train/{train_img}.png')

    for val_img in tqdm(val_df.image_id.unique()):
        with open(f'{train_file}/labels/val/{val_img}.txt', 'w+') as f:
            row_df = val_df[val_df['image_id'] == val_img]
            row_df['x_center'] = row_df['x_center'] / row_df['w']
            row_df['y_center'] = row_df['y_center'] / row_df['h']
            row_df['width'] = row_df['width'] / row_df['w']
            row_df['height'] = row_df['height'] / row_df['h']
            row = row_df[['class_id', 'x_center', 'y_center', 'width', 'height']].values
            row = row.astype('str')
            for box in range(len(row)):
                text = ' '.join(row[box])
                f.write(text)
                f.write('\n')
        sh.copy(f'{train_folder}/{val_img}.png',
                f'{train_file}/images/val/{val_img}.png')


if __name__ == '__main__':
    size_df = pd.read_csv(args.train_meta_path)
    size_df.columns = ['image_id', 'h', 'w']
    train_df = pd.read_csv(args.train_path)
    train_df = train_df.merge(size_df, on='image_id', how='left')
    train_df[['x_min', 'y_min']] = train_df[['x_min', 'y_min']].fillna(0)
    train_df[['x_max', 'y_max']] = train_df[['x_max', 'y_max']].fillna(1)

    train_abnormal = Abnormal_filter(train_df)

    fold_csv = split_df(train_abnormal, config=DetectConfig)
    fold_csv = fold_csv.merge(size_df, on='image_id', how='left')

    create_file(train_abnormal, fold_csv, DetectConfig.yolo_create_path, DetectConfig.train_original_path,
                DetectConfig.fold_train)