import pydicom
import sys
import os
import random
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def img_size(path):
    information = pydicom.dcmread(path)
    h, w = information.Rows, information.Columns
    return h, w


def label_resize(org_size, img_size0, *bbox):
    x0, y0, x1, y1 = bbox
    x0_new = int(np.round(x0 * img_size0[1] / org_size[1]))
    y0_new = int(np.round(y0 * img_size0[0] / org_size[0]))
    x1_new = int(np.round(x1 * img_size0[1] / org_size[1]))
    y1_new = int(np.round(y1 * img_size0[0] / org_size[0]))
    return x0_new, y0_new, x1_new, y1_new


def list_color(class_list, use_rectangle=False):
    dict_color = dict()
    for classid in class_list:
        if use_rectangle:
            dict_color[classid] = [i / 256 for i in random.sample(range(256), 3)]
        else:
            dict_color[classid] = tuple([i for i in random.sample(range(256), 3)])

    return dict_color


def split_df(df, config):
    kf = MultilabelStratifiedKFold(n_splits=config.fold_num, shuffle=True, random_state=config.seed)
    df['id'] = df.index
    annot_pivot = pd.pivot_table(df, index=['image_id'], columns=['class_id'],
                                 values='id', fill_value=0, aggfunc='count') \
        .reset_index().rename_axis(None, axis=1)
    for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                         annot_pivot.iloc[:, 1:(1 + df['class_id'].nunique())])):
        annot_pivot[f'fold_{fold}'] = 0
        annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
    return annot_pivot


def Preprocess_wbf(df, size, iou_thr=0.5, skip_box_thr=0.0001):
    list_image = []
    list_boxes = []
    list_cls = []
    list_h, list_w = [], []
    new_df = pd.DataFrame()
    for image_id in tqdm(df['image_id'].unique(), leave=False):
        image_df = df[df['image_id'] == image_id].reset_index(drop=True)
        h, w = image_df.loc[0, ['h', 'w']].values
        boxes = image_df[['x_min_resize', 'y_min_resize',
                          'x_max_resize', 'y_max_resize']].values.tolist()
        boxes = [[j / (size - 1) for j in i] for i in boxes]
        scores = [1.0] * len(boxes)
        labels = [float(i) for i in image_df['class_id'].values]
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels],
                                                      weights=None,
                                                      iou_thr=iou_thr,
                                                      skip_box_thr=skip_box_thr)
        list_image.extend([image_id] * len(boxes))
        list_h.extend([h] * len(boxes))
        list_w.extend([w] * len(boxes))
        list_boxes.extend(boxes)
        list_cls.extend(labels.tolist())
    list_boxes = [[int(j * (size - 1)) for j in i] for i in list_boxes]
    new_df['image_id'] = list_image
    new_df['class_id'] = list_cls
    new_df['h'] = list_h
    new_df['w'] = list_w
    new_df['x_min_resize'], new_df['y_min_resize'], new_df['x_max_resize'], new_df['y_max_resize'] = np.transpose(list_boxes)
    new_df['x_center'] = 0.5 * (new_df['x_min_resize'] + new_df['x_max_resize'])
    new_df['y_center'] = 0.5 * (new_df['y_min_resize'] + new_df['y_max_resize'])
    new_df['width'] = new_df['x_max_resize'] - new_df['x_min_resize']
    new_df['height'] = new_df['y_max_resize'] - new_df['y_min_resize']
    new_df['area'] = new_df.apply(
        lambda x: (x['x_max_resize'] - x['x_min_resize']) * (x['y_max_resize'] - x['y_min_resize']), axis=1)

    return new_df


def Abnormal_filter(train_df):
    train_normal = train_df[train_df['class_name'] == 'No finding'].reset_index(drop=True)
    train_normal['x_min_resize'] = 0
    train_normal['y_min_resize'] = 0
    train_normal['x_max_resize'] = 1
    train_normal['y_max_resize'] = 1

    train_abnormal = train_df[train_df['class_name'] != 'No finding'].reset_index(drop=True)
    train_abnormal[['x_min_resize', 'y_min_resize', 'x_max_resize', 'y_max_resize']] = train_abnormal \
        .apply(lambda x: x[['x_min', 'y_min', 'x_max', 'y_max']].values, axis=1, result_type="expand")
    train_abnormal['x_center'] = 0.5 * (train_abnormal['x_min_resize'] + train_abnormal['x_max_resize'])
    train_abnormal['y_center'] = 0.5 * (train_abnormal['y_min_resize'] + train_abnormal['y_max_resize'])
    train_abnormal['width'] = train_abnormal['x_max_resize'] - train_abnormal['x_min_resize']
    train_abnormal['height'] = train_abnormal['y_max_resize'] - train_abnormal['y_min_resize']
    train_abnormal['area'] = train_abnormal.apply(
        lambda x: (x['x_max_resize'] - x['x_min_resize']) * (x['y_max_resize'] - x['y_min_resize']), axis=1)

    return train_abnormal


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


def save_image_test(df, size_df, list_image):
    dict_color = list_color(range(15))
    image_row_random = np.random.choice(len(df), num_image, replace=(len(df) < num_image))
    for image_idx in image_row_random:
        image_id, pred = df.loc[image_idx, 'image_id'], df.loc[image_idx, 'PredictionString']
        org_size = size_df[size_df['image_id'] == image_id][['h', 'w']].values[0].tolist()
        fig, ax = plt.subplots(1, figsize=(15, 15))
        img_path = [i for i in list_image if image_id in i][0]
        img = cv2.imread(img_path)
        ax.imshow(img, plt.cm.bone)
        if pred != '14 1 0 0 1 1':
            list_pred = pred.split(' ')
            for box_idx in range(len(list_pred) // 6)[:-1]:
                bbox = map(int, list_pred[6 * box_idx + 2:6 * box_idx + 6])
                x_min, y_min, x_max, y_max = label_resize(org_size, IMG_SIZE, *bbox)
                class_name, score = int(list_pred[6 * box_idx]), float(list_pred[6 * box_idx + 1])
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=1, edgecolor=dict_color[class_name], facecolor='none')
                ax.add_patch(rect)
                plt.text(x_min, y_min, f'{class_name}: {score}', fontsize=15, color='red')

        plt.title(image_id)
        plt.show()
