from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_fn(label, pred):
    return nn.BCEWithLogitsLoss()(pred, label)


def score_fn(label, pred):
    return accuracy_score(label, (torch.sigmoid(pred) >= 0.5).float())


def train_lf(train_loader, model, optimizer, scheduler, accumulation_step, device):
    model.train()
    summary_loss = AverageMeter()
    for idx, (images, labels) in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      leave=False):
        images = torch.stack(images).to(device).float()
        labels = torch.stack(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(labels, outputs)
        loss.backward()

        summary_loss.update(loss.detach().item(), images.shape[0])

        if idx % accumulation_step == 0:
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return summary_loss.avg


def val_lf(valid_loader, model, device):
    model.eval()
    summary_loss = AverageMeter()

    list_label, list_pred = [], []
    for steps, (images, labels) in tqdm(enumerate(valid_loader),
                                        total=len(valid_loader),
                                        leave=False):
        with torch.no_grad():
            images = torch.stack(images).to(device).float()
            labels = torch.stack(labels).to(device).float()
            outputs = model(images)
            loss = loss_fn(labels, outputs)

            summary_loss.update(loss.detach().item(), images.shape[0])
            list_label.extend(labels.detach().cpu().numpy())
            list_pred.extend((torch.sigmoid(outputs) >= 0.5).float().detach().cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    score = f1_score(list_label, list_pred)

    return summary_loss.avg, score


def f1_display(df):
    df_max = df[df['is_max'] == 1].reset_index(drop=True)
    display(df_max)
    fig = px.scatter(df, x='thres_hold', y='f1_score', color='is_max')
    fig.add_annotation(x=df_max.thres_hold.mean(), y=df_max.f1_score[0],
                       text=f'{df_max.thres_hold.mean():.4f}: {df_max.f1_score[0]:.4f}',
                       showarrow=True, arrowhead=1)
    fig.show()


def F1_optimizer(df):
    true_label = df['label'].values
    score = df['classifier_score'].values
    list_thresh = np.arange(0, 1, 0.0001).tolist()
    list_f1 = []
    for thresh_hold in tqdm(list_thresh, total=len(list_thresh), leave=False):
        predict_label = np.where(score >= thresh_hold, 1, 0)
        f1 = f1_score(true_label, predict_label)
        list_f1.append(f1)
    f1_df = pd.DataFrame({'thres_hold': list_thresh, 'f1_score': list_f1})
    f1_df['is_max'] = f1_df['f1_score'].apply(lambda x: 1 if x == f1_df['f1_score'].max() else 0)
    f1_display(f1_df)