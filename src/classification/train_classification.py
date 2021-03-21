import __init__
from config import *
from network_classification import EfficientnetCus
from dataset_classification import ChestClassifierDataset
from transform_classifier import aug
from engineer_classification import *
import random
import gc
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
from IPython.display import display
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, sampler


class Train_process(object):

    def __init__(self, config=ClassifierConfig):
        super(Train_process, self).__init__()
        self.config = config

    def split_df(self, df):
        kf = MultilabelStratifiedKFold(n_splits=self.config.fold_num, shuffle=True, random_state=self.config.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df,
                                                             df.iloc[:, 1:train_df['class_id'].nunique() + 1])):
            df[f'fold_{fold}'] = 0
            df.loc[val_idx, f'fold_{fold}'] = 1
        return df

    def process_data(self, df, folder_image, fold_idx, is_train=True):
        if is_train:
            train_data = df[df[f'fold_{fold_idx}'] == 0].reset_index(drop=True)
            val_data = df[df[f'fold_{fold_idx}'] == 1].reset_index(drop=True)
            # Create dataset
            train_dataset = ChestClassifierDataset(train_data, folder_image,
                                                   transform=aug('train'), is_train=is_train)
            valid_dataset = ChestClassifierDataset(val_data, folder_image,
                                                   transform=aug('validation'),
                                                   is_train=is_train)
            # Create dataloader
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.batch_size,
                                      collate_fn=train_dataset.collate_fn,
                                      shuffle=True, pin_memory=True, drop_last=True,
                                      num_workers=self.config.num_workers)
            valid_loader = DataLoader(valid_dataset, pin_memory=True,
                                      batch_size=self.config.batch_size,
                                      collate_fn=valid_dataset.collate_fn, drop_last=True,
                                      num_workers=self.config.num_workers)

            del train_data, val_data, train_dataset, valid_dataset
            gc.collect()
            return train_loader, valid_loader
        else:
            test_dataset = ChestClassifierDataset(df, folder_image,
                                                  transform=aug('test'), is_train=is_train)
            test_loader = DataLoader(test_dataset, pin_memory=True, shuffle=False,
                                     batch_size=self.config.predict_batch_size,
                                     collate_fn=test_dataset.collate_fn,
                                     num_workers=self.config.num_workers)
            del test_dataset
            gc.collect()
            return test_loader

    def fit(self, df, folder_image):
        self.split_df(df)
        fold = random.choice(np.arange(self.config.fold_num))
        print(50 * '-')
        print(f'Fold_{fold}:')
        model = EfficientnetCus(self.config.model_use, 1, None).to(self.config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        scheduler = self.config.SchedulerClass(optimizer, **self.config.scheduler_params)
        train_loader, valid_loader = self.process_data(df, folder_image, fold)
        best_val_loss = np.Inf
        best_val_score = 0
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        list_f1_val = []
        for epoch in range(self.config.n_epochs):
            train_loss = train_lf(train_loader, model, optimizer, scheduler, self.config.accumulation_step,
                                  self.config.device)
            val_loss, f1_val = val_lf(valid_loader, model, self.config.device)
            list_f1_val.append(f1_val)
            print(f'Epoch{epoch}: Train_loss: {train_loss:.5f} | Val_loss: {val_loss:.5f} | F1_score: {f1_val:.5f}')

            if best_val_score < f1_val:
                best_val_loss = val_loss
                best_val_score = f1_val
                torch.save(model.state_dict(), f'{self.config.save_path}/model_classifier.pth')
                print('Model improved, saving model!')
            elif best_val_score == f1_val:
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{self.config.save_path}/model_classifier.pth')
                    print('Model improved, saving model!')

            if self.config.validation_scheduler:
                scheduler.step(val_loss)

        f1_df = pd.DataFrame([np.arange(len(list_f1_val)), list_f1_val], index=['epoch', 'f1_score']).T
        f1_df['is_max'] = f1_df['f1_score'].apply(lambda x: 1 if x == f1_df['f1_score'].max() else 0)

        fig = px.line(f1_df, x='epoch', y='f1_score')
        fig.add_annotation(x=f1_df[f1_df.is_max == 1]['epoch'].mean(), y=f1_df.f1_score.max(),
                           text=f'{f1_df[f1_df.is_max == 1].epoch.mean():.1f}: {f1_df.f1_score.max():.4f}',
                           showarrow=True, arrowhead=1)
        fig.show()
        torch.cuda.empty_cache()

    def fit_predict(self, df, folder_image):
        model = EfficientnetCus(self.config.model_use, 1,
                                self.config.weight_classifier_trained, is_train=False).to(self.config.device)
        model.eval()
        test_loader = self.process_data(df, folder_image, None, is_train=False)
        list_pred = []
        for _, images in tqdm(test_loader, total=len(test_loader), leave=False):
            images = torch.stack(images).to(self.config.device).float()
            with torch.no_grad():
                outputs = model(images, images.shape[0]).detach().cpu().numpy()
                list_pred.extend(outputs)
        df['classifier_score'] = list_pred

        return df


if __name__ == '__main__':
    size_df = pd.read_csv(args.train_meta_path)
    size_df.columns = ['image_id', 'h', 'w']
    train_df = pd.read_csv(args.train_path)
    train_df = train_df.merge(size_df, on='image_id', how='left')
    train_df[['x_min', 'y_min']] = train_df[['x_min', 'y_min']].fillna(0)
    train_df[['x_max', 'y_max']] = train_df[['x_max', 'y_max']].fillna(1)

    class_each_image = pd.pivot_table(train_df, columns='class_id', index='image_id',
                                      values='rad_id', aggfunc='count', fill_value=0).reset_index()
    class_each_image['label'] = class_each_image[14].apply(lambda x: 1 if x == 0 else 0)
    class_each_image = class_each_image[:100]
    train_list = glob(f'{args.train_classifier_path}/*.png')
    logger.info(f'Train have {len(train_list)} file')

    train_pr = Train_process(ClassifierConfig)
    # train_pr.fit(class_each_image, train_list)
    class_each_image = train_pr.fit_predict(class_each_image, train_list)
    display(class_each_image.head())
    F1_optimizer(class_each_image)
