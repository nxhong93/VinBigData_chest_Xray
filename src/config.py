import os
import logging
import torch
import albumentations as al
from albumentations.pytorch import ToTensorV2, ToTensor
import argparse

logging.basicConfig(format='%(asctime)s +++ %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

os.environ["WANDB_API_KEY"] = '8f435998b1a6f9a4e59bfaef1deed81c1362a97d'
os.environ["WANDB_MODE"] = "dryrun"

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
inputdir = os.path.join(parentdir, 'input')
modeldir = os.path.join(parentdir, 'model')

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--num_label', type=int, default=14, help='Number of label unique')
parser.add_argument('--img_size', type=int, default=1024, help='Size of image in training classifier')
parser.add_argument('--model_use_classifier', type=str, default='b3', help='Model classifier name')
parser.add_argument('--train_meta_path', type=str, default=os.path.join(inputdir, 'train_meta.csv'), help='Original size image train classifier')
parser.add_argument('--test_meta_path', type=str, default=os.path.join(inputdir, 'test_meta.csv'), help='Original size image test')
parser.add_argument('--train_path', type=str, default=os.path.join(inputdir, 'train.csv'), help='Train file csv')
parser.add_argument('--sub_path', type=str, default=os.path.join(inputdir, 'sample_submission.csv'), help='Submission file csv')
parser.add_argument('--train_classifier_path', type=str, default=os.path.join(inputdir, 'image/train'), help='Train image file classifier')
parser.add_argument('--train_original_path', type=str, default=os.path.join(inputdir, 'image/train'), help='Train image file with original size')
parser.add_argument('--test_original_path', type=str, default=os.path.join(inputdir, 'image/test'), help='Test image file with original size')
parser.add_argument('--save_path', type=str, default=modeldir, help='path image save')
parser.add_argument('--save_test_path', type=str, default=os.path.join(inputdir, 'image/save_test'), help='test image save')
parser.add_argument('--yolo_pretrained_path', type=str, default=os.path.join(modeldir, 'best.pt'), help='yolo model weight trained')
parser.add_argument('--weight_classifier_trained', type=str, default=os.path.join(modeldir, 'model_classifier.pth'), help='Pretrained model classifier')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='gpu or cpu')
parser.add_argument('--yolo_create_path', type=str, default=os.path.join(inputdir, 'chest_yolo'), help='gpu or cpu')


args = parser.parse_args()


class DetectConfig:
    yolo_create_path = args.yolo_create_path
    train_original_path = args.train_original_path


class ClassifierConfig:
    img_size = args.img_size
    model_use = args.model_use_classifier
    save_path = args.save_path
    weight_classifier_trained = args.weight_classifier_trained
    device = args.device
    accumulation_step = 1
    fold_num = 5
    batch_size = 1
    predict_batch_size = 1*batch_size
    num_workers = 12
    seed = 23
    lr = 1e-3
    n_epochs = 5
    verbose = 1
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.2,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-7
    )


class PredictConfig:
    model_use_classifier = args.model_use_classifier
    weight_classifier_trained = args.weight_classifier_trained
    img_size = args.img_size
    device = args.device
    template = os.path.join(currentdir, 'detect')
    list_weights = [args.yolo_pretrained_path]
    source = args.test_original_path
    save_test_path = args.save_test_path
    batch_size = 4
    score_thresh = 0.05
    iou_thresh = 0.2
    iou_thresh2 = 0.1
    iou_thresh11 = 0.0001
    skip_thresh = 0.0001
    score_last = 0.0
    score_last2 = 0.95
    score_9 = 0.1
    score_11 = 0.015
    score_13 = 0.01
    classification_thresh = 1e-4
    augment = True
    conf_thresh = [0.005]
    agnostic_nms = True
