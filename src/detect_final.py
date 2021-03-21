import __init__
from classification.network_classification import EfficientnetCus
from utils_yolo import *
from config import *
from classification.transform_classifier import aug
import cv2
import gc
import shutil as sh
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from ensemble_boxes import nms, weighted_boxes_fusion


def label_process(detect_result, iou_thresh, iou_thresh11):
    assert detect_result != ''
    x_center, y_center = detect_result[1::6], detect_result[2::6]
    w_center, h_center = detect_result[3::6], detect_result[4::6]
    detect_result[1::6] = [i - 0.5 * j for i, j in zip(x_center, w_center)]
    detect_result[2::6] = [i - 0.5 * j for i, j in zip(y_center, h_center)]
    detect_result[3::6] = [i + 0.5 * j for i, j in zip(x_center, w_center)]
    detect_result[4::6] = [i + 0.5 * j for i, j in zip(y_center, h_center)]
    list_new = []

    for label_values in np.unique(detect_result[::6]):
        list_values = np.array(
            [detect_result[6 * idx:6 * idx + 6] for idx, i in enumerate(detect_result[::6]) if i == label_values])
        boxes = list_values[:, 1:5].tolist()
        scores = list_values[:, 5].tolist()
        labels = list_values[:, 0].tolist()
        if label_values in [2, 11]:
            boxes, scores, labels = nms([boxes], [scores], [labels],
                                        weights=None, iou_thr=iou_thresh11)
        else:
            boxes, scores, labels = nms([boxes], [scores], [labels],
                                        weights=None, iou_thr=iou_thresh)

        for box in list_values:
            if box[-1] in scores:
                list_new.extend(box)
    return list_new


class Predict_process(object):
    def __init__(self, config):
        super(Predict_process, self).__init__()
        self.config = config

    def load_image(self, image_path, transforms):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image, (self.config.img_size, self.config.img_size))
        image = transforms(image=image)['image']
        image_resize = transforms(image=image_resize)['image']
        return image, image_resize

    def classifier_image(self, images):
        model = EfficientnetCus(model=self.config.model_use_classifier, num_class=1,
                                model_weight=self.config.weight_classifier_trained, is_train=False).to(
            self.config.device)
        model.eval()
        with torch.no_grad():
            outputs = model(images.to(self.config.device))
        return outputs

    def read_label(self, model, img_origin):
        os.makedirs(self.config.template, exist_ok=True)
        sh.copy(f'{self.config.source}/{img_origin}.png', f'{self.config.template}/{img_origin}.png')

        stride = int(model[0].stride.max())
        imgsz = check_img_size(self.config.img_size, s=stride)
        dataset = LoadImages(self.config.template, img_size=imgsz, stride=stride)
        path, img, im0s, vid_cap = next(iter(dataset))
        img = torch.from_numpy(img).to(self.config.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        detect_result = []
        for idx, weights in enumerate(self.config.list_weights):
            pred = model[idx](img, augment=self.config.augment)[0]
            pred = non_max_suppression(pred, self.config.conf_thresh[idx],
                                       self.config.iou_thresh,
                                       agnostic=self.config.agnostic_nms)
            for i, det in enumerate(pred):
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    for c in det[:, -1].unique():
                        (det[:, -1] == c).sum()
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        detect_result.append([cls.item(), *xywh, conf.item()])
        detect_result = sorted(detect_result, key=lambda x: (x[-1], x[0]), reverse=True)
        detect_result = [i for j in detect_result for i in j]
        detect_result = label_process(detect_result,
                                      self.config.iou_thresh,
                                      self.config.iou_thresh11)
        detect_result = [int(i) if idx % 6 == 0 else self.config.img_size * i if idx % 6 < 5 else i for idx, i in
                         enumerate(detect_result)]
        os.remove(f'{self.config.template}/{img_origin}.png')

        return detect_result

    @blockPrinting
    def fit(self, df, folder_image, use_classifier=True):
        global class_df
        transforms = aug('test')
        all_results = []
        dict_color = list_color(range(args.num_label+1))
        os.makedirs(args.save_test_path, exist_ok=True)
        if not use_classifier:
            class_df = pd.read_csv(KD_CLASSIFIER)
        model = [attempt_load(i, map_location=self.config.device) for i in self.config.list_weights]
        model = [i.half() for i in model]
        for images_id, images in tqdm(df.iterrows(), total=len(df), leave=False):
            fig, ax = plt.subplots(1, figsize=(15, 15))
            image_path = os.path.join(folder_image, images.image_id + '.png')
            if use_classifier:
                image, image_resize = self.load_image(image_path, transforms)
                class_labels = self.classifier_image(image_resize)
            else:
                class_labels = torch.tensor([class_df.loc[images_id, 'class1']])
            img_origin = df.loc[images_id, 'image_id']
            detect_result = self.read_label(model, img_origin)
            result_one_image = []
            im0 = cv2.imread(image_path)
            ax.imshow(im0, plt.cm.bone)
            if detect_result != '':
                img_sizes = [images.h, images.w]
                list_label = []
                for box_id in range(len(detect_result) // 6):
                    label, *box, score = detect_result[6 * box_id:6 * box_id + 6]
                    if class_labels.item() >= self.config.classification_thresh:
                        if (score > self.config.score_last) and \
                                not (np.min(box) < 10 or np.max(box[1::2]) > img_sizes[0] - 10 or np.max(box[::2]) > img_sizes[1] - 10) and \
                                not (label in [0, 3] and label in list_label) and \
                                not (label == 11 and score < self.config.score_11) and \
                                not (label == 9 and score < self.config.score_9) and \
                                not (label == 13 and score < self.config.score_13):
                            list_label.append(label)
                            box = label_resize((self.config.img_size, self.config.img_size), img_sizes, *box)
                            result_one_image.append(int(label))
                            result_one_image.append(np.round(score, 3))
                            result_one_image.extend([int(i) for i in box])

                            cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), dict_color[label], 3)
                            cv2.putText(im0, f'{label}: {score:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, dict_color[label], 3)
                    else:
                        if score > self.config.score_last2 and \
                                not (np.min(box) < 10 or np.max(box[1::2]) > img_sizes[0] - 10 or np.max(box[::2]) > img_sizes[1] - 10) and \
                                not (label in [0, 3] and label in list_label) and \
                                not (label == 11 and score < self.config.score_11) and \
                                not (label == 9 and score < self.config.score_9) and \
                                not (label == 13 and score < self.config.score_13):
                            list_label.append(label)
                            box = label_resize((self.config.img_size, self.config.img_size), img_sizes, *box)
                            result_one_image.append(int(label))
                            result_one_image.append(np.round(score, 3))
                            result_one_image.extend([int(i) for i in box])

                            cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), dict_color[label], 3)
                            cv2.putText(im0, f'{label}: {score:.3f}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, dict_color[label], 3)

            if len(result_one_image) == 0:
                all_results.append('14 1 0 0 1 1')
            else:
                result_str = ' '.join(map(str, result_one_image)) + f' 14 0 0 0 1 1'
                all_results.append(result_str)
            cv2.imwrite(os.path.join(self.config.save_test_path, img_origin+'.png'), im0)
        df['PredictionString'] = all_results
        df = df.drop(['h', 'w'], 1)
        os.rmdir(self.config.template)
        del model
        gc.collect()

        return df


if __name__ == '__main__':
    size_df = pd.read_csv(args.test_meta_path)
    size_df.columns = ['image_id', 'h', 'w']

    sub_df = pd.read_csv(args.sub_path)
    sub_df = sub_df.merge(size_df, on='image_id', how='left')

    test_list = glob(f'{args.test_original_path}/*.png')
    test_list_check = [i.split('/')[-1][:-4] for i in test_list]
    sub_df = sub_df[sub_df['image_id'].isin(test_list_check)].reset_index(drop=True)
    logger.info(f'Test have {len(test_list)} file')

    predict_pr = Predict_process(PredictConfig)
    sub_df = predict_pr.fit(sub_df, args.test_original_path, use_classifier=True)
    display(sub_df.head(60))
