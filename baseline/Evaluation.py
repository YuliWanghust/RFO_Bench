import torch
import torchvision
import pandas as pd
import utils
import os
import matplotlib.pyplot as plt
import wget
import subprocess

from PIL import Image, ImageDraw
from engine import train_one_epoch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
data_dir = 'E:\\Yuli\\Projects\\ROF\\object-CXR\\'

labels_tr = pd.read_csv(data_dir + 'train.csv', na_filter=False)
labels_dev = pd.read_csv(data_dir + 'dev.csv', na_filter=False)
labels_test = pd.read_csv(data_dir + 'jhu_img_Dec12.csv', na_filter=False)
labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))
img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))
img_class_dict_test = dict(zip(labels_test.image_name, labels_test.annotation))

#Classes

class ForeignObjectDataset(object):

    def __init__(self, datafolder, datatype='train', transform=True, labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.labels_dict = labels_dict
        self.image_files_list = [s for s in sorted(os.listdir(datafolder)) if s in labels_dict.keys()]
        self.transform = transform
        self.annotations = [labels_dict[i] for i in self.image_files_list]

    def __getitem__(self, idx):
        # load images
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]

        if self.datatype == 'train':
            annotation = self.labels_dict[img_name]

            boxes = []

            if type(annotation) == str:
                annotation_list = annotation.split(';')
                for anno in annotation_list:
                    x = []
                    y = []

                    anno = anno[2:]
                    anno = anno.split(' ')
                    for i in range(len(anno)):
                        if i % 2 == 0:
                            x.append(float(anno[i]))
                        else:
                            y.append(float(anno[i]))

                    xmin = min(x) / width * 600
                    xmax = max(x) / width * 600
                    ymin = min(y) / height * 600
                    ymax = max(y) / height * 600
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transform is not None:
                img = self.transform(img)

            return img, target

        if self.datatype == 'dev':

            if self.labels_dict[img_name] == '':
                label = 0
            else:
                label = 1

            if self.transform is not None:
                img = self.transform(img)

            return img, label, width, height

        if self.datatype == 'test':

            if self.labels_dict[img_name] == '':
                label = 0
            else:
                label = 1

            if self.transform is not None:
                img = self.transform(img)

            return img, label, width, height

    def __len__(self):
        return len(self.image_files_list)


#Functions

def _get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


data_transforms = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset_train = ForeignObjectDataset(datafolder= data_dir + 'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict_tr)
dataset_dev = ForeignObjectDataset(datafolder= data_dir + 'dev/', datatype='dev', transform=data_transforms, labels_dict=img_class_dict_dev)
dataset_test = ForeignObjectDataset(datafolder= data_dir + 'jhu_img_Dec12/', datatype='test', transform=data_transforms, labels_dict=img_class_dict_test)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=8, shuffle= True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    dataset_dev, batch_size=1, shuffle= False, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle= False, num_workers=4,
    collate_fn=utils.collate_fn)

# Evaluation
def evaluation():
    model = _get_detection_model(num_classes)
    model.to(device)
    model.load_state_dict(torch.load("model_1.pt"))

    model.eval()

    preds = []
    labels = []
    locs_box = []
    locs = []

    for image, label, width, height in tqdm(data_loader_test):

        image = list(img.to(device) for img in image)
        labels.append(label[-1])

        outputs = model(image)

        vertice_points = []
        vertice_points_preds = []

        center_points = []
        center_points_preds = []

        if len(outputs[-1]['boxes']) == 0:
            preds.append(0)
            center_points.append([])
            center_points_preds.append('')
            locs_box.append('')
            locs.append('')
        else:
            preds.append(torch.max(outputs[-1]['scores']).tolist())

            new_output_index = torch.where((outputs[-1]['scores'] > 0.1))
            new_boxes = outputs[-1]['boxes'][new_output_index]
            new_scores = outputs[-1]['scores'][new_output_index]

            # calculate and output the bounding box (x1, y1, x2, y2)
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                #center_x = (new_box[0] + new_box[2]) / 2
                #center_y = (new_box[1] + new_box[3]) / 2
                #center_points.append([center_x / 600 * width[-1], center_y / 600 * height[-1]])
                vertice_points.append([new_box[0] / 600 * width[-1], new_box[1] / 600 * height[-1], new_box[2] / 600 * width[-1], new_box[3] / 600 * height[-1]])
            vertice_points_preds += new_scores.tolist()

            boxes = ''
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    boxes += str(vertice_points_preds[i]) + ' ' + str(vertice_points[i][0]) + ' ' + str(
                        vertice_points[i][1]) + ' ' + str(vertice_points[i][2]) + ' ' + str(vertice_points[i][3])
                else:
                    boxes += str(vertice_points_preds[i]) + ' ' + str(vertice_points[i][0]) + ' ' + str(
                        vertice_points[i][1]) + ' ' + str(vertice_points[i][2]) + ' ' + str(vertice_points[i][3]) + ';'
            locs_box.append(boxes)

            # calculate and output the center point (x,y)
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                center_x = (new_box[0] + new_box[2]) / 2
                center_y = (new_box[1] + new_box[3]) / 2
                center_points.append([center_x / 600 * width[-1], center_y / 600 * height[-1]])
            center_points_preds += new_scores.tolist()

            line = ''
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(
                        center_points[i][1])
                else:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(
                        center_points[i][1]) + ';'
            locs.append(line)

    cls_res = pd.DataFrame({'image_name': dataset_test.image_files_list, 'prediction': preds})
    cls_res.to_csv('classification_jhu.csv', columns=['image_name', 'prediction'], sep=',', index=None)
    print('classification.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset_test.image_files_list, 'prediction': locs_box})
    loc_res.to_csv('localization_jhu_boxes.csv', columns=['image_name', 'prediction'], sep=',', index=None)
    print('localization.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset_test.image_files_list, 'prediction': locs})
    loc_res.to_csv('localization_jhu.csv', columns=['image_name', 'prediction'], sep=',', index=None)
    print('localization.csv generated.')

if __name__ == '__main__':
    evaluation()

