import torch
import torchvision
import numpy as np
import os
import pandas as pd

import utils
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from engine import train_one_epoch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


np.random.seed(0)
torch.manual_seed(0)

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

data_dir = 'E:\\Yuli\\Projects\\ROF\\object-CXR\\'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2  # object (foreground); background

def draw_annotation(im, anno_str, fill=(255, 63, 63, 40)):
    draw = ImageDraw.Draw(im, mode="RGBA")
    for anno in anno_str.split(OBJECT_SEP):
        anno = list(map(int, anno.split(ANNOTATION_SEP)))
        if anno[0] == 0:
            draw.rectangle(anno[1:], fill=fill)
        elif anno[0] == 1:
            draw.ellipse(anno[1:], fill=fill)
        else:
            draw.polygon(anno[1:], fill=fill)

labels_tr = pd.read_csv(data_dir + 'train.csv', na_filter=False)
labels_dev = pd.read_csv(data_dir + 'dev.csv', na_filter=False)

labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))
img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))

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

    def __len__(self):
        return len(self.image_files_list)

def main():

    data_transforms = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = ForeignObjectDataset(datafolder=data_dir + 'train/', datatype='train', transform=data_transforms,
                                         labels_dict=img_class_dict_tr)
    dataset_dev = ForeignObjectDataset(datafolder=data_dir + 'dev/', datatype='dev', transform=data_transforms,
                                       labels_dict=img_class_dict_dev)

    def _get_detection_model(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_dev, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    model_ft = _get_detection_model(num_classes)
    model_ft.to(device)

    params = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    num_epochs = 20
    auc_max = 0

    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=20)
        lr_scheduler.step()

        model_ft.eval()
        val_pred = []
        val_label = []
        for batch_i, (image, label, width, height) in enumerate(data_loader_val):
            image = list(img.to(device) for img in image)

            val_label.append(label[-1])

            outputs = model_ft(image)
            if len(outputs[-1]['boxes']) == 0:
                val_pred.append(0)
            else:
                val_pred.append(torch.max(outputs[-1]['scores']).tolist())

        val_pred_label = []
        for i in range(len(val_pred)):
            if val_pred[i] >= 0.5:
                val_pred_label.append(1)
            else:
                val_pred_label.append(0)

        number = 0

        for i in range(len(val_pred_label)):
            if val_pred_label[i] == val_label[i]:
                number += 1
        acc = number / len(val_pred_label)

        auc = roc_auc_score(val_label, val_pred)
        print('Epoch: ', epoch, '| val acc: %.4f' % acc, '| val auc: %.4f' % auc)

        if auc > auc_max:
            auc_max = auc
            print('Best Epoch: ', epoch, '| val acc: %.4f' % acc, '| Best val auc: %.4f' % auc_max)
            torch.save(model_ft.state_dict(), "model_1.pt")

if __name__ == '__main__':
    main()

