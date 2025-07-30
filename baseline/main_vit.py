import torch
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import DetrImageProcessor, DetrForObjectDetection

np.random.seed(0)
torch.manual_seed(0)

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'E:\\Yuli\\Projects\\ROF\\Hopkins_RFO_Bench\\'

# Load annotations
labels_tr = pd.read_csv(data_dir + 'train.csv', na_filter=False)
labels_dev = pd.read_csv(data_dir + 'dev.csv', na_filter=False)

labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))
img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))


class ForeignObjectDataset(torch.utils.data.Dataset):
    def __init__(self, datafolder, datatype='train', transform=None, labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.labels_dict = labels_dict
        self.image_files_list = [s for s in sorted(os.listdir(datafolder)) if s in labels_dict]
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_name)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        if self.datatype == 'train':
            annotation = self.labels_dict[img_name]
            boxes = []
            if annotation:
                for anno in annotation.split(OBJECT_SEP):
                    anno = list(map(float, anno.split(ANNOTATION_SEP)[1:]))  # skip shape token
                    x = anno[::2]
                    y = anno[1::2]
                    xmin = min(x) / width
                    xmax = max(x) / width
                    ymin = min(y) / height
                    ymax = max(y) / height
                    boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # all foreground class = 1

            target = {
                'boxes': boxes,
                'labels': labels
            }

            return image, target

        elif self.datatype == 'dev':
            label = 1 if self.labels_dict[img_name] else 0
            return image, label, width, height


def main():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=2)
    model.to(device)

    dataset_train = ForeignObjectDataset(datafolder=data_dir + 'train/', datatype='train',
                                         labels_dict=img_class_dict_tr)
    dataset_dev = ForeignObjectDataset(datafolder=data_dir + 'dev/', datatype='dev',
                                       labels_dict=img_class_dict_dev)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4)
    dev_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=1, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 10
    auc_max = 0

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for image, target in pbar:
            image = image[0]  # single image
            encoding = processor(images=image, return_tensors="pt").to(device)

            boxes = target['boxes'].squeeze(0).to(device)
            class_labels = target['labels'].squeeze(0).to(device)

            target_dict = [{"class_labels": class_labels, "boxes": boxes}]
            outputs = model(**encoding, labels=target_dict)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for image, label, width, height in dev_loader:
                image = image[0]
                label = label[0].item()
                encoding = processor(images=image, return_tensors="pt").to(device)

                outputs = model(**encoding)
                results = processor.post_process_object_detection(outputs, target_sizes=[(height[0], width[0])])[0]

                score = max(results["scores"].tolist()) if results["scores"] else 0.0
                val_preds.append(score)
                val_labels.append(label)

        val_pred_labels = [1 if p >= 0.5 else 0 for p in val_preds]
        acc = np.mean([p == l for p, l in zip(val_pred_labels, val_labels)])
        auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

        if auc > auc_max:
            auc_max = auc
            print(f"✅ Best Epoch {epoch} | AUC: {auc:.4f}")
            model.save_pretrained("best_detr_model")
            processor.save_pretrained("best_detr_model")


if __name__ == '__main__':
    main()
