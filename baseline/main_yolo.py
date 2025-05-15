import os
import pandas as pd
from PIL import Image
import yaml
import subprocess

# Define global paths
data_root = 'E:/Yuli/Projects/ROF/object-CXR'
train_img_dir = os.path.join(data_root, 'images/train')
val_img_dir = os.path.join(data_root, 'images/val')
train_csv = os.path.join(data_root, 'train.csv')
val_csv = os.path.join(data_root, 'dev.csv')
train_label_dir = os.path.join(data_root, 'labels/train')
val_label_dir = os.path.join(data_root, 'labels/val')
yaml_path = os.path.join(data_root, 'data.yaml')

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

def convert_annotation_to_yolo(img_name, annotation_str, img_w, img_h):
    yolo_labels = []
    for poly in annotation_str.split(OBJECT_SEP):
        coords = list(map(float, poly.split(ANNOTATION_SEP)[1:]))
        x = coords[::2]
        y = coords[1::2]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        yolo_labels.append(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
    return yolo_labels

def convert_csv_to_yolo(csv_file, img_dir, label_dir):
    df = pd.read_csv(csv_file)
    df = df[df['annotation'].astype(bool)].reset_index(drop=True)
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_name'])
        img = Image.open(img_path)
        w, h = img.size
        yolo_labels = convert_annotation_to_yolo(row['image_name'], row['annotation'], w, h)
        label_path = os.path.join(label_dir, os.path.splitext(row['image_name'])[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

def create_data_yaml(train_dir, val_dir, yaml_path):
    data = {
        'train': os.path.abspath(train_dir),
        'val': os.path.abspath(val_dir),
        'nc': 1,
        'names': ['object']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_yolov5(data_yaml='object-CXR/data.yaml', weights='yolov5s.pt', epochs=20):
    subprocess.run([
        'python', 'yolov5/train.py',
        '--img', '640',
        '--batch', '16',
        '--epochs', str(epochs),
        '--data', data_yaml,
        '--weights', weights,
        '--project', 'runs/train',
        '--name', 'yolov5_rfo',
        '--exist-ok'
    ])

if __name__ == '__main__':
    convert_csv_to_yolo(train_csv, train_img_dir, train_label_dir)
    convert_csv_to_yolo(val_csv, val_img_dir, val_label_dir)
    create_data_yaml(train_img_dir, val_img_dir, yaml_path)
    train_yolov5(data_yaml=yaml_path)
