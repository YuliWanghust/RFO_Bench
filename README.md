<!--img src="logo.png" width="100%"/-->

- [Background](#background)
- [Data](#data)
- [Annotation](#annotation)
- [Download](#download)
- [Evaluation](#evaluation)
- [Baseline](#baseline)
- [Submission](#submission)
- [Leaderboard](#leaderboard)
- [Organizers](#organizers)


## Background
Critical retained foreign objects (RFOs), including surgical instruments like sponges and needles, pose serious patient safety risks and carry significant financial and legal implications for healthcare institutions. Detecting critical RFOs using artificial intelligence remains challenging due to their rarity and the limited availability of chest X-ray datasets that specifically feature critical RFOs cases. Existing datasets only contain non-critical RFOs, like necklace or zipper, further limiting their utility for developing clinically impactful detection algorithms. To address these limitations, we introduce "Hopkins RFO Bench" a novel dataset containing 144 chest X-ray images of critical RFO cases collected over 18 years from the Johns Hopkins Health System. Using this dataset, we benchmark several state-of-the-art object detection models, highlighting the need for enhanced detection methodologies for critical RFO cases. Recognizing data scarcity challenges, we further explore image synthesis methods to bridge this gap. We evaluate two advanced synthetic image methods—DeepDRR-RFO, a physics-based method, and RoentGen-RFO, a diffusion-based method—for creating realistic radiographs featuring critical RFOs. Our comprehensive analysis identifies the strengths and limitations of each synthetic method, providing insights into effectively utilizing synthetic data to enhance model training. The Hopkins RFO Bench and our findings significantly advance the development of reliable, generalizable AI-driven solutions for detecting critical RFOs in clinical chest X-rays.

In this project, we 
- Developed and open-access dataset, called [Hopkins RFOs Bench], the first and largest dataset of its kind, comprising 144 chest X-ray images containing critical RFOs collected from the Johns Hopkins Health System over the past 18 years.
- Benchmarking existing object detection models on our proposed dataset.
- Evaluating two customized synthetic image generation models, [DeepDRR-RFO]() and [RoentGen-RFO](), for creating images with critical RFOs. We train object detection models using these synthetic images, analyze the strengths and weaknesses of each approach, and provide valuable insights to guide future improvements utilizing our openly accessible dataset.

## Data
5000 frontal chest X-ray images with foreign objects presented and 5000 frontal chest X-ray images without foreign objects were filmed and collected from about 300 township hosiptials in China. 12 medically-trained radiologists with 1 to 3 years of experience annotated all the images. Each annotator manually annotates the potential foreign objects on a given chest X-ray presented within the lung field. Foreign objects were annotated with bounding boxes, bounding ellipses or masks depending on the shape of the objects. Support devices were excluded from annotation. A typical frontal chest X-ray with foreign objects annotated looks like this:
![annotation](annotation.png)
We randomly split the 10000 images into training, validation and test dataset:

**training** 4000 chest X-rays with foreign objects presented; 4000 chest X-rays without foreign objects. 

**validation** 500 chest X-rays with foreign objects presented; 500 chest X-rays without foreign objects. 

**test** 500 chest X-rays with foreign objects presented; 500 chest X-rays without foreign objects. 


## Annotation

We provide object-level annotations for each image, which indicate the rough location of each foreign object using a closed shape.

Annotations are provided in csv files and a csv example is shown below.

```csv
image_path,annotation
/path/#####.jpg,ANNO_TYPE_IDX x1 y1 x2 y2;ANNO_TYPE_IDX x1 y1 x2 y2 ... xn yn;...
/path/#####.jpg,
/path/#####.jpg,ANNO_TYPE_IDX x1 y1 x2 y2
...
```

Three type of shapes are used namely rectangle, ellipse and polygon. We use `0`, `1` and `2` as `ANNO_TYPE_IDX` respectively.

- For rectangle and ellipse annotations, we provide the bounding box (upper left and lower right) coordinates in the format `x1 y1 x2 y2` where `x1` < `x2` and `y1` < `y2`.

- For polygon annotations, we provide a sequence of coordinates in the format `x1 y1 x2 y2 ... xn yn`.

> ### Note:
> Our annotations use a Cartesian pixel coordinate system, with the origin (0,0) in the upper left corner. The x coordinate extends from left to right; the y coordinate extends downward.

## Download
The training and validation dataset can be accessed here at [Google Drive](https://drive.google.com/drive/folders/1SubfNALJn6aO56lUYeJsVpFLZuXurlBC). The imaging data has been
anonymousized and free to download for scientific research and non-commercial usage.


## Evaluation
We use two metrics to evaluate the classification and localization performance of foreign objects detection on chest X-rays: Area Under Curve (AUC) and  Free-response Receiver Operating Characteristic (FROC).

### Classification
For the test dataset, each algorithm is required to generate a `prediction_classification.csv` file in the format below:
```
image_path,prediction
/path/#####.jpg,0.90
/path/#####.jpg,0.85
/path/#####.jpg,0.15
...
```
where each line corresponds to the prediciton result of one image. The first column is the image path, the second column is the predicted probability, ranging from 0 to 1, indicating whether this image has foregin objects or not.

We use AUC to evaluate the algorithm performance of classifying whether each given chest X-ray has foreign objects presented or not. AUC is commonly used to evaluate binary classification in medical imaging challenges. For example, the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) competition for chest X-ray classification, and the classification task within [CAMELYON16](https://camelyon16.grand-challenge.org/Evaluation/) challenge for whole slide imaging classification. We believe AUC is adequate enough to measure the performance of the classification task of our challenge, especially given our positive and negative data is balanced.

### Localization
For the test dataset, each algorithm is required to generate a `prediction_localization.csv` file in the format below:
```
image_path,prediction
/path/#####.jpg,0.90 1000 500;0.80 200 400
/path/#####.jpg,
/path/#####.jpg,0.75 300 600;0.50 400 200;0.15 1000 200
...
```
where each line corresponds to the prediciton result of one image. The first column is the image path, the second column
is space seperated 3-element tuple of predicted foreign object coordinates with its probability in the format of (probability x y), where x and y are the width and height coordinates of the predicted foreign object. It is allowed to have zero predicted 3-element tuple for certain images, if there are no foreign objects presented. But please note the `,` after the first column even if the prediction is empty.

We use FROC to evaluate the algorithm performance of localizing foreign obects on each given chest X-ray. Because our object annotations are provided in different format, i.e. boxes, ellipses or masks depending on radiologists' annotation habits, it's not suitable to use other common metric, such as mean average precision (mAP) in natural object detection with pure bounding box annotations. FROC is more suitable in this case, since only localization coordinates are required for evaluation. FROC has been used as the metric for measuring medical imaging localization, for example, the lesion localization task within [CAMELYON16](https://camelyon16.grand-challenge.org/Evaluation/) challenge, and tuberculosis localization in (EJ Hwang, 2019, Clinical Infectious Disease).

FROC is computed as follow. A foregin object is counted as detected as long as one predicted cooridinate lies within its annotation. The sensitivity is the number of detected foreign objects dividide by the number of total foreign objects. A predicted coordinate is false positive, if it lies outside any foreign object annotation. When the numbers of false positive coordinates per image are 0.125, 0.25, 0.5, 1, 2, 4, 8, FROC is the average sensitivty of these different versions of predictions. 



[froc.py](https://github.com/jfhealthcare/object-CXR/tree/master/froc.py) provides the details of how FROC is computed.


## Baseline

We provide a baseline result in this [Jupyter Notebook](https://github.com/jfhealthcare/object-CXR/tree/master/baseline/baseline.ipynb).

## Submission
We host the online submission system at [codalab](https://worksheets.codalab.org/worksheets/0xcd2fb3db8ae74d03b53ad4c5bf81ebe2)

## Leaderboard

| Rank    | Date |  Model  | AUC| FROC|
| ------- | -----| --------| ---| ----| 
| 1       |Feb 21, 2020 | [JF Healthcare baseline](https://github.com/jfhealthcare/object-CXR#baseline) | 0.9209 |0.8031|

## Organizers
[JF Healthcare](http://www.jfhealthcare.com/) is the primary organizer of this challenge

<img src="jf.png" width="50%" align="middle"/>

